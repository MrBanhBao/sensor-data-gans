import numpy as np
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.utils import shuffle
from tensorflow.keras.models import load_model
import seaborn as sns

from utils.plotting import plot_n_lineplots, plot_n_heatmaps
from utils.preprocess import filter_by_activity_index, calc_consultant
from utils.windowing import transform_windows_df, windowing_dataframe


class GanEvaluator:
    def __init__(self, generator_file, act_id, labels=['standing', 'walking', 'jogging']):
        self.act_id = act_id
        self.latent_dim = None
        self.generator = self.__load_model(generator_file)
        self.labels = labels

        # Data Parameters
        self.x_train, self.y_train = None, None
        self.x_train_clf, self.y_train_clf = None, None
        self.x_test_clf, self.y_test_clf = None, None

        # Data Processing
        self.window_size = None
        self.step_size = None
        self.col_names = None
        self.method = None
        self.input_cols_eval = None

        # eval
        self.best_train_f1_score = None
        self.orig_train_acc = None
        self.orig_train_f1_score = None

        self.best_test_f1_score = None
        self.orig_test_acc = None
        self.orig_test_f1_score = None

    def __load_model(self, generator_file):
        model = load_model(generator_file)
        self.latent_dim = model.input.shape[1]
        return model

    def init_data(self, train_path, test_path,
                  window_size=5 * 50,
                  step_size=int(5 * 50 / 2),
                  col_names=['userAcceleration.x', 'userAcceleration.y', 'userAcceleration.z', 'userAcceleration.c'],
                  method='sliding',
                  input_cols_train=['userAcceleration.x', 'userAcceleration.y', 'userAcceleration.z'],
                  input_cols_eval=['userAcceleration.c']
                  ):
        self.window_size = window_size
        self.step_size = step_size
        self.col_names = col_names
        self.method = 'sliding'
        self.input_cols_eval = input_cols_eval

        print('Load Data...')
        train_df = pd.read_hdf(train_path)
        test_df = pd.read_hdf(test_path)

        train_windowed_df = windowing_dataframe(train_df, window_size=window_size, step_or_sample_size=step_size,
                                                col_names=col_names, method=method)
        test_windowed_df = windowing_dataframe(test_df, window_size=window_size, step_or_sample_size=step_size,
                                               col_names=col_names, method=method)

        print('Transform Data...')
        self.x_train, self.y_train = transform_windows_df(train_windowed_df, input_cols=input_cols_train,
                                                          one_hot_encode=False,
                                                          as_channel=False)

        x_train_clf, self.y_train_clf = transform_windows_df(train_windowed_df, input_cols=input_cols_eval,
                                                             one_hot_encode=False, as_channel=False)
        x_test_clf, self.y_test_clf = transform_windows_df(test_windowed_df, input_cols=input_cols_eval,
                                                           one_hot_encode=False, as_channel=False)

        self.x_train_clf = x_train_clf.reshape((len(x_train_clf), window_size))
        self.x_test_clf = x_test_clf.reshape((len(x_test_clf), window_size))

        self.x_train_activity, _ = filter_by_activity_index(x=self.x_train, y=self.y_train, activity_idx=self.act_id)

        print('Calculate origin performance...')
        self.calc_origin_train_test_performance(True)
        print('Done!')

    def calc_origin_train_test_performance(self, verbose=False):
        if (self.x_train_clf is None) or (self.x_test_clf is None):
            print('Please run method: init_data first.')
            return

        orig_svm_clf = SVC()
        orig_svm_clf.fit(self.x_train_clf, self.y_train_clf)

        y_train_head = orig_svm_clf.predict(self.x_train_clf)
        self.orig_train_acc = accuracy_score(self.y_train_clf, y_train_head)
        self.orig_train_f1_score = f1_score(self.y_train_clf, y_train_head, average=None)[self.act_id]
        self.best_train_f1_score = self.orig_train_f1_score

        y_test_head = orig_svm_clf.predict(self.x_test_clf)
        self.orig_test_acc = accuracy_score(self.y_test_clf, y_test_head)
        self.orig_test_f1_score = f1_score(self.y_test_clf, y_test_head, average=None)[self.act_id]
        self.best_test_f1_score = self.orig_test_f1_score

        if verbose:
            print('Original training acc: ', self.orig_train_acc)
            print('Original training f1_score for act_id ', self.act_id, ': ', self.orig_train_f1_score, '\n')
            print('Original test acc: ', self.orig_test_acc)
            print('Original test f1_score for act_id ', self.act_id, ': ', self.orig_test_f1_score, '\n')

    def generate_data(self, percentage=0.1):
        num_gen = int(np.ceil(len(self.x_train_activity) * percentage))
        random_latent_vectors = np.random.normal(size=(num_gen, self.latent_dim))
        self.generated_sensor_data = self.generator.predict(random_latent_vectors)

        gen_df = pd.DataFrame(np.array([ts.transpose() for ts in self.generated_sensor_data]).tolist(),
                              columns=['userAcceleration.x', 'userAcceleration.y', 'userAcceleration.z'])
        gen_df['userAcceleration.c'] = calc_consultant(gen_df)
        gen_df['act'] = self.act_id

        gen_windowed_df = windowing_dataframe(gen_df, window_size=self.window_size, step_or_sample_size=self.step_size,
                                              col_names=self.col_names, method=self.method)

        input_cols = ['userAcceleration.c']
        x_gen, y_gen = transform_windows_df(gen_windowed_df, input_cols=input_cols, one_hot_encode=False,
                                            as_channel=False)
        x_gen = x_gen.reshape((len(x_gen), self.window_size))

        x_train_gen = np.concatenate([self.x_train_clf, x_gen[:num_gen]])
        y_train_gen = np.concatenate([self.y_train_clf, y_gen[:num_gen]])

        return x_train_gen, y_train_gen, num_gen

    def eval_performance(self, x_train_gen, y_train_gen, verbose=False):
        gen_svm_clf = SVC()
        gen_svm_clf.fit(x_train_gen, y_train_gen)

        y_train_head = gen_svm_clf.predict(self.x_train_clf)
        gen_train_acc = accuracy_score(self.y_train_clf, y_train_head)
        gen_train_f1_score = f1_score(self.y_train_clf, y_train_head, average=None)[self.act_id]

        y_test_head = gen_svm_clf.predict(self.x_test_clf)
        gen_test_acc = accuracy_score(self.y_test_clf, y_test_head)
        gen_test_f1_score = f1_score(self.y_test_clf, y_test_head, average=None)[self.act_id]

        if gen_train_f1_score > self.best_train_f1_score or gen_test_f1_score > self.best_test_f1_score:
            self.gen_svm_clf = gen_svm_clf

        if verbose:
            print('Train Acc:', self.orig_train_acc < gen_train_acc, self.orig_train_f1_score < gen_train_f1_score)
            print('Origin: ', self.orig_train_acc, ' vs. ', gen_train_acc)
            print('F1-Origin', self.orig_train_f1_score, ' vs. ', gen_train_f1_score)
            print('\n')
            print('Test Performance', self.orig_test_acc < gen_test_acc, self.orig_test_f1_score < gen_test_f1_score)
            print('Origin: ', self.orig_test_acc, ' vs. ', gen_test_acc)
            print('F1-Origin', self.orig_test_f1_score, ' vs. ', gen_test_f1_score)
            print('\n')
            print('\n')

        return gen_train_f1_score, gen_test_f1_score

    def plot_line_plot(self, num=10, random_state=None):
        if self.generated_sensor_data is None:
            print('Run method generate_data first.')
            return

        x_train_activity = shuffle(self.x_train_activity, random_state=random_state)
        generated_sensor_data = shuffle(self.generated_sensor_data, random_state=random_state)

        return plot_n_lineplots(x_train_activity, generated_sensor_data, n=num)

    def plot_heat_maps(self, num=10, random_state=None):
        if self.generated_sensor_data is None:
            print('Run method generate_data first.')
            return

        x_train_activity = shuffle(self.x_train_activity, random_state=random_state)
        generated_sensor_data = shuffle(self.generated_sensor_data, random_state=random_state)

        return plot_n_heatmaps(x_train_activity, generated_sensor_data, n=num)

    def train_classification_report(self):
        y_train_head = self.gen_svm_clf.predict(self.x_train_clf)
        print(classification_report(self.y_train_clf, y_train_head))

    def test_classification_report(self):
        y_test_head = self.gen_svm_clf.predict(self.x_test_clf)
        print(classification_report(self.y_test_clf, y_test_head))

    def train_cm(self):
        y_train_head = self.gen_svm_clf.predict(self.x_train_clf)
        cm = confusion_matrix(self.y_train_clf, y_train_head)
        cm_df = pd.DataFrame(cm, index=self.labels,
                             columns=self.labels)
        return sns.heatmap(cm_df, annot=True, cmap='YlGnBu', fmt='g')

    def test_cm(self):
        y_test_head = self.gen_svm_clf.predict(self.x_test_clf)
        cm = confusion_matrix(self.y_test_clf, y_test_head)
        cm_df = pd.DataFrame(cm, index=self.labels,
                             columns=self.labels)
        return sns.heatmap(cm_df, annot=True, cmap='YlGnBu', fmt='g')