import os
import numpy as np
import pandas as pd
from sklearn.svm import SVC
from sklearn.utils import shuffle
from sklearn.metrics import accuracy_score, f1_score
from tensorflow.python.keras.models import Sequential
from tensorflow.keras.optimizers import Adam

from gans.utils import smooth_labels
from utils.preprocess import filter_by_activity_index, calc_consultant
from utils.utils import check_dir_creation
from utils.windowing import windowing_dataframe, transform_windows_df


class GanTrainer:
    def __init__(self, out_dir):
        check_dir_creation(out_dir)
        self.out_dir = out_dir

        # Gan Parameters
        self.generator = None
        self.discriminator = None
        self.gan = None

        self.latent_dim = None

        # Data Parameters
        self.x_train, self.y_train = None, None
        self.x_val, self.y_val = None, None
        self.x_test, self.y_test = None, None

        self.x_train_clf, self.y_train_clf = None, None
        self.x_val_clf, self.y_val_clf = None, None
        self.x_test_clf, self.y_test_clf = None, None

        self.x_train_activity = None

        # Data Processing
        self.act_id = None
        self.window_size = None
        self.step_size = None
        self.col_names = None
        self.method = None
        self.input_cols_train = None
        self.input_cols_eval = None

        # eval
        self.best_train_f1_score = 0.0
        self.best_val_f1_score = 0.0

        self.orig_svm_clf = None
        self.orig_train_acc = None
        self.orig_train_f1_score = None

        self.orig_val_acc = None
        self.orig_val_f1_score = None

    def set_out_dir(self, out_dir):
        check_dir_creation(out_dir)
        self.out_dir = out_dir

    def create_gan(self, generator, discriminator, optimizer=Adam()):
        self.latent_dim = generator.input.shape[1]

        self.generator = generator
        self.discriminator = discriminator
        self.discriminator.trainable = False

        self.gan = Sequential()
        self.gan.add(generator)
        self.gan.add(discriminator)
        self.gan.compile(loss='binary_crossentropy', optimizer=optimizer)

    def init_data(self, train_path, val_path, test_path, act_id,
                  window_size=5 * 50,
                  step_size=int(5 * 50 / 2),
                  col_names=['userAcceleration.x', 'userAcceleration.y', 'userAcceleration.z', 'userAcceleration.c'],
                  method='sliding',
                  input_cols_train=['userAcceleration.x', 'userAcceleration.y', 'userAcceleration.z'],
                  input_cols_eval=['userAcceleration.c'],
                  ):
        self.act_id = act_id
        self.window_size = window_size
        self.step_size = step_size
        self.col_names = col_names
        self.method = 'sliding'
        self.input_cols_train = input_cols_train
        self.input_cols_eval = input_cols_eval

        print('Load Data...')
        train_df = pd.read_hdf(train_path)
        val_df = pd.read_hdf(val_path)
        test_df = pd.read_hdf(test_path)

        train_windowed_df = windowing_dataframe(train_df, window_size=window_size, step_or_sample_size=step_size,
                                                col_names=col_names, method=method)
        val_windowed_df = windowing_dataframe(val_df, window_size=window_size, step_or_sample_size=step_size,
                                              col_names=col_names, method=method)
        test_windowed_df = windowing_dataframe(test_df, window_size=window_size, step_or_sample_size=step_size,
                                               col_names=col_names, method=method)

        print('Transform Data...')
        self.x_train, self.y_train = transform_windows_df(train_windowed_df, input_cols=input_cols_train,
                                                          one_hot_encode=False,
                                                          as_channel=False)
        self.x_val, self.y_val = transform_windows_df(val_windowed_df, input_cols=input_cols_train,
                                                      one_hot_encode=False,
                                                      as_channel=False)
        self.x_test, self.y_test = transform_windows_df(test_windowed_df, input_cols=input_cols_train,
                                                        one_hot_encode=False,
                                                        as_channel=False)

        x_train_clf, self.y_train_clf = transform_windows_df(train_windowed_df, input_cols=input_cols_eval,
                                                             one_hot_encode=False,
                                                             as_channel=False)
        x_val_clf, self.y_val_clf = transform_windows_df(val_windowed_df, input_cols=input_cols_eval,
                                                         one_hot_encode=False,
                                                         as_channel=False)
        x_test_clf, self.y_test_clf = transform_windows_df(test_windowed_df, input_cols=input_cols_eval,
                                                           one_hot_encode=False,
                                                           as_channel=False)

        self.x_train_clf = x_train_clf.reshape((len(x_train_clf), window_size))
        self.x_val_clf = x_val_clf.reshape((len(x_val_clf), window_size))
        self.x_test_clf = x_test_clf.reshape((len(x_test_clf), window_size))

        self.x_train_activity, _ = filter_by_activity_index(x=self.x_train, y=self.y_train, activity_idx=self.act_id)

        print('Calculate origin performance...')
        self.calc_origin_train_val_performance()
        print('Done!')

    def calc_origin_train_val_performance(self, verbose=False):
        if (self.x_train_clf is None) or (self.x_val_clf is None):
            print('Please run method: init_data first.')
            return

        self.orig_svm_clf = SVC()
        self.orig_svm_clf.fit(self.x_train_clf, self.y_train_clf)

        y_train_head = self.orig_svm_clf.predict(self.x_train_clf)
        self.orig_train_acc = accuracy_score(self.y_train_clf, y_train_head)
        self.orig_train_f1_score = f1_score(self.y_train_clf, y_train_head, average=None)[self.act_id]

        y_val_head = self.orig_svm_clf.predict(self.x_val_clf)
        self.orig_val_acc = accuracy_score(self.y_val_clf, y_val_head)
        self.orig_val_f1_score = f1_score(self.y_val_clf, y_val_head, average=None)[self.act_id]

        if verbose:
            print('Original training acc: ', self.orig_train_acc)
            print('Original training f1_score for act_id ', self.act_id, ': ', self.orig_train_f1_score, '\n')
            print('Original validation acc: ', self.orig_val_acc)
            print('Original validation f1_score for act_id ', self.act_id, ': ', self.orig_val_f1_score, '\n')

    def train_gan(self, steps, batch_size=64, eval_step=100, random=False, label_smoothing=False):
        if (self.gan is None) and (self.x_train_activity is None):
            print('Please run method "create_gan" and "init_data" first.')
            return
        elif self.gan is None:
            print('Please run method "create_gan" first.')
            return
        elif self.x_train_activity is None:
            print('Please run method "init_data" first.')
            return

        start = 0
        for step in range(steps):
            random_latent_vectors = np.random.normal(size=(batch_size, self.generator.input_shape[1]))

            generated_sensor_data = self.generator.predict(random_latent_vectors)

            if random:
                index = np.random.choice(self.x_train_activity.shape[0], batch_size, replace=False)
                real_sensor_data = self.x_train_activity[index]
            else:
                stop = start + batch_size
                real_sensor_data = self.x_train_activity[start:stop]
                start += batch_size

            combined_sensor_data = np.concatenate([generated_sensor_data, real_sensor_data])
            if label_smoothing:
                labels = np.concatenate([smooth_labels(np.zeros((batch_size, 1)), 0.0, 0.3),
                                         smooth_labels(np.ones((batch_size, 1)), -0.3, 0.3)])
            else:
                labels = np.concatenate([np.zeros((batch_size, 1)), np.ones((batch_size, 1))])

            # shuffle data
            combined_sensor_data, labels = shuffle(combined_sensor_data, labels)

            d_loss = self.discriminator.train_on_batch(combined_sensor_data, labels)

            misleading_targets = np.ones((batch_size, 1))

            a_loss = self.gan.train_on_batch(random_latent_vectors, misleading_targets)

            if start > len(self.x_train_activity) - batch_size:
                start = 0

            if step % eval_step == 0:
                save = False
                print('discriminator loss:', d_loss)
                print('adversarial loss:', a_loss)
                print('\n')

                gen_train_f1_score, gen_val_f1_score = self.eval()

                if gen_train_f1_score > self.best_train_f1_score:
                    print('Train f1-score for act_id ', self.act_id, 'improved from ', self.best_train_f1_score, 'to ',
                          gen_train_f1_score)
                    self.best_train_f1_score = gen_train_f1_score
                    save = True
                if gen_val_f1_score > self.best_val_f1_score:
                    print('Validation f1-score for act_id ', self.act_id, 'improved from ', self.best_val_f1_score,
                          'to ', gen_val_f1_score)
                    self.best_val_f1_score = gen_val_f1_score
                    save = True

                if save:
                    self.generator.save(
                        os.path.join(self.out_dir, 'generator_{}_tf1-{}_vf1-{}.keras'.format(self.act_id,
                                                                                             self.best_train_f1_score,
                                                                                             self.best_val_f1_score)))
                    self.discriminator.save(
                        os.path.join(self.out_dir, 'discriminator_{}_tf1-{}_vf1-{}.keras'.format(self.act_id,
                                                                                                 self.best_train_f1_score,
                                                                                                 self.best_val_f1_score)))

    def eval(self, percentage=0.2):
        num_gen = int(np.ceil(len(self.x_train_activity) * percentage))
        random_latent_vectors = np.random.normal(size=(num_gen, self.latent_dim))
        generated_sensor_data = self.generator.predict(random_latent_vectors)

        gen_df = pd.DataFrame(np.array([ts.transpose() for ts in generated_sensor_data]).tolist(),
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

        svm_clf = SVC()
        svm_clf.fit(x_train_gen, y_train_gen)

        y_train_head = svm_clf.predict(self.x_train_clf)
        # train_acc = accuracy_score(self.y_train_clf, y_train_head)
        gen_train_f1_score = f1_score(self.y_train_clf, y_train_head, average=None)[self.act_id]

        y_val_head = svm_clf.predict(self.x_val_clf)
        # test_acc = accuracy_score(self.x_val_clf, y_val_head)
        gen_val_f1_score = f1_score(self.y_val_clf, y_val_head, average=None)[self.act_id]

        return gen_train_f1_score, gen_val_f1_score
