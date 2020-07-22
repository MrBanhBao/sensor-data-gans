import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import Adam


def create_gan(generator_model, discriminator_model, optimizer=Adam(), verbose=0) -> Sequential:
    discriminator_model.trainable = False
    gan = Sequential()
    gan.add(generator_model)
    gan.add(discriminator_model)
    gan.compile(loss='binary_crossentropy', optimizer=optimizer)

    if verbose:
        print(gan.summary())

    return gan

def smooth_labels(labels, lower, upper, round_decimel=1, random_state=None):
    n_samples = len(labels)
    return labels + np.round(np.random.uniform(lower, upper, n_samples, random_state=random_state), round_decimel)[:, None]


def train_gan(generator, discriminator, gan, x_train_activity, steps, batch_size=64, eval_step=100, random=False):
    start = 0
    for step in range(steps):
        random_latent_vectors = np.random.normal(size=(batch_size, generator.input_shape[1]))

        generated_sensor_data = generator.predict(random_latent_vectors)

        if random:
            index = np.random.choice(x_train_activity.shape[0], batch_size, replace=False)
            real_sensor_data = x_train_activity[index]
        else:
            stop = start + batch_size
            real_sensor_data = x_train_activity[start:stop]
            start += batch_size

        combined_sensor_data = np.concatenate([generated_sensor_data, real_sensor_data])
        labels = np.concatenate([np.zeros((batch_size, 1)), np.ones((batch_size, 1))])

        d_loss = discriminator.train_on_batch(combined_sensor_data, labels)

        misleading_targets = np.ones((batch_size, 1))

        a_loss = gan.train_on_batch(random_latent_vectors, misleading_targets)

        if start > len(x_train_activity) - batch_size:
            start = 0

        if step % eval_step == 0:
            print('discriminator loss:', d_loss)
            print('adversarial loss:', a_loss)

