import os
import tensorflow as tf
import numpy as np
from file_handling.utils import read_batch


class DenseLayersModel:

    def __init__(self, topology, name,
                 optimizer=tf.keras.optimizers.Adam(),
                 loss_function=tf.losses.MeanSquaredError()):
        model = tf.keras.Sequential()
        l0 = tf.keras.layers.Dense(
            units=topology[1],
            input_shape=(topology[0],)
        )
        model.add(l0)
        for units in topology[2:]:
            layer = tf.keras.layers.Dense(
                units=units,
                activation=tf.keras.activations.relu
            )
            model.add(layer)

        model.compile(
            optimizer=optimizer,
            loss=loss_function,
            metrics=["accuracy"]
        )

        self.model = model
        self.topology = topology
        self.name = name

    def plot(self):
        tf.keras.utils.plot_model(
            self.model, to_file='model.png', show_shapes=True, show_dtype=True,
            show_layer_names=False, rankdir='LR'
        )

    def train(self, px, py, nx, ny, save_weights_path, px_length):
        group_weights_save_path = os.path.join(save_weights_path, self.name)
        os.mkdir(group_weights_save_path)
        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(group_weights_save_path, self.name),
            save_weights_only=True,
            verbose=1
        )
        py = py.read().split('\t')
        ny = ny.read().split('\t')
        training_samples_num = px_length
        samples_fed = 0
        while samples_fed != training_samples_num:
            print("test")
            batch_size = min(512, training_samples_num-samples_fed)
            batch_px = read_batch(px, batch_size)
            batch_nx = read_batch(nx, batch_size*4)  # The number multiplied has to be the same as the ratio 1:4
            batch_x = batch_px + batch_nx

            batch_py = py[samples_fed:samples_fed+batch_size]
            batch_ny = ny[samples_fed*4:samples_fed+batch_size*4]
            batch_y = [float(i) for i in batch_py+batch_ny]
            samples_fed += batch_size
            history = self.model.fit(
                np.asarray(batch_x), np.asarray(batch_y),
                callbacks=cp_callback,
                epochs=10)
        return history

    def predict(self):
        pass  # TO DO
