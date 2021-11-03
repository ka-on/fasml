import os
import tensorflow as tf
import numpy as np
from file_handling.utils import read_batch
import json


class DenseLayersModel:

    def __init__(self, topology, name,
                 optimizer=tf.keras.optimizers.SGD(),
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

    def train(self, px, nx, save_weights_path, regions_binary_path, px_length, eval_length):
        print(self.name)
        group_weights_save_path = os.path.join(save_weights_path, self.name)
        if not os.path.isdir(group_weights_save_path):
            os.mkdir(group_weights_save_path)

        # Log the training
        log_dict = {}
        if os.path.isfile(os.path.join(group_weights_save_path, "log.txt")):
            i = 1
            while os.path.isfile(os.path.join(group_weights_save_path, f"log_{i}.txt")):
                i += 1
            log_file = open(os.path.join(group_weights_save_path, f"log_{i}.txt"), 'a+')
        else:
            log_file = open(os.path.join(group_weights_save_path, "log.txt"), 'a+')


        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(group_weights_save_path, self.name),
            save_weights_only=True,
            verbose=1
        )
        samples_fed = 0
        run = 1
        while samples_fed != px_length:
            print(f"Run {run}/{int(px_length/32)}")

            batch_size = min(32, px_length-samples_fed)
            batch_px = read_batch(px, batch_size)
            batch_nx = read_batch(nx, batch_size*4)  # The number multiplied has to be the same as the ratio 1:4
            batch_x = batch_px + batch_nx
            batch_py = [1 for i in batch_px]
            batch_ny = [0 for i in batch_nx]

            batch_y = batch_py + batch_ny
            samples_fed += batch_size
            history = self.model.fit(
                np.asarray(batch_x), np.asarray(batch_y),
                callbacks=cp_callback,
                epochs=100)
            run += 1
            log_dict["runs"][run]["epochs"] = history.params["epochs"]
            log_dict["runs"][run]["steps"] = history.params["steps"]

        eval_samples_fed = 0
        while eval_samples_fed != eval_length:
            batch_size = min(512, eval_length-eval_samples_fed)
            eval_samples_fed += batch_size
            print(eval_samples_fed)
            batch_px = read_batch(px, batch_size)
            batch_nx = read_batch(nx, batch_size*4)
            batch_x = batch_px + batch_nx
            batch_py = [1 for i in batch_px]
            batch_ny = [0 for i in batch_nx]
            batch_y = batch_py + batch_ny
            print("EVAL")
            if batch_x:
                self.model.evaluate(
                    np.asarray(batch_x),
                    np.asarray(batch_y))
        return history


    def predict(self):
        pass  # TO DO
