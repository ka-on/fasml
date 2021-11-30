import os
import tensorflow as tf
import numpy as np
from fasml.file_handling.utils import read_batch


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

    def train(self, px, nx, save_weights_path, regions_binary_path, px_length, eval_length, epochv):
        print(self.name)
        group_weights_save_path = os.path.join(save_weights_path, self.name)
        if not os.path.isdir(group_weights_save_path):
            os.mkdir(group_weights_save_path)

        cp_callback = tf.keras.callbacks.ModelCheckpoint(
            filepath=os.path.join(group_weights_save_path, self.name),
            save_weights_only=True,
            verbose=1
        )
        samples_fed = 0
        run = 1
        batch_x = []
        batch_y = []
        print("Reading training data...", end = "\r")
        while samples_fed != px_length:
            batch_size = min(10, px_length-samples_fed)
            batch_px = read_batch(px, batch_size)
            batch_nx = read_batch(nx, batch_size*4)  # The number multiplied has to be the same as the ratio 1:4
            batch_x = batch_x + batch_px + batch_nx
            batch_py = [1 for i in batch_px]
            batch_ny = [0 for i in batch_nx]

            batch_y = batch_y + batch_py + batch_ny
            samples_fed += batch_size
        print("Reading training data... Done!")
        statistics = [len(batch_y)]
        history = self.model.fit(
            np.asarray(batch_x), np.asarray(batch_y),
            callbacks=cp_callback,
            epochs=epochv, batch_size=50)
        eval_samples_fed = 0
        batch_x = []
        batch_y = []
        while eval_samples_fed != eval_length:
            batch_size = min(500, eval_length-eval_samples_fed)
            eval_samples_fed += batch_size
            print(eval_samples_fed)
            batch_px = read_batch(px, batch_size)
            batch_nx = read_batch(nx, batch_size*4)
            batch_x = batch_x + batch_px + batch_nx
            batch_py = [1 for i in batch_px]
            batch_ny = [0 for i in batch_nx]
            batch_y = batch_y + batch_py + batch_ny
            print("EVAL")
        statistics.append(len(batch_x))
        scores = None
        if batch_x:
            scores = self.model.evaluate(
                np.asarray(batch_x),
                np.asarray(batch_y), 
                batch_size=2500, 
                verbose=1)
        # Log the training
        with open(os.path.join(group_weights_save_path, "log.txt"), 'w+') as log_file:
            log_file.write("epochs: " + str(history.params["epochs"]) 
                           + "\nsteps: " + str(history.params["steps"])
                           + "\ntraining size: " + str(statistics[0])
                           + "\neval size: " + str(statistics[1])
                           + "\n\n# statistics per training epoch [loss & accuracy] #\n")
            for i in range(len(history.history["loss"])):
                log_file.write(f"epoch {str(i+1)}: {str(history.history['loss'][i]):.6} {str(history.history['accuracy'][i]):.6}\n")
            if scores:
                log_file.write(f"evaluation: {str(scores[0]):.6} {str(scores[1]):.6}")
        return history


    def predict(self):
        pass  # TO DO

