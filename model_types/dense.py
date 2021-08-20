import tensorflow as tf


class DenseLayersModel:

    def __init__(self, topology, optimizer, loss_function, metrics):
        """
        Model Initialization
        """
        input_layer = tf.keras.layers.Input(      # Add the input layer, as many neurons as
            shape=(topology[0],),                 # the number of regions we divide the sequence into
            dtype='int32'
        )

        previous_layer = input_layer
        for layer in topology[1:-1]:  # Add hidden layers according to the topology
            hidden_layer = tf.keras.layers.Dense(
                layer,
                activation=tf.nn.relu
            )(previous_layer)
            previous_layer = hidden_layer

        output_layer = tf.keras.layers.Dense(
            topology[-1],
            activation=tf.nn.softmax
        )(previous_layer)

        model = tf.keras.Model(inputs=input_layer, outputs=output_layer)

        model.compile(
            optimizer=optimizer,
            loss=loss_function,  # False because we apply softmax at the output layer
            metrics=metrics
        )
        """
        End of Model Initialization
        """

        #  Set class variables
        self.model = model
        self.topology = topology

    def plot(self):
        tf.keras.utils.plot_model(
            self.model, to_file='model.png', show_shapes=True, show_dtype=True,
            show_layer_names=False, rankdir='LR'
        )

    def train(self, input_data, labels):
        history = self.model.fit(input_data, labels)
        return history


    def predict(self):
        pass  # TO DO
