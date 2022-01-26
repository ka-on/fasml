#!/bin/env python

#######################################################################
# Copyright (C) 2021 Onur Kaya, Julian Dosch
#
# This file is part of fasml.
#
#  fasml is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  fasml is distributed in the hope that it will be useful,
#  but WITHOUT ANY WARRANTY; without even the implied warranty of
#  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#  GNU General Public License for more details.
#
#  You should have received a copy of the GNU General Public License
#  along with fasml.  If not, see <http://www.gnu.org/licenses/>.
#
#######################################################################

import json
import tensorflow as tf
import numpy as np
from math import ceil
from random import shuffle
import time


class CNNModel:

    def __init__(self, name, features,
                 optimizer=tf.keras.optimizers.SGD(),
                 loss_function=tf.losses.MeanSquaredError()):
        model = tf.keras.Sequential()
        convolution = tf.keras.layers.Conv1D(64, 20, activation='relu', input_shape=(None, features))
        pooling = tf.keras.layers.GlobalMaxPooling1D()
        convolution2 = tf.keras.layers.Conv1D(128, 4, activation='relu')
        convolution3 = tf.keras.layers.Conv1D(256, 4, activation='relu')
        model.add(convolution)
        model.add(convolution2)
        model.add(convolution3)
        model.add(pooling)
        model.add(tf.keras.layers.Flatten())
        for units in [128, 64, 32, 16, 8, 4, 2, 1]:
            layer = tf.keras.layers.Dense(
                units=units,
                activation=tf.keras.activations.relu
            )
            model.add(layer)

        model.compile(optimizer=optimizer, loss=loss_function, metrics=["accuracy"])
        self.model = model
        self.topology = {
            'input': [features, '>=20'],
            'layers': [
                ['conv', [features, 20, 32]],
                ['maxPool', [32]],
                ['conv', [1, 4, 64]],
                ['maxPool', [64]],
                ['conv', [1, 4, 128]],
                ['maxPool', [128]],
                ['dense', [128]],
                ['dense', [64]],
                ['dense', [32]],
                ['dense', [16]],
                ['dense', [8]],
                ['dense', [4]],
                ['dense', [2]],
                ['dense', [1]],
            ]
            }
        self.name = name
        self.loss_fn = loss_function
        self.optimizer = optimizer
        self.train_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()
        self.val_acc_metric = tf.keras.metrics.SparseCategoricalAccuracy()

    def get_topology(self):
        return self.topology

    def plot(self):
        tf.keras.utils.plot_model(
            self.model, to_file='model.png', show_shapes=True, show_dtype=True,
            show_layer_names=False, rankdir='LR'
        )

    def train(self, positive, negative, p_exclude, n_exclude, epochs):
        print(self.name)
        # prepare batches
        pos_data, pos_size = self.create_datadict(positive, p_exclude, 20, 20)
        neg_data, neg_size = self.create_datadict(negative, n_exclude, 20, 20)
        pos_batches, neg_batches = self.prepare_batches(pos_data, pos_size, neg_data, neg_size)

        # train model
        print('# Training Model')
        train_info = self.train_on_batches(pos_batches, neg_batches, epochs)
        return train_info

    def train_on_batches(self, pos_batches, neg_batches, epochs):
        train_data = {'t_acc': []}
        for epoch in range(epochs):
            print("\nStart of epoch %d" % (epoch,))
            start_time = time.time()
            loss_value = None

            # Iterate over the batches of the dataset.
            for step in range(len(pos_batches)):
                loss_value = []
                for minibatch in pos_batches[step]:
                    loss_value.append(self.model.train_on_batch(np.array(minibatch), np.ones(len(minibatch)))[1])
                for minibatch in neg_batches[step]:
                    loss_value.append(self.model.train_on_batch(np.array(minibatch), np.zeros(len(minibatch)))[1])

                # Log every 200 batches.
                if step % 1 == 0:
                    print(
                        "Training loss (for one batch) at step %d: %.4f"
                        % (step, float(sum(loss_value)/len(loss_value)))
                    )
                    print("Seen so far: %d batches" % (step + 1))

            evaluation = []
            for step in range(len(pos_batches)):
                for minibatch in pos_batches[step]:
                    evaluation.append(self.model.evaluate(np.array(minibatch), np.ones(len(minibatch)))[1])
                for minibatch in neg_batches[step]:
                    evaluation.append(self.model.evaluate(np.array(minibatch), np.zeros(len(minibatch)))[1])

            train_acc = sum(evaluation) / len(evaluation)
            print("Training acc over epoch: %.4f" % (float(train_acc),))
            print("Time taken: %.2fs" % (time.time() - start_time))
            train_data['t_acc': train_acc]
        return train_data

    def prepare_batches(self, pos_data, pos_size, neg_data, neg_size):
        pos_to_neg = pos_size / neg_size
        n_pos = 1
        n_neg = 1
        print('Positive to Negative data ratio: ' + str(pos_to_neg))
        if pos_to_neg < 0.2:
            n_pos = ceil(0.2 / pos_to_neg)
        elif pos_to_neg > 2.0:
            n_neg = ceil(pos_to_neg / 2.0)
        print('Positive to Negative show ratio: ' + str(n_pos) + ' / ' + str(n_neg))
        pos_keys = []
        neg_keys = []
        print('# Preparing Batches')
        for i in range(n_pos):
            tmp = list(pos_data.keys())
            shuffle(tmp)
            pos_keys.extend(tmp)
        for x in range(n_neg):
            tmp = list(neg_data.keys())
            shuffle(tmp)
            neg_keys.extend(tmp)
        if len(pos_keys) > len(neg_keys):
            pos_batches, neg_batches = self.prepare_batches_01(pos_keys, neg_keys, pos_data, neg_data)
        else:
            neg_batches, pos_batches = self.prepare_batches_01(neg_keys, pos_keys, neg_data, pos_data)
        return pos_batches, neg_batches

    def prepare_batches_01(self, keys_01, keys_02, data_01, data_02):
        b_size = ceil(len(keys_01) / len(keys_02))
        batches_01 = []
        tmp = np.array_split(keys_01, b_size)
        c = 0
        for i in tmp:
            tmp2 = []
            for x in i:
                tmp2.append(data_01[x])
            batches_01.append(tmp2)
        batches_02 = []
        for i in keys_02:
            batches_02.append([data_02[i]])
        return batches_01, batches_02

    def create_datadict(self, path, exclude, max_batch, min_len):
        datadict = {}
        data_out = {}
        empty_col = None
        with open(path, 'r') as infile:
            indata = json.load(infile)
        size = 0
        for prot in indata:
            if not empty_col:
                empty_col = []
                for f in indata[prot][0]:
                    empty_col.append(0)
            if prot not in exclude:
                tmp = indata[prot]
                while len(tmp) < min_len:
                    tmp.append(empty_col)
                if str(len(tmp)) not in datadict:
                    datadict[str(len(tmp))] = []
                datadict[str(len(tmp))].append(tmp)
                size += 1

        for length in datadict:
            if len(datadict[length]) <= max_batch:
                data_out[length] = datadict[length]
            else:
                for x in range(ceil(len(datadict[length])/max_batch)):
                    data_out[length + '_' + str(x)] = datadict[length][max_batch * x: max_batch * (x + 1)]
        return data_out, size

    def predict(self, query):
        query_data = []
        for line in query.readlines():
            if line:
                query_data.append([int(j) for j in line.split('\t') if j != ''])
        results = self.model.predict(
            query_data, batch_size=50, verbose=0, steps=None, callbacks=None, max_queue_size=10,
            workers=1, use_multiprocessing=True
        )
        return results

    def load_weights(self, path):
        self.model.load_weights(path)

    def save_weights(self, path):
        self.model.save_weights(path)

