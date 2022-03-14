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


import os
import json
import argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
from fasml.model_types import cnn



def main(pos_path, neg_path, name, features, outpath, epochs, p_exclude, n_exclude, acc_threshold):
    model = cnn.CNNModel(name, features)
    topology = model.get_topology()
    outpath2 = os.path.join(outpath, name)
    if not os.path.isdir(outpath2):
        os.mkdir(outpath2)
    train_data = model.train(pos_path, neg_path, p_exclude, n_exclude, epochs, acc_threshold)
    topology['name'] = name
    model.save_weights(outpath2 + '/' + name)
    with open(os.path.join(outpath2 + '/topology.json'), 'w') as out:
        json.dump(topology, out)
    with open(os.path.join(outpath2 + '/statistics.tsv'), 'w') as out:
        out.write('positive data:\t' + str(train_data['p_size']) + '\nnegative data:\t' + str(train_data['n_size'])
                  + '\nration[p/n]:\t' + train_data['ratio'] + '\ntraining accuracy:\t' + str(train_data['t_acc']))


def get_args():
    parser = argparse.ArgumentParser(epilog="creates CNN")
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    required.add_argument("-p", "--posPath", type=str, required=True,
                          help="path to positive training data")
    required.add_argument("-n", "--negPath", type=str, required=True,
                          help="path to negative training data")
    required.add_argument("-o", "--outPath", type=str, required=True,
                          help="path to directory where network data is saved")
    required.add_argument("-g", "--name", type=str, required=True,
                          help="name of the model")
    optional.add_argument("-e", "--epochs", type=int, default=5,
                          help="maximum number of training epochs")
    optional.add_argument("-a", "--target_accuracy", type=float, default=0.95,
                          help="training will stop when reaching this threshold")
    optional.add_argument("--p_exclude", default=(), nargs='*', type=str,
                          help="Choose specific proteins (ids divided by spaces) that will be removed from the "
                               "positive training set")
    optional.add_argument("--n_exclude", default=(), nargs='*', type=str,
                          help="Choose specific proteins (ids divided by spaces) that will be removed from the "
                               "negative training set")
    optional.add_argument("-f", "--features", default=9, type=int,
                          help="Number of features (dimensionality of input)")
    args = parser.parse_args()
    main(args.posPath, args.negPath, args.name, args.features, args.outPath,
         args.epochs, args.p_exclude, args.n_exclude, args.target_accuracy)


if __name__ == '__main__':
    get_args()
