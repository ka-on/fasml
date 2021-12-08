#!/bin/env python

#######################################################################
# Copyright (C) 2021 Onur Kaya, Julian Dosch
#
# This file is part of fasml.
#
#  greedyFAS is free software: you can redistribute it and/or modify
#  it under the terms of the GNU General Public License as published by
#  the Free Software Foundation, either version 3 of the License, or
#  (at your option) any later version.
#
#  greedyFAS is distributed in the hope that it will be useful,
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
import tensorflow as tf


def categorize_by_groups(path, output_path):
    # TODO: It's possible to make this more efficient
    i = 1
    full_progress = len(os.listdir(path))
    for f in os.listdir(path):
        print(f"Processing {f}; {i} of {full_progress}")
        file = open(os.path.join(path, f))
        data = json.load(file)

        for protein in data["feature"]:
            group_name = protein.split('#')[0]
            try:
                output_file = open(f"{output_path}/{group_name}.json", "r")
                output_data = json.load(output_file)
            except FileNotFoundError:
                output_file = open(f"{output_path}/{group_name}.json", "w+")
                output_data = {"feature": {}}
            output_file.close()

            output_file = open(f"{output_path}/{group_name}.json", "w")
            output_data["feature"][protein] = data["feature"][protein]
            output_file.write(json.dumps(output_data))
            output_file.close()
        i += 1


def shortest(data):
    return min([data["feature"][protein]["length"] for protein in data["feature"]])


def occurring_features(data):
    features = []
    for protein in data["feature"]:
        for feature in data["feature"][protein]:
            if data["feature"][protein][feature] != {} and feature != "length":
                for feature_type in data["feature"][protein][feature]:
                    if feature_type not in features:
                        features.append(feature_type)
    return features


def topology_gen(path, outpath):
    topology_dict = {}
    i = 1
    for f in os.listdir(path):
        print(f"{f}: {i} of {len(os.listdir(path))}")
        i += 1
        file = open(os.path.join(path, f))
        data = json.load(file)
        regions = data["metadata"]["regions"]
        features = len(data["metadata"]["features"])
        topology = []
        neurons = regions*features
        while neurons > 1:
            topology.append(int(neurons))
            neurons /= 2
        topology_dict[f] = topology
    out_file = open(os.path.join(outpath, "topology.json"), "w+")
    out_file.write(json.dumps(topology_dict))


def randint(maxval):
    return int(tf.random.uniform(shape=[1], minval=0, maxval=maxval, dtype=tf.dtypes.int32).numpy())


def read_batch(file_obj, batch_num):
    """
    p = [k for k in file_obj.read().split('\n')]
    r = [m.split('\t') for m in p if m != '']
    c = 0
    for ele in r:
        c += 1
        print(len(ele))
        if c == 10: quit()
    #quit()
    """
    seqs = []
    for i in range(0, batch_num):
        line = file_obj.readline()
        if not line:
            return seqs
        else:
            seqs.append([int(j) for j in line.split('\t') if j != ''])
    return seqs



