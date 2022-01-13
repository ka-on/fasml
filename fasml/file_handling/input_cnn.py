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
import argparse


def to_str_array_single(sequence, mode):
    ss3, ss8, dis = sequence
    i_dict = {'C': 0, 'H': 1, 'E': 2, 'S': 3, 'T': 4, 'I': 5, 'B': 6, 'G': 7}
    str_array = []
    for i in len(ss8):
        if mode:
            tmp = [0, 0, 0, 0, 0, 0, 0, 0, 0]
            tmp[i_dict[ss8[i]]] = 1
            tmp[8] = int(dis[i])
        else:
            tmp = [0, 0, 0, 0]
            tmp[i_dict[ss3[i]]] = 1
            tmp[3] = int(dis[i])
        str_array.append(tmp)
    return str_array


def to_str_array(path, mode):
    str_dict = {}
    with open(path, 'r') as infile:
        line = infile.readline()
        while line:
            prot_id = line.rstrip('\n').lstrip('>')
            line = infile.readline()
            ss3 = line.rstrip('\n')
            line = infile.readline()
            ss8 = line.rstrip('\n')
            line = infile.readline()
            dis = line.rstrip('\n')
            str_array = to_str_array_single((ss3, ss8, dis), mode)
            str_dict[prot_id] = str_array
            line = infile.readline()
    return str_dict


def main(inpath, outpath, mode):
    str_dict = to_str_array(inpath, mode)
    with open(outpath, "w+") as output_file:
        json.dump(str_dict, output_file)


def get_args():
    parser = argparse.ArgumentParser(epilog="transform the .structure file into an array format")
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    required.add_argument("-i", "--inputPath", default=None, type=str, required=True,
                          help="path to input .structure file")
    required.add_argument("-o", "--outPath", default='.', type=str, required=True,
                          help="path to output directory")
    optional.add_argument("-s", "--ss3", action="store_true",
                          help="use ss3 structure instead of ss8")
    args = parser.parse_args()
    main(args.inputPath, args.outPath, args.group_json)


if __name__ == '__main__':
    get_args()
