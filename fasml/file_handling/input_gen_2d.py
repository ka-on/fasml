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


from fasml.file_handling import input_gen
from fasml.file_handling.utils import randint
import os
import json
import argparse


def read_input(path):    # read fasta like format for sec structure
    sec_features = {}
    with open(path, 'r') as infile:
        line = infile.readline().rstrip('\n')
        while line:
            prot_id = line.lstrip('>').rstrip('\n')
            ss3 = infile.readline().rstrip('\n')
            ss8 = infile.readline().rstrip('\n')
            diso = infile.readline().rstrip('\n')
            if prot_id and ss3 and ss8 and diso:
                sec_features[prot_id] = (ss3, ss8, diso)
            else:
                raise Exception('Error in structure file:\n' + path)
            line = infile.readline()
    return sec_features


def to_fas_format(sec_fasta_format):    # takes fasta like format and turns it into fas feature format
    sec_fas_format = {}
    for prot_id in sec_fasta_format:
        sec_fas_format[prot_id] = {'ss3': {'ss3_C': {'instance': []}, 'ss3_H': {'instance': []},
                                           'ss3_E': {'instance': []}},
                                   'ss8': {'ss8_C': {'instance': []}, 'ss8_H': {'instance': []},
                                           'ss8_E': {'instance': []}, 'ss8_S': {'instance': []},
                                           'ss8_T': {'instance': []}, 'ss8_I': {'instance': []},
                                           'ss8_B': {'instance': []}, 'ss8_G': {'instance': []}},
                                   'disorder': {'disorder_D': {'instance': []}},
                                   'length': len(sec_fasta_format[prot_id][0])}
        for stype in ['ss3', 'ss8', 'disorder']:
            sec_fas_format = to_fas_single(sec_fasta_format[prot_id], sec_fas_format, stype, prot_id)
    sec_fas_format = {'feature': sec_fas_format}
    return sec_fas_format


def to_fas_single(sec_fasta_format, sec_fas_format, stype, prot_id):
    sdict = {'ss3': 0, 'ss8': 1, 'disorder': 2}
    current = None
    start = None
    for position in range(len(sec_fasta_format[sdict[stype]])):
        if not current:
            current = sec_fasta_format[sdict[stype]][position]
            start = position + 1
        elif not sec_fasta_format[sdict[stype]][position] == current and not stype == 'disorder':
            sec_fas_format[prot_id][stype][stype + '_' + current]['instance'].append((start, position, None))
            current = sec_fasta_format[sdict[stype]][position]
            start = position + 1
        elif stype == 'disorder' and not sec_fasta_format[sdict[stype]][position] == current:
            if current == '1':
                sec_fas_format[prot_id][stype]['disorder_D']['instance'].append((start, position, None))
            current = sec_fasta_format[sdict[stype]][position]
            start = position + 1
    if stype == 'disorder' and current == '1':
        sec_fas_format[prot_id][stype]['disorder_D']['instance'].append((start, len(sec_fasta_format[sdict[stype]]) - 1, None))
    elif not stype == 'disorder':
        sec_fas_format[prot_id][stype][stype + '_' + current]['instance'].append((start, len(sec_fasta_format[sdict[stype]])-1, None))
    return sec_fas_format


def regions_binary_2d(data, ss3):
    pids = [protein for protein in data["feature"]]
    if ss3 == 'ss3':
        features = ['ss3_C', 'ss3_H', 'ss3_E', 'disorder_D']
    else:
        features = ['ss8_C', 'ss8_H', 'ss8_E', 'ss8_S', 'ss8_T', 'ss8_I', 'ss8_B', 'ss8_G', 'disorder_D']
    regions = input_gen.shortest(data)
    matrix = input_gen.matrix_gen(data, pids, features, regions).T
    regions_dict = {}
    for row in matrix:
        regions_dict[row] = matrix[row].tolist()
    metadata = {
        "regions": regions,
        "features": features,
        "#proteins": len(pids)
    }
    output_data = {
        "data": regions_dict,
        "metadata": metadata
    }
    return output_data


def main(inpath, ss3, outpath, group):    # inpath: path to .struc file; ss3: boolean if true that use ss3 structure, else ss8
    sec_features = read_input(inpath+group+'.structure')
    sec_fas_format = to_fas_format(sec_features)
    output_data = regions_binary_2d(sec_fas_format, ss3)
    with open(outpath + group + ".json", "w+") as output_file:
        json.dump(output_data, output_file)
    dataset_gen_2d(outpath, inpath, group, outpath)


def dataset_gen_2d(path, annopath, group_name, out_dir):
    """
    TODO: This function could be broken down to smaller functions
    """
    with open(os.path.join(path, group_name+'.json')) as file:
        group_data = json.load(file)
    # Positive data
    px = [group_data["data"][protein] for protein in group_data["data"]]
    py = [1 for i in range(0, len(px))]
    # Generate file
    outfile_px_path = os.path.join(out_dir, f"{group_name}_px.tsv")
    outfile_px = open(outfile_px_path, "a")
    for j, pa_pattern in enumerate(px):
        outfile_px.write('\t'.join([str(i) for i in pa_pattern])+'\n')
    outfile_px.close()

    #Negative data
    negative = len(px)*4
    outfile_nx_path = os.path.join(out_dir, f"{group_name}_nx.tsv")
    outfile_nx = open(outfile_nx_path, "a")
    files = os.listdir(annopath)
    if group_name + '.structure' in files:
        files.remove(group_name + '.structure')
    division = int(negative / len(files))
    rest = negative % len(files)
    per_file = {}
    for i in files:
        per_file[i] = division
    for i in range(rest):
        random_file = files[randint(len(files))]
        per_file[random_file] += 1
        files.remove(random_file)
    for n_file in per_file:
        negative_data = to_fas_format(read_input(os.path.join(annopath, n_file)))
        for i in range(per_file[n_file]):
            random_protein = [protein for protein in
                              negative_data["feature"]][randint(len([protein for protein in negative_data["feature"]]))]
            pa = input_gen.matrix_gen_single(
                negative_data["feature"][random_protein],
                group_data["metadata"]["features"],
                group_data["metadata"]["regions"])
            outfile_nx.write('\t'.join([str(i) for i in pa]) + '\n')
    outfile_nx.close()


def queryset_gen(inpath, outdir, groupjson):    # inpath: path to .struc file
    sec_features = read_input(inpath)
    sec_fas_format = to_fas_format(sec_features)
    with open(groupjson, 'r') as file:
        group_data = json.load(file)
    with open(os.path.join(outdir, ''.join(inpath.split('/')[-1].split('.')[0:-1]) + '.csv'), 'w') as out:
        labels = open(os.path.join(outdir, ''.join(inpath.split('/')[-1].split('.')[0:-1]) + '_labels.csv'), 'w')
        for protein in sec_fas_format['feature']:
            line = input_gen.matrix_gen_single(
                sec_fas_format["feature"][protein],
                group_data["metadata"]["features"],
                group_data["metadata"]["regions"])
            out.write('\t'.join([str(i) for i in line]) + '\n')
            labels.write(protein + '\n')


def query_gen_entry():
    parser = argparse.ArgumentParser(epilog="creates the queryinput for the ml")
    required = parser.add_argument_group('required arguments')
    required.add_argument("-i", "--inputPath", default=None, type=str, required=True,
                          help="path to input .structure file")
    required.add_argument("-o", "--outPath", default='.', type=str, required=True,
                          help="path to output directory")
    required.add_argument("-j", "--group_json", type=str, required=True,
                          help="data of the search group (json file)")
    args = parser.parse_args()
    queryset_gen(args.inputPath, args.outPath, args.group_json)


def get_args():
    parser = argparse.ArgumentParser(epilog="creates the input for the ml")
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    required.add_argument("-a", "--annotation", default='.', type=str, required=True,
                          help="path to input folder with .structure files")
    required.add_argument("-o", "--outPath", default='.', type=str, required=True,
                          help="path to output directory")
    required.add_argument("-g", "--group", type=str, required=True,
                          help="name of the group, filename [NAME.structure]")
    optional.add_argument("-s", "--ss3", action="store_true",
                          help="use ss3 structure instead of ss8")
    args = parser.parse_args()
    main(args.annotation, args.ss3, args.outPath, args.group)


if __name__ == '__main__':
    get_args()
