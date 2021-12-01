import os
import json
import argparse
from fasml.model_types import dense


def main(group, inpath, outpath, epochv):
    with open(os.path.join(inpath, group + '.json'), 'r') as infile:
        metadata = json.load(infile)['metadata']
    topology = calc_topology(metadata['regions'], len(metadata['features']))
    model = dense.DenseLayersModel(topology, group)
    training_length = int(metadata['#proteins'] * 0.9)
    eval_length = metadata['#proteins'] - training_length
    px = open(os.path.join(inpath, group + '_px.tsv'), 'r')
    nx = open(os.path.join(inpath, group + '_nx.tsv'), 'r')
    model.train(px, nx, outpath, training_length, eval_length, epochv)
    model_data = {'topology': topology, 'name': group}
    with open(os.path.join(outpath + group + '/topology.json'), 'w') as out:
        json.dump(model_data, out)


def calc_topology(regions, features):
    layers = [regions * features]
    while layers[-1] > 1:
        newlayer = int(layers[-1] / 2)
        layers.append(newlayer)
    return layers


def get_args():
    parser = argparse.ArgumentParser(epilog="creates dense NN")
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    required.add_argument("-i", "--inpath", default='.', type=str, required=True,
                          help="path to input folder with training data")
    required.add_argument("-o", "--outPath", default='.', type=str, required=True,
                          help="path to directory where network data is saved")
    required.add_argument("-g", "--group", type=str, required=True,
                          help="name of the group, filename [NAME.structure]")
    required.add_argument("-e", "--epochs", type=int, required=True)
    args = parser.parse_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
    main(args.group, args.inpath, args.outPath, args.epochs)


if __name__ == '__main__':
    get_args()
