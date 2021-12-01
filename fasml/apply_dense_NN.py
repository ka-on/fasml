import os
import json
import argparse
from fasml.model_types import dense


def main(inpath, outpath, network, cpus):
    with open(os.path.join(network, 'topology.json'), 'r') as infile:
        topology = json.load(infile)
        model = dense.DenseLayersModel(topology['topology'], topology['name'])
    out = open(outpath, 'w')
    with open(inpath) as queryfile:
        results = model.predict(queryfile, cpus)
    print(results)


def get_args():
    parser = argparse.ArgumentParser(epilog="creates dense NN")
    required = parser.add_argument_group('required arguments')
    optional = parser.add_argument_group('optional arguments')
    required.add_argument("-i", "--inPath", default='.', type=str, required=True,
                          help="path to query.csv")
    required.add_argument("-o", "--outPath", default='.', type=str, required=True,
                          help="path to output file")
    required.add_argument("-n", "--network", type=str, required=True,
                          help="path to folder with network data")
    optional.add_argument("-c", "--cpus", type=int, default=1,
                          help="number of cpus")
    args = parser.parse_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
    main(args.inPath, args.outPath, args.network)


if __name__ == '__main__':
    get_args()
