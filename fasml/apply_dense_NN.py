import os
import json
import argparse
from fasml.model_types import dense


def main(inpath, outpath, network):
    with open(os.path.join(network, 'topology.json'), 'r') as infile:
        topology = json.load(infile)
        model = dense.DenseLayersModel(topology['topology'], topology['name'])
    model.load_weights(os.path.join(network, topology['name']))
    out = open(outpath, 'w')
    with open(inpath) as queryfile:
        results = model.predict(queryfile)
    labels = []
    with open(''.join(inpath.split('.')[0:-1]) + '_labels.csv') as labelfile:
        for line in labelfile.readlines():
            labels.append(line.rstrip('\n'))
    for i in range(len(results)):
        out.write(labels[i] + '\t' + str(round(results[i][0], 0)) + '\n')
    out.close()


def get_args():
    parser = argparse.ArgumentParser(epilog="creates dense NN")
    required = parser.add_argument_group('required arguments')
    required.add_argument("-i", "--inPath", default='.', type=str, required=True,
                          help="path to query.csv")
    required.add_argument("-o", "--outPath", default='.', type=str, required=True,
                          help="path to output file")
    required.add_argument("-n", "--network", type=str, required=True,
                          help="path to folder with network data")
    args = parser.parse_args()
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
    main(args.inPath, args.outPath, args.network)


if __name__ == '__main__':
    get_args()
