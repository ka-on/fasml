from file_handling.utils import shortest, occurring_features
from networks.citrat_cycle.storage import CitrateStatic
import json
import os
import pandas as pd


def matrix_gen(proteome, pids, fids, regions):
    feature_dict = {}
    counter = 0
    for fid in fids:
        counter += 1
        print(f"\tFeature {counter} of {len(fids)}")
        for region in range(1, regions+1):
            f_region = fid + '#' + str(region)
            feature_dict[f_region] = []
            for pid in pids:
                check = False
                if fid.split('_')[0] == 'coils':
                    tool = 'coils2'
                else:
                    tool = fid.split('_')[0]
                if fid in proteome["feature"][pid][tool]:
                    r_size = int(proteome["feature"][pid]['length'] / region)
                    r_overhang = proteome["feature"][pid]['length'] % region
                    r_start = 1
                    if r_overhang > 0:
                        r_end = r_size + 1
                    else:
                        r_end = r_size
                    for i in range(1, region):
                        r_start = r_end + 1
                        if i < r_overhang:
                            r_end += r_size + 1
                        else:
                            r_end += r_size
                    for instance in proteome["feature"][pid][tool][fid]['instance']:
                        if r_start <= instance[0] <= r_end or r_start <= instance[1] <= r_end:
                            check = True
                if check:
                    feature_dict[f_region].append(1)
                else:
                    feature_dict[f_region].append(0)
    df = pd.DataFrame(feature_dict, index=pids)
    return df


def regions_binary(data):
    pids = [protein for protein in data["feature"]]
    features = occurring_features(data)
    regions = shortest(data)
    matrix = matrix_gen(data, pids, features, regions).T
    regions_dict = {}
    for row in matrix:
        regions_dict[row] = matrix[row].tolist()
    metadata = {
        "regions": regions,
        "features": features
    }
    output_data = {
        "data": regions_dict,
        "metadata": metadata
    }
    return output_data


def regions_binary_file_gen(path, outpath):
    """
    Make sure the files are categorized by groups
    """
    i = 1
    for f in os.listdir(path):
        print(f"Processing {f}: {i} of {len(os.listdir(path))}")
        file = open(os.path.join(path, f))
        data = json.load(file)
        output = json.dumps(regions_binary(data))
        output_file = open(os.path.join(outpath, f), "w+")
        output_file.write(output)
        output_file.close()
        i += 1
