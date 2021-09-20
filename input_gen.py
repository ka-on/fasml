from file_handling.utils import shortest, occurring_features, randint
import json
import os
import pandas as pd
import numpy as np


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


def matrix_gen_single(protein, fids, regions):
    pa_pattern = []
    for fid in fids:
        tool = "coils2" if fid.split('_')[0] == "coils" else fid.split('_')[0]
        for region in range(1, regions+1):
            r_size = int(protein["length"] / region)
            r_overhang = protein["length"] % region
            r_start = 1
            r_end = r_size+1 if r_overhang > 0 else r_size
            for i in range(1, region):
                r_start = r_end + 1
                r_end = (r_end+r_size+1) if i < r_overhang else r_end+r_size
            check = False
            if fid in protein[tool]:
                for instance in protein[tool][fid]['instance']:
                    if (r_start <= instance[0] <= r_end) or (r_start <= instance[1] <= r_end):
                        pa_pattern.append(1)
                        check = True
                        break
                if not check:
                    pa_pattern.append(0)
            else:
                pa_pattern.append(0)
    return pa_pattern


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


def dataset_gen(path, annopath, group_name, root_dir):
    """
    TODO: This function could be broken down to smaller functions
    """
    file = open(os.path.join(path, group_name))
    group_data = json.load(file)
    # Positive data
    px = [group_data["data"][protein] for protein in group_data["data"]]
    py = [1 for i in range(0, len(px))]
    # Generate file
    outfile_path = os.path.join(root_dir,
                                  "networks",
                                  "citrate_cycle",
                                  "data",
                                  "datasets")
    outfile_px_path = os.path.join(outfile_path, f"{group_name.split('.')[0]}_px.tsv")
    outfile_py_path = os.path.join(outfile_path, f"{group_name.split('.')[0]}_py.tsv")
    outfile_px = open(outfile_px_path, "a")
    outfile_py = open(outfile_py_path, "a")
    for j, pa_pattern in enumerate(px):
        outfile_px.write('\t'.join([str(i) for i in pa_pattern])+'\n')
        outfile_py.write(str(py[j])+'\t')
    outfile_px.close()
    outfile_py.close()

    #Negative data
    negative = len(px)*4
    outfile_nx_path = os.path.join(outfile_path, f"{group_name.split('.')[0]}_nx.tsv")
    outfile_ny_path = os.path.join(outfile_path, f"{group_name.split('.')[0]}_ny.tsv")
    outfile_nx = open(outfile_nx_path, "a")
    outfile_ny = open(outfile_ny_path, "a")
    for i in range(0, negative):
        if i % 20 == 0: print(f"Negative data {i} of {negative}")
        random_file = os.listdir(annopath)[randint(len(os.listdir(path)))]
        f = open(os.path.join(annopath, random_file))
        negative_data = json.load(f)
        f.close()
        random_protein = [protein for protein in
                          negative_data["feature"]][randint(len([protein for protein in negative_data["feature"]]))]
        pa = matrix_gen_single(
            negative_data["feature"][random_protein],
            group_data["metadata"]["features"],
            group_data["metadata"]["regions"])
        outfile_nx.write('\t'.join([str(i) for i in pa]) + '\n')
        outfile_ny.write('0\t')
    outfile_nx.close()
    outfile_ny.close()
