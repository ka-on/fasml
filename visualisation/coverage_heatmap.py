import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from scripts.binary_representation import _get_occurring_features


def get_continious_instances(data, instance_eval_threshold, exclude_NA):
    fids = _get_occurring_features(data)

    instance_dict = {
        p: {
            feature: [] for feature in fids
        } for p in data['feature'].keys()
    }

    for p, pval in data['feature'].items():
        for f in fids:
            tool = 'coils2' if f.split('_')[0] == 'coils' else f.split('_')[0]
            if f in pval[tool]:
                instances = iter(pval[tool][f]['instance'])

                while True:
                    try:
                        instance = next(instances)

                        if instance[2] == 'NA':
                            if exclude_NA:
                                continue
                        elif instance[2] < instance_eval_threshold:
                            continue
                        instance_dict[p][f].append((instance[0], instance[1]))
                        break
                    except StopIteration:
                        break

                while True:
                    try:
                        instance = next(instances)
                        
                        if instance[2] == 'NA':
                            if exclude_NA:
                                continue
                        elif instance[2] < instance_eval_threshold:
                            continue

                        if instance[0] in range(instance_dict[p][f][-1][0], instance_dict[p][f][-1][1]+1):
                            instance_dict[p][f][-1] = (instance_dict[p][f][-1][0], instance[1])
                        else:
                            instance_dict[p][f].append((instance[0], instance[1]))
                    except StopIteration:
                        break

    return instance_dict


def calc_coverage(instance_dict, data):
    fids = _get_occurring_features(data)
    coverage_dict = {
        p :{
            f: 0.0 for f in fids
        } for p in instance_dict.keys()
    }

    for p, pval in instance_dict.items():
        for f, fval in pval.items():
            covered_area = 0
            for instance in fval:
                covered_area += instance[1] - instance[0]
            coverage_dict[p][f] = covered_area / data['feature'][p]['length']

    return coverage_dict
            

def to_heatmap(coverage_dict, fids, title):
    df = pd.DataFrame(
        data = coverage_dict,
        index = fids
    )
    plt.rcParams.update({'font.size': 3})
    plt.pcolor(df, vmin=0.0, vmax=1.0)
    plt.colorbar()
    plt.title(title)
    plt.yticks(
        np.arange(len(fids)),
        fids,
    )
    xticks_locations = (np.arange(1, len(coverage_dict.keys())+1, int(len(coverage_dict.keys())/10)) 
                        if len(coverage_dict.keys()) > 10 else np.arange(1, len(coverage_dict.keys())+1))
    xticks_labels = (list(i for i in range(1, len(coverage_dict.keys())+1, int(len(coverage_dict.keys())/10)))
                        if len(coverage_dict.keys()) > 10 else list(i for i in range(1, len(coverage_dict.keys())+1)))
    plt.xticks(
        xticks_locations,
        xticks_labels
    )
    plt.savefig(f"./visualisation/coverage_heatmaps/{title}.png", bbox_inches='tight', dpi=400)
    plt.clf()


def process_annotations(annopath, exclude, instance_eval_threshold=0, exclude_NA=False):
    for i, fname in enumerate(os.listdir(annopath)):
        if fname in exclude:
            continue
        print(f"Creating heatmaps from annotations: {fname} ({i}/{len(os.listdir(annopath))-len(exclude)})")
        f = open(os.path.join(annopath, fname))
        data = json.load(f)
        fids = _get_occurring_features(data)
        idict = get_continious_instances(
            data=data,
            instance_eval_threshold=instance_eval_threshold,
            exclude_NA=exclude_NA
        )
        cdict = calc_coverage(idict, data)
        to_heatmap(cdict, fids, fname.split('.')[0])


process_annotations(
    os.path.join(os.getcwd(), './data/tca/train/anno'),
    exclude=['mock.json']
)