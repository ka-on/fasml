import json
import os
from binary_representation import _to_binary_representation_single, _get_occurring_features


def _to_coverage_representation_single(protein, fids, regions):
    binary_representation = _to_binary_representation_single(protein, fids, protein['length'])
    coverage_representation = {
        f: [] for f in fids
    }
    r_size = protein['length'] // regions
    for f, fval in binary_representation.items():
        for i in range(regions):
            try:
                coverage_representation[f].append(sum(fval[i*r_size : (i+1)*r_size]) / r_size)
            except IndexError:
                coverage_representation[f].append(sum(fval[i*r_size :]) / (protein['length'] % regions))
    return coverage_representation


def _to_coverage_representation_multiple(proteome, pids, fids, regions):
    pa = {}
    for p in pids:
        pa[p] = _to_coverage_representation_single(proteome['feature'][p], fids, regions)
    return pa


def to_coverage_representation(data, multiple):
    pids = [protein for protein in data["feature"]]
    fids = _get_occurring_features(data)
    regions = min([data["feature"][protein]["length"] for protein in data["feature"]])

    metadata = {
        "regions": regions,
        "features": fids
    }

    return ((_to_coverage_representation_multiple(data, pids, fids, regions), metadata)
            if multiple else
            (_to_coverage_representation_single(data, pids, fids, regions), metadata))


def process_annotations(annopath, reprpath):
    for i, f in enumerate(os.listdir(annopath)):
        print(f"To Coverage Representation: {f} ({i+1} of {len(os.listdir(annopath))})")

        with open(os.path.join(annopath, f)) as file:
            data = json.load(file)
        
        coverage_representation, metadata = to_coverage_representation(
            data=data,
            multiple=True
        )

        output = {
            "metadata": metadata,
            "data": coverage_representation
        }

        with open(os.path.join(reprpath, 'coverage_representation', f), 'w+') as outfile:
            outfile.write(json.dumps(output))
