import json
import os


def _get_occurring_features(data):
    features = []
    for protein in data["feature"]:
        for feature in data["feature"][protein]:
            if data["feature"][protein][feature] != {} and feature != "length":
                for feature_type in data["feature"][protein][feature]:
                    if feature_type not in features:
                        features.append(feature_type)
    return features


def _to_binary_representation_single(protein, fids, regions):
    full_dict = {
        f: [
            0 for i in range(protein['length'])
        ] for f in fids
    }

    for f in fids:
        tool = 'coils2' if f.split('_')[0] == 'coils' else f.split('_')[0]
        if f in protein[tool]:
            for instance in protein[tool][f]['instance']:
                for i in range(instance[0]-1, instance[1]):
                    full_dict[f][i] = 1
    
    r_size = protein['length'] // regions
    pa = {
        f: [] for f in fids
    }

    for f, fval in full_dict.items():
        for i in range(regions):
            try:
                pa[f].append(int(1 in fval[i*r_size : (i+1)*r_size]))
            except IndexError:
                pa[f].append(int(1 in fval[f][i*r_size :]))
    
    return pa


def _to_binary_representation_multiple(proteome, pids, fids, regions):
    pa = {}
    for p in pids:
        pa[p] = _to_binary_representation_single(proteome['feature'][p], fids, regions)
    return pa


def to_binary_representation(data, multiple):
    pids = [protein for protein in data["feature"]]
    fids = _get_occurring_features(data)
    regions = min([data["feature"][protein]["length"] for protein in data["feature"]])

    metadata = {
        "regions": regions,
        "features": fids
    }

    return ((_to_binary_representation_multiple(data, pids, fids, regions), metadata)
            if multiple else
            (_to_binary_representation_single(data, pids, fids, regions), metadata))


def process_annotations(annopath, reprpath):
    for i, f in enumerate(os.listdir(annopath)):
        print(f"To Binary Representation: {f} ({i+1} of {len(os.listdir(annopath))})")

        with open(os.path.join(annopath, f)) as file:
            data = json.load(file)
        
        binary_representation, metadata = to_binary_representation(
            data=data,
            multiple=True
        )

        output = {
            "metadata": metadata,
            "data": binary_representation
        }

        with open(os.path.join(reprpath, 'binary_representation', f), 'w+') as outfile:
            outfile.write(json.dumps(output))
