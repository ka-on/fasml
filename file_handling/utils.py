import os
import json


def categorize_by_groups(path, output_path):
    # TODO: It's possible to make this more efficient
    i = 1
    full_progress = len(os.listdir(path))
    for f in os.listdir(path):
        print(f"Processing {f}; {i} of {full_progress}")
        file = open(os.path.join(path, f))
        data = json.load(file)

        for protein in data["feature"]:
            group_name = protein.split('#')[0]
            try:
                output_file = open(f"{output_path}/{group_name}.json", "r")
                output_data = json.load(output_file)
            except FileNotFoundError:
                output_file = open(f"{output_path}/{group_name}.json", "w+")
                output_data = {"feature": {}}
            output_file.close()

            output_file = open(f"{output_path}/{group_name}.json", "w")
            output_data["feature"][protein] = data["feature"][protein]
            output_file.write(json.dumps(output_data))
            output_file.close()
        i += 1


def shortest(data):
    return min([data["feature"][protein]["length"] for protein in data["feature"]])


def occurring_features(data):
    features = []
    for protein in data["feature"]:
        for feature in data["feature"][protein]:
            if data["feature"][protein][feature] != {} and feature != "length":
                for feature_type in data["feature"][protein][feature]:
                    if feature_type not in features:
                        features.append(feature_type)
    return features
