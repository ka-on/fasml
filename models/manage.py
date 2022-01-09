import json

from models.dense import DenseLayersModel
from file_handling.input_gen import dataset_gen
import os


def master(root_dir, network, group):
    # Fetch data
    path_to_group_topology = os.path.join(root_dir, "networks", network, "topology.json")
    file = open(path_to_group_topology)
    data = json.load(file)
    # Construct model
    model = DenseLayersModel(data[group])
    # Get data
    bin_regions_path = os.path.join(root_dir, "networks", network, "data", "regions_binary")
    anno_path = os.path.join(root_dir, "networks", network, "data", "groups_annotations")
    x, y = dataset_gen(bin_regions_path, anno_path, group)
    save_weights_path = os.path.join(root_dir, "networks", network, f"{group.split('.')[0]}_weights")
    model.train(x, y, save_weights_path)
