import tensorflow
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = "2"
import json

ROOT_DIR = os.getcwd()
path = os.path.join(ROOT_DIR, "networks", "citrate_cycle", "data", "regions_binary")
annopath = os.path.join(ROOT_DIR, "networks", "citrate_cycle", "data", "groups_annotations")
topologypath = os.path.join(ROOT_DIR, "networks", "citrate_cycle", "topology.json")
datasetpath = os.path.join(ROOT_DIR, "networks", "citrate_cycle", "data", "datasets")
weightspath = os.path.join(ROOT_DIR, "networks", "citrate_cycle", "weights")

GROUP = "K17753.json"
topology = json.load(open(topologypath, 'r'))
#model = DenseLayersModel(topology[GROUP], GROUP.split('.')[0])

"""
for g in topology:
    g = GROUP
    model = DenseLayersModel(topology[g], g.split('.')[0])
    px = open(os.path.join(datasetpath, g.split('.')[0]+'_px.tsv'))
    px_length = sum(1 for line in px)
    training_length = int(px_length*0.9)
    eval_length = px_length - training_length
    px.close()
    px = open(os.path.join(datasetpath, g.split('.')[0]+'_px.tsv'), 'r')
    nx = open(os.path.join(datasetpath, g.split('.')[0]+'_nx.tsv'), 'r')
    print("test")
    model.train(px, nx, weightspath, path, training_length, eval_length)
    break
"""
for g in topology:
    print(g)
    f = open(os.path.join(datasetpath, f"{g.split('.')[0]}_px.tsv"))
    fanno = open(os.path.join(path, g))
    data = json.load(fanno)
    print("regions", data["metadata"]["regions"])
    print("sequences", sum([1 for line in f])*4)