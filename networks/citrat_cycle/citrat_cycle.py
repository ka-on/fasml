from networks.citrat_cycle.storage import CitrateStatic
from model_types.dense import DenseLayersModel
import os
import json
from ast import literal_eval


class CitrateCycle:
    all_features = CitrateStatic.ALL_FEATURES
    metrics = CitrateStatic.metrics
    shortest_protein_len = CitrateStatic.LENGTH_OF_SHORTEST_PROTEIN
    num_of_protein_functions = CitrateStatic.NUMBER_OF_PROTEIN_FUNCTIONS
    models_dict = {}


