from file_handling.input_gen import regions_binary_file_gen
import os
import json

path = "/home/onur/src/FASML/networks/citrat_cycle/data/groups_annotations"
outpath = "/home/onur/src/FASML/networks/citrat_cycle/data/regions_binary"

regions_binary_file_gen(path, outpath)