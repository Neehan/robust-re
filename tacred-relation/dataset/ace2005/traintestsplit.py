import argparse
import json
from tqdm import tqdm
import numpy as np
from os import listdir
from os.path import isfile, join
import re

np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--processed-dir", "-pd",
    dest="pdir",
    default="tacred_coarse/",
    help="path to the processed ace2005 data directory",
)
parser.add_argument(
    "--outdir", "-o", default="tacred_coarse/", help="path to the output directory",
)
args = parser.parse_args()

DATA_DOMAINS = ["bn", "bc", "cts", "un", "nw", "wl"]


datasets = {
    "train": [], # bn + nw
    "dev" : [], # 1/2 of bc
    "bc_test": [], # other half of 1/2 of bc
}

# create train
for domain in ("bn", "nw"):
    with open(args.pdir + "" + domain + ".json", "r") as file:
        data = json.load(file)
        datasets["train"] += data
        file.close()

with open(args.pdir + "bc.json", "r") as file:
    data = json.load(file)
    for x in data:
        choice = np.random.choice(["dev", "bc_test"])
        datasets[choice].append(x) 
    file.close()


print("writing to disk...")
for dataset_name in datasets.keys():
    with open(args.outdir + dataset_name + ".json", "w") as datafile:
        json.dump(datasets[dataset_name], datafile)
        datafile.close()
