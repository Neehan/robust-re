import argparse
import json
from tqdm import tqdm
import numpy as np
from os import listdir
from os.path import isfile, join
np.random.seed(0)

parser = argparse.ArgumentParser()
parser.add_argument(
    "-p",
    "--processed-dir",
    dest="pd",
    default="processed",
    help="path to the processed ace2005 data directory",
)
parser.add_argument(
    "-rf",
    "--rel-file",
    dest="rf",
    help="path to the relations file",
    default="relations",
)
parser.add_argument(
    "-tr",
    "--train-pt",
    dest="train_pt",
    type=float,
    help="train percentile",
    default=0.6,
)
parser.add_argument(
    "-ts", "--test-pt", dest="test_pt", help="test percentile", type=float, default=0.2
)
args = parser.parse_args()

with open(args.rf, "r") as relfile:
    tacred_relation_map = json.load(relfile)
    relfile.close()

filepaths = [
    filepath for filepath in listdir(args.pd) if isfile(join(args.pd, filepath))
]


data_split = {"train": [], "test": [], "dev": []}
ace_entity_types = {"WEAPON", "VEHICLE", "FACILITY", "CONTACT_INFO", "GPE"}

assert float(args.train_pt) + float(args.test_pt) < 1

for filepath in tqdm(filepaths):
    with open(args.pd+"/"+filepath, "r") as file:
        data = json.load(file)
        file.close()

    for x in data:
        if x["subj_type"] in ace_entity_types or x["obj_type"] in ace_entity_types:
            continue

        # some ace relations are already mapped to tacred type relations
        # map remaining ace relations to tacred relations
        if x["relation"] in tacred_relation_map.keys():
            x["relation"], flip = tacred_relation_map[x["relation"]]
            if flip:
                # swap subject and object fields
                x["subj"], x["obj"] = x["obj"], x["subj"]
                x["subj_type"], x["obj_type"] = x["obj_type"], x["subj_type"]
                x["subj_start"], x["obj_start"] = x["obj_start"], x["subj_start"]
                x["subj_end"], x["obj_end"] = x["obj_end"], x["subj_end"]

        # now keep only tacred type relations
        allowed_relations = [v[0] for v in tacred_relation_map.values()]


        if x["relation"] not in allowed_relations:
            continue

        choice = np.random.choice(
            list(data_split.keys()),
            p=[args.train_pt, args.test_pt, 1 - args.train_pt - args.test_pt],
        )

        # downsample per:employee_of by half because there is too much of it
        if x["relation"] == "per:employee_of":
            if np.random.choice([0, 1]):
                continue
        data_split[choice].append(x)

print("# samples")
print(f"train:{len(data_split['train']):<5}", 
    f"dev:{len(data_split['dev']):<5}", 
    f"test:{len(data_split['test']):<5}"
)

rel_stats = dict()
for dataset in data_split.values():
    for x in dataset:
        rel_stats[x["relation"]] = rel_stats.get(x["relation"], 0) + 1

for rel, count in rel_stats.items():
    print(f"{rel:<30}: {count/sum(rel_stats.values())*100:<.2f}%")

print("writing to disk...")
for data_partition in data_split.keys():
    with open(data_partition + ".json", "w") as datafile:
        json.dump(data_split[data_partition], datafile)
        datafile.close()
