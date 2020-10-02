import json
import pandas as pd 
import numpy as np 
import argparse
from tqdm import tqdm


"""
Sample p% data from the tacred/train.json dataset and save it as tacred/trainp.json
"""

def print_stats(d1, d2, dr1, dr2):
	for relation in sorted(dr1.keys()):
		count1, count2 = dr1[relation], dr2.get(relation, 0)
		print(f"{relation:>36}: {count1:>6} ({count1/len(d1)*100:>3.2f}%)  {relation:>36}: {count2:>6} ({count2/len(d2)*100:>3.2f}%)")
# print(newdata)


parser = argparse.ArgumentParser()
parser.add_argument('-d', '--datadir', help="directory to tacred data", type=str, default="tacred-relation/dataset/tacred")
parser.add_argument('-p', '--percent', help="percentage of data to sample", type=float, dest='p', default=0.2)
parser.add_argument('--dataname', help="dataset name", default="train_adv")
args = parser.parse_args()

p = args.p
datadir = args.datadir
dataname = args.dataname
ftrain = open(datadir+f"/{dataname}.json", 'r+')

data = json.load(ftrain)
data_relations = {}
ftrain.close()

newdata = []
newdata_relations = {}

for x in tqdm(data):
	if x['relation'] not in data_relations.keys():
		data_relations[x['relation']] = 0
	data_relations[x['relation']] += 1
	
	accept = np.random.choice([0, 1], p=[1-p, p])
	if accept:
		newdata.append(x)
		if x['relation'] not in newdata_relations.keys():
			newdata_relations[x['relation']] = 0
		newdata_relations[x['relation']] += 1


print("original dataset\t\t\t\t\t\t\tnew dataset")
print_stats(data, newdata, data_relations, newdata_relations)

outfilepath = datadir+f"/{dataname}{p*100:.0f}.json"
with open(outfilepath, 'w+') as fsample:
	json.dump(newdata, fsample)
	fsample.close()
	print(f"\nSaved to {outfilepath}")