import numpy as np
import json
import pickle
import argparse
import tqdm


def main(args):
	data_dir = args.data_dir
	dataset = args.dataset
	out_dir = args.out

	in_filepath = data_dir +'/' + dataset + '.json'
	out_filepath = out_dir + '/' + "ner_sym.pkl"

	sim_dict = dict()
	ignore_ner = {'O'}

	with open(in_filepath, 'r') as f:
		data = json.load(f)
		for sentence in tqdm.tqdm(data):
			last_ner = None
			for (ner_i, token_i) in zip(sentence["stanford_ner"], sentence["token"]):
				if ner_i not in ignore_ner:
					if sim_dict.get(ner_i) is None:
						sim_dict[ner_i] = []
					
					if ner_i != last_ner:	
						sim_dict[ner_i].append([token_i])
						last_ner = ner_i
					else:
						sim_dict[ner_i][-1].append(token_i)
						
		f.close()
	
	# separate by length and remove duplicates
	for key in sim_dict.keys():
		valdict = []
		for word in sim_dict[key]:
			while len(word) >= len(valdict):
				valdict.append(set())
			valdict[len(word)].add(tuple(word))
		sim_dict[key] = valdict

	# convert sets and tuples to list
	for key in sim_dict.keys():
		val = []
		for word_list in sim_dict[key]:
			valdict = [list(words) for words in word_list]
			val.append(valdict)
		sim_dict[key] = val


	for key in sim_dict.keys():
		for i in range(1, 3):
			print(f"key: {key:<15} size: {len(sim_dict[key][i]):<7} top 5: {str(sim_dict[key][i][:5]):<10}")

	with open(out_filepath, 'wb') as f:
		pickle.dump(sim_dict, f)
		f.close()


if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('--data_dir', type=str, default='dataset/tacred')
	parser.add_argument('--dataset', type=str, default='train', help="name of the dataset")
	parser.add_argument('--out', type=str, default='dataset/tacred', help="Save output to this dir.")
	args = parser.parse_args()

	main(args)


