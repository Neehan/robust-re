import numpy as np
import json
from sklearn.neighbors import NearestNeighbors






"""
Run evaluation with saved models.
"""

import os
import random
import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

from data.loader import DataLoader
from model.rnn import RelationModel
from utils import torch_utils, scorer, constant, helper
from utils.vocab import Vocab

parser = argparse.ArgumentParser()
parser.add_argument('model_dir', type=str, help='Directory of the model.')
parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
parser.add_argument('--data_dir', type=str, default='dataset/tacred')
parser.add_argument('--dataset', type=str, default='test', help="Evaluate on dev or test.")
parser.add_argument('--out', type=str, default='', help="Save model predictions to this dir.")

parser.add_argument('--seed', type=int, default=1234)
parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
parser.add_argument('--cpu', action='store_true')
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(1234)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

# load opt
model_file = args.model_dir + '/' + args.model
print("Loading model from {}".format(model_file))
opt = torch_utils.load_config(model_file)
model = RelationModel(opt)
model.load(model_file)

# load vocab
vocab_file = args.model_dir + '/vocab.pkl'
vocab = Vocab(vocab_file, load=True)
assert opt['vocab_size'] == vocab.size, "Vocab size must match that in the saved model."

# load data
data_file = opt['data_dir'] + '/{}.json'.format(args.dataset)
print("Loading data from {} with batch size {}...".format(data_file, opt['batch_size']))
batch = DataLoader(data_file, opt['batch_size'], opt, vocab, evaluation=True)

helper.print_config(opt)
id2label = dict([(v,k) for k,v in constant.LABEL_TO_ID.items()])

predictions = []
all_probs = []
for i, b in enumerate(batch):
    preds, probs, _ = model.predict(b)
    predictions += preds
    all_probs += probs
predictions = [id2label[p] for p in predictions]
p, r, f1 = scorer.score(batch.gold(), predictions, verbose=True)

# save probability scores
if len(args.out) > 0:
    helper.ensure_dir(os.path.dirname(args.out))
    with open(args.out, 'wb') as outfile:
        pickle.dump(all_probs, outfile)
    print("Prediction scores saved to {}.".format(args.out))

print("Evaluation ended.")











def main():
	vocab, word2id, embedding = load_glove_vocab()

	def 



def load_glove_vocab(filename='dataset/glove/glove.840B.300d.txt', wv_dim=30):
    """
    Load all words from glove.
    """
    vocab = []
    embedding = []
    word2id = {}
    with open(filename, 'r+') as f:
        i = 0
        for line in f:
            elems = line.split()
            word = ""
            j = 0
            for x in elems:
                try:
                    j += 1
                    float(x)
                    break
                except ValueError:
                    word += x

            vocab.append(word)
            word2id[word] = i
            i += 1
            embedding.append(elems[j:])
    return np.array(vocab), word2id, np.array(embedding)


def knn(k, word, vocab, word2id, embedding):
    if word not in word2id.keys():
        return None
    knn = NearestNeighbors(k)
    knn.fit(embedding)
    nbrs = knn.kneighbors([embedding[word2id[word]]], k, return_distance=False)
    return vocab[nbrs[0]]



def adv_ex(sentence, target_relation, num_gen, num_pop, fitness_fn, predict_fn, k, vocab, word2id, embedding):
	"""
	sentence:        list of tokens
	target_relation: the relation class we want the model to predict
	
	num_gen:         number of generations
	num_pop:         population size
	
	fitness_fn:      fitness function(x, y), here it is the probability of 
	                 the model predicting y for input x
	
	predict_fn:      prediction (relation class) of the model for input sentence x
	         k:      number of similar words to look for
	    vocab:       list of all words
        word2id:     map of each word to its index in vocab
      embedding:     glove embedding of the words from vocab
	"""
	population = []
	fitness = []
	
	for i in range(num_pop):
		population.append(perturb(sentence, target_relation, fitness_fn, k, vocab, word2id, embedding))

	for g in range(num_gen):
		for i in range(num_pop):
			fitness.append(fitness_fn(population[i], target_relation))
		
		sentence_adv = np.argmax(fitness)
		if predict_fn(sentence_adv) == target_relation:
			return sentence_adv
		else:
			probs = np.array(fitness) / np.sum(fitness)
			children = []
			for i in range(num_pop):
				parent1, parent2 = np.random.choice(range(num_pop), 2, p=probs)
				child = crossover(parent1, parent2)
				children.append(perturb(child, target_relation, fitness_fn, k, vocab, word2id, embedding))
			population = children


def perturb(sentence, target_relation, fitness_fn, k, vocab, word2id, embedding):
	pos = np.random.choice(range(len(sentence)))
	sim_words = knn(k, sentence[pos], vocab, word2id, embedding)
	
	sim_words = language_model(sentence, pos, sim_words)

	fitness = []
	for sim_word in sim_words:
		fitness.append(fitness_fn(sentence[:pos]+sim_word+sentence[pos+1:], target_relation))

	best_substitute = np.argmax(fitness)
	return sentence[:pos]+sim_words[best_substitute]+sentence[pos+1:]


def language_model(sentence, pos, sim_words):
	return sim_words

def crossover(parent1, parent2):
	child = []
	for i in range(len(parent1)):
		choice = np.random.choice([1, 2])
		if choice == 1:
			child.append(parent1[i])
		else:
			child.append(parent2[i])
	return child
