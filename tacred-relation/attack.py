import numpy as np
import copy
import json
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
import os
from os import path
import torch
import random

import random
import argparse
import pickle
import torch
import torch.nn as nn
import torch.optim as optim

from tqdm import tqdm

from data.loader import DataLoader
from model.rnn import RelationModel
from utils import torch_utils, scorer, constant, helper
from utils.vocab import Vocab

torch.manual_seed(1234)
random.seed(1234)
numpy.random.seed(1234)


def load_similarity_dict(filename='dataset/tacred/ner_sym.pkl'):
    """
    Load all words from glove. wv_dim is the dimension of embedding to keep
    vocab: list of words
    word2id: map each word to a id
    embedding: find word embedding according to id
    """
    print("loading similarity dict...")
    with open(filename, 'rb') as f:
        sim_dict = pickle.load(f)
        f.close()
    return sim_dict
    

# def nearest_neighbors(k, word, vocab, word2id, embedding):
#     """
#     find k nearest neighbors of a word from vocab
#     return: neighbor words as np 1d vectors
#     """

def find_similar(ner, word, sim_dict, topn=5):
    """
    return at most topn similar words to word from sim_dict
    """
    assert 0 < topn < 25
    if sim_dict.get(ner) is None or len(word) >= len(sim_dict[ner]):
        return []
    words = sim_dict[ner][len(word)]
    indices = np.random.choice(range(len(words)), topn, replace=False)
    # print("indices: ", indices)
    sim_words = np.array(words)[indices].tolist()
    # print(word, sim_words)
    return sim_words


def prob_relation_id(model, relation_id, inp):
    """
    firnd probability of relation_id
    """
    prob_relation_ids,_ = model(inp)
    return np.max([prob_relation_ids[i] for i in range(len(prob_relation_ids)) if i != relation_id])

def adv_ex(inp, relation_id,  model, sim_dict, n_gen, n_pop, k, n=1):
    """
    create an adversarial example from tokens for a model such that it predicts
    the adversarial example to belong to class relation_id

    tokens: list of tokens
    relation_id: class id of relation model originally predicts
    n_gen: number of generations
    n_pop: population size
    model: a function tokens that returns distribution of model's prediction probabilities for all relation classes
    k: number of similar words to look for
    vocab: list of all words
    word2id: map of each word to its index in vocab
    embedding: glove embedding of the words from vocab
    """
    population  = [perturb(inp, relation_id, model, sim_dict, k, n) for i in range(n_pop)]
    for g in tqdm(range(n_gen)):
        fitness = [prob_relation_id(model, relation_id, population[i]) for i in range(n_pop)]
        adv_inp = population[np.argmax(fitness)]
        
        if np.argmax(model(adv_inp)[0]) != relation_id:
            return adv_inp
        else:
            probs = np.array(fitness) / np.sum(fitness)
            children = []
            for i in range(n_pop):
                parent1, parent2 = np.random.choice(range(n_pop), 2, p=probs, replace=False)
                child = crossover(population[parent1], population[parent2])
                children.append(perturb(child, relation_id, model, sim_dict, k, n))
            population = children
    print("no adversarial example found. returning original tokens...")
    return inp

def join_ners(inp):
    """
    get all non obj ners and their start indices as list

    a ner phrase may contain multiple consecutive words
    """
    ners = inp['stanford_ner']
    tokens = inp['token']
    ignore_ner = ['O']
    last_ner = None
    new_tokens = []
    joint_ner = []
    for (ner_i, token_i) in zip(ners, tokens):
        if ner_i not in ignore_ner:
            if ner_i != last_ner:   
                new_tokens.append([token_i])
                last_ner = ner_i
            else:
                new_tokens[-1].append(token_i)
        else:
            new_tokens.append(token_i)
            last_ner = ner_i
    return new_tokens


def perturb(inp, relation_id, model, sim_dict, k, n=1):
    """
    tokens : a list of tokens, look at k similar words at each substitution location
    substitute n words
    """
    ners = inp['stanford_ner']
    tokens = inp['token']
    new_inp = copy.deepcopy(inp)
    joint_ner_tokens = join_ners(inp)
    
    i = 0
    for joint_token in joint_ner_tokens:
        if isinstance(joint_token, list):
            sim_words = find_similar(inp['stanford_ner'][i], joint_token, sim_dict, k)
            # use language model later
            fitness = []
            for sim_word in sim_words:
                # print("\noriginal: ", joint_token, "sim: ", sim_word)
                new_tokens = tokens[:i]+sim_word+tokens[i+len(joint_token):]
                new_inp['token'] = new_tokens
                # print(new_tokens)
                fitness.append(prob_relation_id(model, relation_id, new_inp))

            best_substitute = np.argmax(fitness)
            tokens = tokens[:i]+sim_words[best_substitute]+tokens[i+len(joint_token):]

            i += len(joint_token)
        else:
            i += 1
    new_inp['token'] = tokens
    return new_inp


def language_model(tokens, pos, sim_words):
    # google's language model, not used for now
    # TODO
    return sim_words

def crossover(parent1, parent2):
    child = copy.deepcopy(parent1)
    child['token'] = []
    child['stanford_ner'] = []
    for i in range(max(len(parent1['token']), len(parent2['token']))):
        choice = np.random.choice([1, 2])
        if choice == 1 and i < len(parent1['token']):
            child['token'].append(parent1['token'][i])
            child['stanford_ner'].append(parent1['stanford_ner'][i])
        elif choice == 2 and i < len(parent2['token']):
            child['token'].append(parent2['token'][i])
            child['stanford_ner'].append(parent2['stanford_ner'][i])
    return child


def load_data(args):
    data_path = args.data_dir + '/' + args.dataset + '.json'

    with open(data_path, 'r') as f:
        data = json.load(f)
        f.close()

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

    # helper.print_config(opt)
    id2label = dict([(v,k) for k,v in constant.LABEL_TO_ID.items()])

    def modelfn(inp):
        batch = DataLoader(json.dumps([inp]), 3, opt, vocab, evaluation=True, load_from_file=False)
        predictions = []
        all_probs = []
        for i, b in enumerate(batch):
            preds, probs, _ = model.predict(b)
            predictions += preds
            all_probs += probs
        predictions = [id2label[p] for p in predictions]
        return all_probs[0], predictions

    sim_dict = load_similarity_dict()    
    
    return data, sim_dict, modelfn

def compute_adversarial_dataset(args, data, sim_dict, modelfn):
    adv_data_path = args.data_dir + '/' + args.dataset + '_adv.json'
    adv_dataset = []
    i = 0
    for inp in np.random.choice(data, 100, replace=False):
        print(f"iter. {i}: original relation: {modelfn(inp)[1]}")
        print(f"original sentence: {inp['token']}")
        relation_id = np.argmax(modelfn(inp)[0])
        new_inp =  adv_ex(inp, relation_id,  model=modelfn, sim_dict=sim_dict, n_gen=args.n_gen, n_pop=args.n_pop, k=args.n_nbrs)
        print(f"iter. {i}: new predicted relation: {modelfn(new_inp)[1]}")
        print(f"new sentence: {new_inp['token']}")
        adv_dataset.append(new_inp)
        i += 1
    with open(adv_data_path, 'w') as f:
        json.dump(adv_dataset, f)
        f.close()
        print("saved at: ", adv_data_path)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default="saved_models/00", help='Directory of the model.')
    parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
    parser.add_argument('--data_dir', type=str, default='dataset/tacred')
    parser.add_argument('--dataset', type=str, default='train', help="Evaluate on dev or test.")
    parser.add_argument('--out', type=str, default='', help="Save model predictions to this dir.")

    parser.add_argument('--n_gen', type=int, default=15)
    parser.add_argument('--n_pop', type=int, default=20)
    parser.add_argument('--n_nbrs', type=int, default=10)
    
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    data, sim_dict, modelfn = load_data(args)
    compute_adversarial_dataset(args, data, sim_dict, modelfn)




