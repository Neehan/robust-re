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

from data.loader import DataLoader
from model.rnn import RelationModel
from utils import torch_utils, scorer, constant, helper
from utils.vocab import Vocab


def load_glove_vocab(filename='dataset/glove/glove.840B.300d.txt', wv_dim=30):
    """
    Load all words from glove. wv_dim is the dimension of embedding to keep
    vocab: list of words
    word2id: map each word to a id
    embedding: find word embedding according to id
    """
    print("loading glove vocabulary...")
    vocab_path = "vocab.npy"
    embedding_path = "embedding.npy"
    if path.exists(vocab_path) and path.exists(embedding_path):
        vocab = np.load(vocab_path)
        embedding = np.load(embedding_path)[:, :wv_dim]
        word2id = dict(zip(vocab, range(len(vocab))))
    else:
        vocab = []
        word2id = {}
        embedding = []
        with open(filename, 'r+') as f:
            for line in f:
                if len(vocab) % 5000 == 0:
                    print("{} word embeddings loaded".format(len(vocab)))
                tokens = line.split(' ')
                word = tokens[0]
                embed = [float(x) for x in tokens[1:wv_dim+1]]
                word2id[word] = len(vocab)
                vocab.append(word)
                embedding.append(embed)
        f.close()
        vocab, embedding = np.array(vocab), np.array(embedding)
        np.save(vocab_path, vocab)
        np.save(embedding_path, embedding)
    print("completed.")

    return vocab, word2id, embedding


def nearest_neighbors(k, word, vocab, word2id, embedding):
    """
    find k nearest neighbors of a word from vocab
    return: neighbor words as np 1d vectors
    """
    if word not in word2id.keys():
        return np.array([word])
    knn = NearestNeighbors(k)
    knn.fit(embedding)
    nbr_ids = knn.kneighbors([embedding[word2id[word]]], k, return_distance=False)
    return vocab[nbr_ids.reshape(-1)]


def prob_target_id(model, sentence, target_id):
    """
    firnd probability of target_id
    """
    prob_target_ids = model(sentence)
    return prob_target_ids[target_id]

def adv_ex(sentence, target_id,  model, vocab, word2id, embedding, n_gen, n_pop, k):
    """
    create an adversarial example from sentence for a model such that it predicts
    the adversarial example to belong to class target_id

    sentence: list of tokens
    target_id: class id of the target relation we want the model to predict
    n_gen: number of generations
    n_pop: population size
    model: a function sentence that returns distribution of model's prediction probabilities for all relation classes
    k: number of similar words to look for
    vocab: list of all words
    word2id: map of each word to its index in vocab
    embedding: glove embedding of the words from vocab
    """
    population  = [perturb(sentence, target_id, model, k, vocab, word2id, embedding) for i in range(n_pop)]

    for g in range(n_gen):
        if g % 5 == 0:
            print("generation: {}".format(g))
        fitness = [prob_target_id(model, population[i], target_id) for i in range(n_pop)]
        adv_sentence = population[np.argmax(fitness)]
        
        if np.argmax(model(adv_sentence)) == target_id:
            return adv_sentence
        else:
            probs = np.array(fitness) / np.sum(fitness)
            children = []
            for i in range(n_pop):
                parent1, parent2 = np.random.choice(range(n_pop), 2, p=probs)
                child = crossover(population[parent1], population[parent2])
                children.append(perturb(child, target_id, model, k, vocab, word2id, embedding))
            population = children
    print("no adversarial example found. returning original sentence...")
    return sentence


def perturb(sentence, target_id, model, k, vocab, word2id, embedding):
    """
    sentence : a list of tokens
    """
    pos = np.random.choice(range(len(sentence)))
    
    # knn similar from glove
    sim_words = nearest_neighbors(k, sentence[pos], vocab, word2id, embedding)
    # similar from language nodel
    sim_words = language_model(sentence, pos, sim_words)

    fitness = []
    for sim_word in sim_words:
        new_sentence = sentence[:pos]+[sim_word]+sentence[pos+1:]
        fitness.append(prob_target_id(model, new_sentence, target_id))

    best_substitute = np.argmax(fitness)
    return sentence[:pos]+[sim_words[best_substitute]]+sentence[pos+1:]


def language_model(sentence, pos, sim_words):
    # google's language model, not used for now
    # TODO
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


def load_data(args):
    data_path = args.data_dir + '/' + args.dataset + '.json'
    adv_data_path = args.data_dir + '/' + args.dataset + '_adv.json'

    torch.manual_seed(1234)
    random.seed(1234)

    with open(data_path, 'r') as f:
        data = json.load(f)
        f.close()
    sentence = copy.deepcopy(data[0])
    label_id = constant.LABEL_TO_ID[data[0]['relation']]
    target_relation = 'per:age'
    target_id = constant.LABEL_TO_ID[target_relation]

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

    helper.print_config(opt)
    id2label = dict([(v,k) for k,v in constant.LABEL_TO_ID.items()])

    def modelfn(sen):
        adv_sen = copy.deepcopy(sentence)
        adv_sen['token'] = sen
        with open(adv_data_path, 'w') as adv:
            json.dump([adv_sen], adv)
            adv.close()

        batch = DataLoader(adv_data_path, 2, opt, vocab, evaluation=True)
        predictions = []
        all_probs = []
        for i, b in enumerate(batch):
            print(b)
            preds, probs, _ = model.predict(b)
            predictions += preds
            all_probs += probs
        print(all_probs)
        return all_probs[0]

    print("loaded the tacred model.")
    print("loading the attack model.")
    vocab_attack, word2id_attack, embedding_attack = load_glove_vocab(filename='dataset/glove/glove.840B.300d.txt', wv_dim=30)    
    print("loaded the attack model.")
    print(f"original sentence: {sentence['token']}")
    print(f"original relation: {data[0]['relation']}, target relation: {target_relation}")
    print("looking for adversarial example")
    new_sen =  adv_ex(sentence=sentence['token'], target_id=target_id,  model=modelfn, 
                      vocab=vocab_attack, word2id=word2id_attack, embedding=embedding_attack, n_gen=50, n_pop=100, k=10)
    print("adversarial example found.")
    print(new_sen)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_dir', type=str, default="saved_models/00", help='Directory of the model.')
    parser.add_argument('--model', type=str, default='best_model.pt', help='Name of the model file.')
    parser.add_argument('--data_dir', type=str, default='dataset/tacred')
    parser.add_argument('--dataset', type=str, default='test', help="Evaluate on dev or test.")
    parser.add_argument('--out', type=str, default='', help="Save model predictions to this dir.")

    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--cuda', type=bool, default=torch.cuda.is_available())
    parser.add_argument('--cpu', action='store_true')
    args = parser.parse_args()

    print(load_data(args))



