import numpy as np
import json
from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier
import os
from os import path


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


def nearest_neighbors(k, word, voab, word2id, embedding):
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
        if g % 2 == 0:
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
    child = np.copy(parent1)
    p2_indices = np.random.choice([True, False], len(parent2))
    child[p2_indices] = parent2[p2_indices]
    return child


if __name__ == '__main__':

    vocab, word2id, embedding = load_glove_vocab(filename='dataset/glove/glove.840B.300d.txt', wv_dim=3)
    # a dummy sentence encoder (just sums up all word embeddings)
    sentence2vec = lambda sentence : np.sum([embedding[word2id[word]] for word in sentence], axis=0)
    # dummy training sentences
    train_sentences = {("i", "am", "happy") : 0, ("i", "am", "sad"): 1, ("i", "am", "very", "happy") : 2}
    print("computing sentence vectors")
    X = np.array([sentence2vec(sentence) for sentence in train_sentences])
    
    # "some relationship" : 0, "no relationship" : 1
    targets = np.array([1, 0, 1])

    r = RandomForestClassifier()
    print("fitting model")
    r.fit(X, targets)

    def model(sentence):
        vec = [sentence2vec(sentence)]
        return r.predict_proba(vec).reshape(-1)

    sen = ["i",  "am", "sad"]
    
    new_sen =  adv_ex(sentence=sen, target_id=1,  model=model, vocab=vocab, word2id=word2id, embedding=embedding, n_gen=100, n_pop=20, k=2)

    print(new_sen, model(sen), model(new_sen))



