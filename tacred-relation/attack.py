import numpy as np
import json
import os.path
from os import path
import pickle

from sklearn.neighbors import NearestNeighbors
from sklearn.ensemble import RandomForestClassifier

def load_glove_vocab(filename='dataset/glove/glove.840B.300d.txt', wv_dim=3):
    """
    Load all words from glove.
    word2id: map each word to a id
    embedding: find word embedding according to id
    """
    print("loading glove vocabulary...")
    vocab_path = "dataset/vocab/vocab.pkl"
    embedding_path = "dataset/vocab/embedding.npy"
    if path.exists(vocab_path) and path.exists(embedding_path):
        vocab = np.array(pickle.load(vocab_path)).reshape(-1)
        print(vocab[:10])
        embedding = np.load(embedding_path)
        word2id = dict(zip(vocab, range(len(vocab))))
    else:
        glove = np.loadtxt(filename, dtype='str', comments=None, delimiter=' ')
        vocab = glove[:, 0].reshape(-1)
        word2id = dict(zip(vocab, range(len(vocab))))
        embedding = glove[:, 1:].astype('float')
    print("completed.")
    
    # np.save(vocab_path, vocab)
    # np.save("embedding.npy", embedding)

    return vocab, word2id, embedding


def nearest_neighbors(k, word, voab, word2id, embedding):
    """
    find nearest neighbors of a word
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
    sentence:        list of tokens
    target_id:          the relation class we want the model to predict
    
    n_gen:         number of generations
    n_pop:         population size
    
    fitness_fn:      fitness function(x, y), here it is the probability of 
                     the model predicting y for input x
    
    predict_fn:      prediction (relation class) of the model for input sentence x
             k:      number of similar words to look for
        vocab:       list of all words
        word2id:     map of each word to its index in vocab
      embedding:     glove embedding of the words from vocab
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
            for i in range(num_pop):
                parent1, parent2 = np.random.choice(range(n_pop), 2, p=probs)
                child = crossover(parent1, parent2)
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


if __name__ == '__main__':

    vocab, word2id, embedding = load_glove_vocab(filename='sample.txt', wv_dim=3)
    sentence2vec = lambda sentence : np.sum([embedding[word2id[word]] for word in sentence], axis=0)
    train_sentences = {("i", "am", "happy") : 0, ("i", "am", "sad"): 1, ("i", "am", "very", "happy") : 2}
    X = np.array([sentence2vec(sentence) for sentence in train_sentences])
    print(X)
    # "some relationship", "no relationship"
    targets = np.array([1, 0, 1])

    r = RandomForestClassifier()
    r.fit(X, targets)

    def model(sentence):
        vec = [sentence2vec(sentence)]
        return r.predict_proba(vec).reshape(-1)

    sen = ["i",  "am", "sad"]
    
    new_sen =  adv_ex(sentence=sen, target_id=1,  model=model, vocab=vocab, word2id=word2id, embedding=embedding, n_gen=5, n_pop=10, k=2)

    print(new_sen, model(sen), model(new_sen))



