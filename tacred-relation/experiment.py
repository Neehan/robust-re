import pickle
import multiprocessing as mp
import time
import tqdm
import numpy as np
# from numba import jit, cuda, prange
# gmodel = KeyedVectors.load_word2vec_format('dataset/glove/glove.840B.300d.w2vformat.txt', binary=False)


with open("dataset/glove/gensim_glove.pkl", 'rb') as f:
    model = pickle.load(f)
    f.close()
    print('model loaded')

def worker2(words):
  for word in words:
      great_dict[word] = list(model.most_similar(word, topn=topn))

def worker(word):
  great_dict[word] = list(model.most_similar(word, topn=topn))

# do 100k vocab for temporary code
vocab = mp.Manager().list(np.random.choice(list(model.vocab), 100000, replace=False))
# vocab = mp.Manager().list([vocab[x:x+100] for x in range(0, len(vocab), 100)])

great_dict = mp.Manager().dict()
topn = 25
with mp.Pool(mp.cpu_count()) as pool:
  for _ in tqdm.tqdm(pool.imap_unordered(worker, vocab), total=len(vocab)):
  # for _ in tqdm.tqdm(pool.imap_unordered(worker2, vocab), total=len(vocab)):
      pass
  pool.close()
  pool.join()

with open("dataset/glove/similarity_dict_parallel_100k.pkl", 'wb') as d:
    pickle.dump(great_dict, d)
    d.close()


