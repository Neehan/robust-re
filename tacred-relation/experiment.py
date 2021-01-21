import json

filedir = "dataset/ace2005/tacred/"

with open(filedir+"train.json", 'r') as f:
	a = json.load(f)
	f.close()


subj = sen['subj']
obj = sen['obj']
token = sen['token']
subj_start, subj_end = sen['subj_start'], sen['subj_end']
obj_start, obj_end = sen['obj_start'], sen['obj_end']

token[subj_start] = '<<<'+token[subj_start]
token[subj_end] = token[subj_end] + '>>>'

token[obj_start] = '<<<'+token[obj_start]
token[obj_end] = token[obj_end] + '>>>'

print(f"subject: {subj} ({subj_start},{subj_end})")
print(f"object: {obj} ({obj_start},{obj_end})")
print(f"sentence: {' '.join(token)}")

# import numpy as np
# import matplotlib.pyplot as plt
# import torch
# import json
# from torch.autograd import Variable
# from collections import Counter

# from transformers import BertTokenizer, BertModel

# tokenizer = BertTokenizer.from_pretrained('bert-base-cased', never_split=["[PUTU]"])
# print(len(tokenizer))
# model = BertModel.from_pretrained('bert-base-cased')
# model.eval()
# text = "I'm your [UNK] [PUTU]"

# print(tokenizer.vocab["I"])

# input_ids = torch.tensor(tokenizer.encode(text)).unsqueeze(0)  # Batch size 1
# print(tokenizer.tokenize(text))
# outputs = model(input_ids)
# last_hidden_states = outputs[0]  # The last hidden-state is the first element of the output tuple

# print(input_ids)
# # print(input_ids.size())
# # print(last_hidden_states.size())



"""
Train a debiased model.
"""
# import argparse
# import numpy as np
# import os
# import pickle
# import random
# import time
# import torch
# import json
# import torch.nn as nn
# import torch.optim as optim

# from data.loader import DataLoader
# from datetime import datetime
# from model.rnn import RelationModel
# from shutil import copyfile
# from utils import torch_utils, scorer, constant, helper
# from utils.vocab import Vocab

# parser = argparse.ArgumentParser()
# parser.add_argument("--amodel_dir", type=str, help="Directory of the biased model.")
# parser.add_argument("--bmodel_dir", type=str, help="Directory of the biased model.")
# parser.add_argument(
#     "--model", type=str, default="best_model.pt", help="Name of the biased model file."
# )
# parser.add_argument("--data_dir", type=str, default="dataset/ace2005/tacred_coarse")
# parser.add_argument("--data_name", type=str, default="train.json")
# parser.add_argument('--seed', type=int, default=1234)
# parser.add_argument("--cuda", type=bool, default=torch.cuda.is_available())
# parser.add_argument("--cpu", action="store_true")



# def get_model_class_probs(args, model_dir):
#     # load opt
#     model_file = model_dir + "/" + args.model
#     print("Loading model from {}".format(model_file))
#     opt = torch_utils.load_config(model_file)
#     model = RelationModel(opt)
#     model.load(model_file)

#     # load vocab
#     vocab_file = model_dir + "/vocab.pkl"
#     vocab = Vocab(vocab_file, load=True)
#     assert opt["vocab_size"] == vocab.size, "Vocab size must match that in the saved model."
#     opt["vocab_size"] = vocab.size
#     emb_file = opt["vocab_dir"] + "/embedding.npy"
#     emb_matrix = np.load(emb_file)
#     assert emb_matrix.shape[0] == vocab.size
#     assert emb_matrix.shape[1] == opt["emb_dim"]

#     # load data
#     data_file = args.data_dir + "/{}".format(args.data_name)
#     print("Loading data from {} with batch size {}...".format(data_file, opt["batch_size"]))
#     batch = DataLoader(data_file, opt["batch_size"], opt, vocab, evaluation=True)
#     with open(data_file, "r") as infile:
#         data_json = json.load(infile)

#     # helper.print_config(opt)
#     id2label = dict([(v, k) for k, v in constant.LABEL_TO_ID.items()])

#     all_probs = []
#     predictions = []
#     for i, b in enumerate(batch):
#         preds, probs, _ = model.predict(b)
#         all_probs.append(probs)
#         predictions += [id2label[pred] for pred in preds]
#     return data_json, predictions, all_probs


# args = parser.parse_args()

# torch.manual_seed(args.seed)
# random.seed(1234)
# if args.cpu:
#     args.cuda = False
# elif args.cuda:
#     torch.cuda.manual_seed(args.seed)

# # get biased model class probs per batch
# data_json, a_preds, _ = get_model_class_probs(args, args.amodel_dir)
# _, b_preds, _ = get_model_class_probs(args, args.bmodel_dir)

# K = 100
# gold = [d["relation"] for d in data_json[:K]]
# subj = [d["subj"] for d in data_json[:K]]
# obj = [d["obj"] for d in data_json[:K]]
# sentences = [" ".join(d["token"]) for d in data_json[:K]]

# for i in range(K):
#     if b_preds[i] != a_preds[i]:
#         print(sentences[i])
#         print(subj[i], " --> ", obj[i])
#         print("gold       a_pred          b_pred")
#         print(f"{gold[i]:<10} {a_preds[i]:<10} {b_preds[i]:<10}")
#         print("")