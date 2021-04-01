"""
Run evaluation with saved models.
"""

import random
import argparse
import torch
import numpy as np

from data.loader import DataLoader
from model.rnn import RelationModel
from utils import torch_utils, scorer, constant
from utils.vocab import Vocab

parser = argparse.ArgumentParser()
parser.add_argument("model_dir", type=str, help="Directory of the model.")
parser.add_argument(
    "--model", type=str, default="best_model.pt", help="Name of the model file."
)
parser.add_argument("--data_dir", type=str, default="dataset/ace2005/final")
parser.add_argument(
    "--dataset", type=str, default="test", help="Evaluate on dev or test."
)
parser.add_argument(
    "--out", type=str, default="", help="Save model predictions to this dir."
)

parser.add_argument("--seed", type=int, default=1234)
parser.add_argument("--cuda", type=bool, default=torch.cuda.is_available())
parser.add_argument("--cpu", action="store_true")
args = parser.parse_args()

torch.manual_seed(args.seed)
random.seed(1234)
if args.cpu:
    args.cuda = False
elif args.cuda:
    torch.cuda.manual_seed(args.seed)

# load opt
model_file = args.model_dir + "/" + args.model
print("Loading model from {}".format(model_file))
opt = torch_utils.load_config(model_file)
model = RelationModel(opt)
model.load(model_file)

# load vocab
vocab_file = args.model_dir + "/vocab.pkl"
vocab = Vocab(vocab_file, load=True)
assert opt["vocab_size"] == vocab.size, "Vocab size must match that in the saved model."

id2label = dict([(v, k) for k, v in constant.LABEL_TO_ID.items()])

# load data
def get_scores(data_file, opt, vocab, model):
    print(
        "Loading data from {} with batch size {}...".format(
            data_file, opt["batch_size"]
        )
    )
    batch = DataLoader(data_file, opt["batch_size"], opt, vocab, evaluation=True)

    predictions = []
    all_probs = []
    for i, b in enumerate(batch):
        preds, probs, attn_weights, _ = model.predict(b)
        predictions += preds
        all_probs += probs
    predictions = [id2label[p] for p in predictions]
    
    # print("predictions")
    # for a, b in zip(batch.gold(), predictions):
    # 	print(f"{a:<28} {b:<28}")

    p, r, f1 = scorer.score(batch.gold(), predictions, verbose=False)
    return p, r, f1


# data_file = args.data_dir + "/{}.json".format("rationale_synthetic_train")
# batch = DataLoader(data_file, opt["batch_size"], opt, vocab, evaluation=True)

# for t in range(30):
#     b = batch[t]
#     wordids = b[0]
#     masks = b[1]
#     relation = [id2label[int(x)] for x in b[6]]
#     # input is sorted by length, so don't unsort it since we need attn
#     preds, probs, attn_weights, _ = model.predict(b, unsort=False)

#     for i in range(len(b)):
#         if relation[i] == 'no_relation':
#             continue
#         print("Relation: ", relation[i])
#         tokens = [vocab.id2word[y] for y in wordids[i]]
#         attn = [np.round(float(x), 3) for x in attn_weights[i]]
#         attn_idx = set(np.argpartition(attn, -10)[-10:])
#         # print(np.array(attn)[attn_idx])
#         mal = []
#         for j, token in enumerate(tokens):
#             if j in attn_idx:
#                 mal.append(token)
#                 print(attn[j], tokens[j], sep='\t')
#             else:
#                 mal.append("_")
    
#         print(" ".join(mal), "\n")

datasets = [
    "rationale_train",
    "rationale_dev",
    "rationale_un",
    "rationale_wl",
    "rationale_cts",
    "rationale_bc",
]

f1s = []
for dataset in datasets:
    data_file = args.data_dir + "/{}.json".format(dataset)
    _, _, f1 = get_scores(data_file, opt, vocab, model)
    f1s.append(f1)

print("model id\ttrain\tdev\tun\twl\tcts\tbc\tcomments")
print(opt["id"], end="\t")
for f1 in f1s:
    print(f"{f1*100:<.3f}", end="\t")

print("\n")
print("Evaluation ended.")
