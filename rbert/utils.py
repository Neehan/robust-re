import logging
import os
import random

import numpy as np
import torch
from transformers import BertTokenizer
from collections import Counter

from official_eval import official_f1


ADDITIONAL_SPECIAL_TOKENS = ["<e1>", "</e1>", "<e2>", "</e2>"]


def get_label(args):
    return [
        label.strip()
        for label in open(
            os.path.join(args.data_dir, args.label_file), "r", encoding="utf-8"
        )
    ]


def load_tokenizer(args):
    tokenizer = BertTokenizer.from_pretrained(args.model_name_or_path)
    tokenizer.add_special_tokens(
        {"additional_special_tokens": ADDITIONAL_SPECIAL_TOKENS}
    )
    return tokenizer


def write_prediction(args, output_file, preds):
    """
    For official evaluation script
    :param output_file: prediction_file_path (e.g. eval/proposed_answers.txt)
    :param preds: [0,1,0,2,18,...]
    """
    relation_labels = get_label(args)
    with open(output_file, "w", encoding="utf-8") as f:
        for idx, pred in enumerate(preds):
            f.write("{}\t{}\n".format(8001 + idx, relation_labels[pred]))


def init_logger(args):
    logfile = args.model_dir+"/train.log"
    # clear old content of the log file
    open(logfile, 'w').close()
    
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[logging.FileHandler(logfile, 'a'), logging.StreamHandler()],
    )


def set_seed(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if not args.no_cuda and torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)


def compute_metrics(preds, labels):
    assert len(preds) == len(labels)
    return acc_and_f1(preds, labels)


def simple_accuracy(preds, labels):
    return (preds == labels).mean()


def acc_and_f1(preds, labels, average="macro", no_relation_id=0):
    acc = simple_accuracy(preds, labels)
    prec, rec, f1 = scorer(preds, labels, no_relation_id)
    return {
        "Accuracy %": acc * 100,
        "Precision (micro) %": prec * 100,
        "Recall (micro) %": rec * 100,
        "F1 (micro) %": f1 * 100,
    }


def scorer(preds, labels, no_relation_id, verbose=False):
    correct_by_relation = Counter()
    guessed_by_relation = Counter()
    gold_by_relation = Counter()

    # Loop over the data to compute a score
    for guess, gold in zip(preds, labels):
        if gold == no_relation_id and guess == no_relation_id:
            pass
        elif gold == no_relation_id and guess != no_relation_id:
            guessed_by_relation[guess] += 1
        elif gold != no_relation_id and guess == no_relation_id:
            gold_by_relation[gold] += 1
        elif gold != no_relation_id and guess != no_relation_id:
            guessed_by_relation[guess] += 1
            gold_by_relation[gold] += 1
            if gold == guess:
                correct_by_relation[guess] += 1

    # Print verbose information
    if verbose:
        # print("Per-relation statistics:")
        relations = gold_by_relation.keys()
        longest_relation = 0
        for relation in sorted(relations):
            longest_relation = max(len(relation), longest_relation)
        for relation in sorted(relations):
            # (compute the score)
            correct = correct_by_relation[relation]
            guessed = guessed_by_relation[relation]
            gold = gold_by_relation[relation]
            prec = 1.0
            if guessed > 0:
                prec = float(correct) / float(guessed)
            recall = 0.0
            if gold > 0:
                recall = float(correct) / float(gold)
            f1 = 0.0
            if prec + recall > 0:
                f1 = 2.0 * prec * recall / (prec + recall)
            # (print the score)
            sys.stdout.write(("{:<" + str(longest_relation) + "}").format(relation))
            sys.stdout.write("  P: ")
            if prec < 0.1:
                sys.stdout.write(" ")
            if prec < 1.0:
                sys.stdout.write(" ")
            sys.stdout.write("{:.2%}".format(prec))
            sys.stdout.write("  R: ")
            if recall < 0.1:
                sys.stdout.write(" ")
            if recall < 1.0:
                sys.stdout.write(" ")
            sys.stdout.write("{:.2%}".format(recall))
            sys.stdout.write("  F1: ")
            if f1 < 0.1:
                sys.stdout.write(" ")
            if f1 < 1.0:
                sys.stdout.write(" ")
            sys.stdout.write("{:.2%}".format(f1))
            sys.stdout.write("  #: %d" % gold)
            sys.stdout.write("\n")
        # print("")

    # Print the aggregate score
    # if verbose:
    #     print("Final Score:")
    prec_micro = 1.0
    if sum(guessed_by_relation.values()) > 0:
        prec_micro = float(sum(correct_by_relation.values())) / float(
            sum(guessed_by_relation.values())
        )
    recall_micro = 0.0
    if sum(gold_by_relation.values()) > 0:
        recall_micro = float(sum(correct_by_relation.values())) / float(
            sum(gold_by_relation.values())
        )
    f1_micro = 0.0
    if prec_micro + recall_micro > 0.0:
        f1_micro = 2.0 * prec_micro * recall_micro / (prec_micro + recall_micro)
    # print( "Precision (micro): {:.3%}".format(prec_micro) )
    # print( "   Recall (micro): {:.3%}".format(recall_micro) )
    # print( "       F1 (micro): {:.3%}".format(f1_micro) )
    return prec_micro, recall_micro, f1_micro
