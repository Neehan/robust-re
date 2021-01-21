import numpy as np
import os
import pickle
from utils import constant


class BERTVocab(object):
    """
    Vocabulary for the bert model
    publicly available methods and variables are:
        - size(): size of the vocabulary dict
        - save(filepath): save the vocabulary dict at filepath
        - word2id: a word (string) to its id in the vocab dict
        - id2word: vocab id to the corresponding word
    """

    def __init__(self):
        subj_entities = constant.SUBJ_NER_TO_ID.keys()
        # exclude the pad and the unk token and add SUBJ-
        subj_tokens = [
            entity
            for entity in subj_entities
            if entity not in {constant.PAD_TOKEN, constant.UNK_TOKEN}
        ]

        obj_entities = constant.OBJ_NER_TO_ID.keys()
        # exclude the pad and the unk token and add OBJ-
        obj_tokens = [
            entity
            for entity in obj_entities
            if entity not in {constant.PAD_TOKEN, constant.UNK_TOKEN}
        ]

        # add new tokens to tokenizer
        # produces a warning since the embedding is random
        constant.tokenizer.add_tokens(subj_tokens+obj_tokens)
        ids = range(len(constant.tokenizer))
        self.id2word = constant.tokenizer.convert_ids_to_tokens(ids)
        self.word2id = {tok:i for i, tok in enumerate(self.id2word)}

    def size(self):
        return len(constant.tokenizer)

    def save(self, filename):
        "save the tokenizer at the filepath"
        if os.path.exists(filename):
            print("Overwriting old vocab file at " + filename)
            os.remove(filename)
        with open(filename, 'wb') as outfile:
            pickle.dump(constant.tokenizer, outfile)
        return


