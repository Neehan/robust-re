"""
Define common constants.
"""
from transformers import BertTokenizer


TRAIN_JSON = "train.json"
DEV_JSON = "dev.json"
TEST_JSON = "test.json"

GLOVE_DIR = "dataset/glove"
# BERT_MODEL = "bert-base-cased"

EMB_INIT_RANGE = 1.0
MAX_LEN = 300

# tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

# vocab
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"
PAD_ID, UNK_ID =  0, 1#tokenizer.convert_tokens_to_ids([PAD_TOKEN, UNK_TOKEN])

VOCAB_PREFIX = [PAD_TOKEN, UNK_TOKEN]

# hard-coded mappings from fields to ids
SUBJ_NER_TO_ID = {
    PAD_TOKEN: PAD_ID,
    UNK_TOKEN: UNK_ID,
    "ORG": 2,
    "LOC": 3,
    "FAC": 4,
    "WEA": 5,
    "GPE": 6,
    "PER": 7,
    "VEH": 8,
}

OBJ_NER_TO_ID = {
    PAD_TOKEN: PAD_ID,
    UNK_TOKEN: UNK_ID,   
    "ORG": 2,
    "LOC": 3,
    "FAC": 4,
    "WEA": 5,
    "GPE": 6,
    "PER": 7,
    "VEH": 8,
}


NER_TO_ID = {
    PAD_TOKEN: PAD_ID,
    UNK_TOKEN: UNK_ID,
    "ORG": 2,
    "LOC": 3,
    "FAC": 4,
    "WEA": 5,
    "GPE": 6,
    "PER": 7,
    "VEH": 8,
    "O": 9,
}

POS_TO_ID = {
    PAD_TOKEN: PAD_ID,
    UNK_TOKEN: UNK_ID,
    "NNP": 2,
    "NN": 3,
    "IN": 4,
    "DT": 5,
    ",": 6,
    "JJ": 7,
    "NNS": 8,
    "VBD": 9,
    "CD": 10,
    "CC": 11,
    ".": 12,
    "RB": 13,
    "VBN": 14,
    "PRP": 15,
    "TO": 16,
    "VB": 17,
    "VBG": 18,
    "VBZ": 19,
    "PRP$": 20,
    ":": 21,
    "POS": 22,
    "''": 23,
    "``": 24,
    "-RRB-": 25,
    "-LRB-": 26,
    "VBP": 27,
    "MD": 28,
    "NNPS": 29,
    "WP": 30,
    "WDT": 31,
    "WRB": 32,
    "RP": 33,
    "JJR": 34,
    "JJS": 35,
    "$": 36,
    "FW": 37,
    "RBR": 38,
    "SYM": 39,
    "EX": 40,
    "RBS": 41,
    "WP$": 42,
    "PDT": 43,
    "LS": 44,
    "UH": 45,
    "#": 46,
}

LABEL_TO_ID = {
    # "no_relation": 0,
    # "per:title": 1,
    # "org:top_members/employees": 2,
    # "per:employee_of": 3,
    # "org:alternate_names": 4,
    # "org:country_of_headquarters": 5,
    # "per:countries_of_residence": 6,
    # "org:city_of_headquarters": 7,
    # "per:cities_of_residence": 8,
    # "per:age": 9,
    # "per:stateorprovinces_of_residence": 10,
    # "per:origin": 11,
    # "org:subsidiaries": 12,
    # "org:parents": 13,
    # "per:spouse": 14,
    # "org:stateorprovince_of_headquarters": 15,
    # "per:children": 16,
    # "per:other_family": 17,
    # "per:alternate_names": 18,
    # "org:members": 19,
    # "per:siblings": 20,
    # "per:schools_attended": 21,
    # "per:parents": 22,
    # "per:date_of_death": 23,
    # "org:member_of": 24,
    # "org:founded_by": 25,
    # "org:website": 26,
    # "per:cause_of_death": 27,
    # "org:political/religious_affiliation": 28,
    # "org:founded": 29,
    # "per:city_of_death": 30,
    # "org:shareholders": 31,
    # "org:number_of_employees/members": 32,
    # "per:date_of_birth": 33,
    # "per:city_of_birth": 34,
    # "per:charges": 35,
    # "per:stateorprovince_of_death": 36,
    # "per:religion": 37,
    # "per:stateorprovince_of_birth": 38,
    # "per:country_of_birth": 39,
    # "org:dissolved": 40,
    # "per:country_of_death": 41,

    ## using subtype
    # using type
    
    "no_relation": 0,
    "PART-WHOLE(e1,e2)": 1,
    "PER-SOC(e1,e2)": 2,
    "PHYS(e1,e2)": 3,
    "GEN-AFF(e1,e2)": 4,
    "ART(e1,e2)": 5,
    "ORG-AFF(e1,e2)": 6,
    "PART-WHOLE(e2,e1)": 7,
    "PER-SOC(e2,e1)": 8,
    "PHYS(e2,e1)": 9,
    "GEN-AFF(e2,e1)": 10,
    "ART(e2,e1)": 11,
    "ORG-AFF(e2,e1)": 12,
}

INFINITY_NUMBER = 1e12
