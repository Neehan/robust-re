import xml.etree.ElementTree as ET
import sys, re
import json
import nltk
from nltk import StanfordTagger
from nltk.tag import StanfordNERTagger
import stanza
from stanza.models.common.doc import Sentence
from tqdm import tqdm
import argparse
import numpy as np

STANZA_PARSER = stanza.Pipeline(lang="en", processors="tokenize,ner,pos,lemma,depparse")
DATA_DOMAINS = ["bn", "bc", "un", "wl", "cts", "nw"]

ALL_NER_TAGS = set()
ALL_RELATION_LABELS = set()
ALL_SUBJ_ENTITY_LABELS = set()
ALL_OBJ_ENTITY_LABELS = set()


def preprocess_ace(apf_tree_path, sgm_tree_path, subtype):
    """
    preprocess the ace 2005 data from xml format to find text, relations
    and entities from it.
    input: the apf.xml and .sgm data paths from ace2005 dataset
           subtype: whether to use subtypes of relations and entities
    output: 
        doc: text blob from the file
        rels: relations in the format {rel_id : (rel_id, rel_type, subj_id, obj_id)}
        named_entities: entities in the format {ne_id, ne_value}
    """
    apf_tree = ET.parse(apf_tree_path)
    apf_root = apf_tree.getroot()

    named_entities = {}
    check_nes = {}
    ne_starts = {}
    ne_ends = {}
    ne_map = {}
    for entity in apf_root.iter("entity"):

        # if subtype flag is set, try to use subtype whenever possible
        if subtype:
            ne_type = entity.attrib.get("SUBTYPE", entity.attrib["TYPE"])
        else:
            ne_type = entity.attrib["TYPE"]

        for mention in entity.iter("entity_mention"):
            ne_id = mention.attrib["ID"]
            for child in mention:
                if child.tag == "head":
                    for charseq in child:
                        start = int(charseq.attrib["START"])
                        end = int(charseq.attrib["END"]) + 1
                        text = re.sub(r"\n", r" ", charseq.text)
                        ne_tuple = (ne_type, start, end, text)
                        if ne_tuple in check_nes:
                            # sys.stderr.write("duplicated entity %s\n" % (ne_id))
                            ne_map[ne_id] = check_nes[ne_tuple]
                            continue
                        check_nes[ne_tuple] = ne_id
                        named_entities[ne_id] = [ne_id, ne_type, start, end, text]
                        if not start in ne_starts:
                            ne_starts[start] = []
                        ne_starts[start].append(ne_id)
                        if not end in ne_ends:
                            ne_ends[end] = []
                        ne_ends[end].append(ne_id)

    rels = {}
    check_rels = []
    for relation in apf_root.iter("relation"):

        # if subtype flag is set, try to use subtype whenever possible
        if subtype:
            rel_type = relation.attrib.get("SUBTYPE", entity.attrib["TYPE"])
        else:
            rel_type = relation.attrib["TYPE"]

        for mention in relation.iter("relation_mention"):
            rel_id = mention.attrib["ID"]
            rel = [rel_id, rel_type, "", ""]
            ignore = False
            for arg in mention.iter("relation_mention_argument"):
                arg_id = arg.attrib["REFID"]
                if arg.attrib["ROLE"] != "Arg-1" and arg.attrib["ROLE"] != "Arg-2":
                    continue
                if arg_id in ne_map:
                    arg_id = ne_map[arg_id]
                rel[int(arg.attrib["ROLE"][-1]) + 1] = arg_id
                if not arg_id in named_entities:
                    ignore = True
                    # ignored duplicated entity
            if ignore:
                sys.stderr.write("ignored relation %s\n" % (rel_id))
                continue
            if rel[1:] in check_rels:
                # sys.stderr.write("duplicated relation %s\n" % (rel_id))
                continue
            check_rels.append(rel[1:])
            rels[rel_id] = rel

    doc = open(sgm_tree_path).read()
    doc = re.sub(r"<[^>]+>", "", doc)
    doc = re.sub(r"(\S+)\n(\S[^:])", r"\1 \2", doc)

    offset = 0
    size = len(doc)
    current = 0
    regions = []
    for i in range(size):
        if i in ne_starts or i in ne_ends:
            inc = 0
            if (doc[i - 1] != " " and doc[i - 1] != "\n") and (
                doc[i] != " " and doc[i] != "\n"
            ):
                regions.append(doc[current:i])
                inc = 1
                current = i
            if i in ne_starts:
                for ent in ne_starts[i]:
                    named_entities[ent][2] += offset + inc
            if i in ne_ends:
                for ent in ne_ends[i]:
                    named_entities[ent][3] += offset
            offset += inc
    regions.append(doc[current:])
    doc = " ".join(regions)

    for ne in named_entities.values():
        if "\n" in doc[int(ne[2]) : int(ne[3])]:
            l = []
            l.append(doc[0 : int(ne[2])])
            l.append(doc[int(ne[2]) : int(ne[3])].replace("\n", " "))
            l.append(doc[int(ne[3]) :])
            doc = "".join(l)

    for ne in named_entities.values():
        assert doc[int(ne[2]) : int(ne[3])].replace("&AMP;", "&").replace(
            "&amp;", "&"
        ).replace(" ", "") == ne[4].replace(" ", ""), (
            "%s <=> %s" % (doc[int(ne[2]) : int(ne[3])], ne[4])
        )
    return doc, rels, named_entities


## Adib's code starts
def get_tags(stanza_sentence):
    """
    deprels: dependency relations
    poses: parts of Speeches
    ners: named entities
    """
    tokens = [token.text for token in stanza_sentence.tokens]
    deprels = [word.deprel for word in stanza_sentence.words]
    poses = [word.xpos for word in stanza_sentence.words]
    # ners = list(tuple(zip(*ner_tagger.tag(tokens)))[1])
    ners = [token.ner for token in stanza_sentence.tokens]
    # remove the prefix I-, B- etc
    ners = [ner[2:] if len(ner) > 1 and ner[1] == '-' else ner for ner in ners]
    return tokens, deprels, poses, ners


def match_entity(entity, entity_infer):
    return entity_infer.replace("&amp;", "&").replace(" ", "") == entity.replace(
        " ", ""
    ).replace("&amp;", "&")


def find_sentence_boundaries(stanza_sentences):
    """
    find the first and the last char positions of each stanza sentence
    in the text
    """
    sentence_starts = [sentence.tokens[0].start_char for sentence in stanza_sentences]
    sentence_ends = [sentence.tokens[-1].end_char for sentence in stanza_sentences]
    return sentence_starts, sentence_ends


def lower_bound(arr, num):
    idx = np.searchsorted(arr, num, side="left")
    if idx == len(arr) or arr[idx] > num:
        return idx - 1
    else:  # arr[idx] <= num
        return idx


def upper_bound(arr, num):
    idx = np.searchsorted(arr, num, side="left")
    # idx == len(arr) cannot occur
    # because it means num is bigger than all items in arr
    # when you search in sentence, there is an exact match
    # when you search in ends of sentences, the max is the
    # last char of the entire document.
    if arr[idx] < num:
        return idx + 1
    else:  # arr[idx] >= num
        return idx


def find_relation_sentence(
    stanza_sentences, sentence_starts, sentence_ends, subj, obj, filepath
):
    """
    given a list of stanza sentence, finds the sentence that contains
    the subject and the object of a relationship

    stanza_sentences: the list of stanza sentences from a text blob
    sentence_starts: the starting char position of each sentence in
                     the text
    sentence_ends: same for the ending chars
    subj: the subject entity, a tuple of (id, type, char start, char end, text)
          of the entity
    obj: same for the object
    filepath: filepath of the text blob

    returns: the sentence containing both entity, if it can't returns none
    """
    _, subj_type, subj_start, subj_end, subj_text = subj
    _, obj_type, obj_start, obj_end, obj_text = obj

    subj_sent_idx1 = lower_bound(sentence_starts, subj_start)
    subj_sent_idx2 = upper_bound(sentence_ends, subj_end)

    # make sure that the subject starts and ends in the same sentence
    if subj_sent_idx1 != subj_sent_idx2:
        return None

    # make sure that the object starts and ends in the same sentence
    obj_sent_idx1 = lower_bound(sentence_starts, obj_start)
    obj_sent_idx2 = upper_bound(sentence_ends, obj_end)

    # make sure that the object starts and ends in the same sentence
    if obj_sent_idx1 != obj_sent_idx2:
        return None

    if subj_sent_idx1 == obj_sent_idx1:
        return stanza_sentences[subj_sent_idx1]

    # so our sentence parsing were wrong and subject and object are in two
    # different sentences. thus, return nothing


def find_entity_position(text, stanza_sentence, char_start, char_end, filepath):
    """
    finds the token position of the entity given the char position in text
    param: text: full text blob
           stanza_sentence: a stanza sentence from text that contains the entity
           char_start, end: char position of the entity in the text
           filepath: the filepath of the text blob

    return:the entity position as the token index of the stanza sentence
           if the subject and the object are on two different sentences,
           return None
    """
    token_starts = [token.start_char for token in stanza_sentence.tokens]
    token_ends = [token.end_char for token in stanza_sentence.tokens]

    entity_start_idx = lower_bound(token_starts, char_start)
    entity_end_idx = upper_bound(token_ends, char_end)

    entity = text[char_start:char_end]
    entity_infer = "".join(
        [
            token.text
            # plus one because you want the token that has the end index in the slice.
            for token in stanza_sentence.tokens[entity_start_idx : entity_end_idx + 1]
        ]
    )

    # check if the token index actually gets the correct entity
    if match_entity(entity, entity_infer):
        # tacred uses [start, end] interval instead of [start, end), so this is fine
        return int(entity_start_idx), int(entity_end_idx)
    else:
        return None, None


def to_tacred_format(text, relations, named_entities, filepath):
    """
    given the text, relations and named entities, create a tacred formatted json file
    return the tacred formated list of datapoints
    """
    num_relations = len(relations.values())
    num_skipped = 0

    data_json = []
    stanza_sentences = STANZA_PARSER(text).sentences
    sentence_starts, sentence_ends = find_sentence_boundaries(stanza_sentences)

    for (rel_id, rel_type, subj_id, obj_id) in relations.values():
        subj_entity = named_entities[subj_id]
        obj_entity = named_entities[obj_id]

        # find the sentence containing subj and obj of the relation
        rel_sentence = find_relation_sentence(
            stanza_sentences=stanza_sentences,
            sentence_starts=sentence_starts,
            sentence_ends=sentence_ends,
            subj=subj_entity,
            obj=obj_entity,
            filepath=filepath,
        )

        # if we fail to find the sentence, then skip this relation
        if rel_sentence is None:
            num_skipped += 1
            continue

        # ace has entity char position, so convert it to
        # token position for tacred
        subj_start, subj_end = find_entity_position(
            text=text,
            stanza_sentence=rel_sentence,
            char_start=int(subj_entity[2]),
            char_end=int(subj_entity[3]),
            filepath=filepath,
        )
        obj_start, obj_end = find_entity_position(
            text=text,
            stanza_sentence=rel_sentence,
            char_start=int(obj_entity[2]),
            char_end=int(obj_entity[3]),
            filepath=filepath,
        )

        if subj_start == None or obj_start == None:
            num_skipped += 1
            continue

        tokens, deprels, poses, ners = get_tags(rel_sentence)

        relation = {
            "relation": rel_type,
            "id": rel_id,
            "subj": subj_entity[4],
            "obj": obj_entity[4],
            "subj_type": subj_entity[1],
            "subj_start": subj_start,
            "subj_end": subj_end,
            "obj_type": obj_entity[1],
            "obj_start": obj_start,
            "obj_end": obj_end,
            "stanford_pos": poses,
            "stanford_ner": ners,
            "stanford_deprel": deprels,
            "token": tokens,
        }

        ALL_NER_TAGS.update(ners)
        ALL_RELATION_LABELS.add(rel_type)
        ALL_SUBJ_ENTITY_LABELS.add(subj_entity[1])
        ALL_OBJ_ENTITY_LABELS.add(obj_entity[1])

        data_json.append(relation)
    return data_json, num_relations, num_skipped


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "filelist",
        help="dir list of .apf.xml and .sgm files. must have 6 .txt files for 6 domains.",
    )
    parser.add_argument("output", help="filepath for the output json")
    parser.add_argument(
        "-s", "--start", help="index of first file in filelist to process", default=0
    )
    parser.add_argument("-e", "--end", help="index of last file in filelist to process")
    parser.add_argument(
        "--subtype", help="use subtypes whenever possible", default=False
    )
    args = parser.parse_args()

    filelist = []

    for domain in DATA_DOMAINS:
        with open(args.filelist + f"{domain}.txt", "r") as inp:
            filelist += [filepath.strip() for filepath in inp.readlines()]
            inp.close()

    start_idx = int(args.start)
    filelist = filelist[int(args.start) :]

    if args.end:
        filelist = filelist[: int(args.end)]

    return filelist, args.output, start_idx, int(args.subtype)


if __name__ == "__main__":

    filelist, outputdir, start_idx, use_subtype = parse_args()
    output_json = dict(zip(DATA_DOMAINS, [[] for i in range(len(DATA_DOMAINS))]))

    outer_bar = tqdm(filelist, desc="files remaining")
    
    num_relations = 0
    num_skipped = 0

    for filepath in outer_bar:
        start_idx += 1
        text, relations, named_entities = preprocess_ace(
            filepath + ".apf.xml", filepath + ".sgm", use_subtype
        )

        tacred, curr_relations, curr_skipped = to_tacred_format(
            text, relations, named_entities, filepath
        )

        domain = filepath[8:10] if filepath[8:10] in DATA_DOMAINS else filepath[8:11]

        output_json[domain] += tacred
        num_relations += curr_relations
        num_skipped += curr_skipped
    
    for domain in DATA_DOMAINS:
        outpath = outputdir + f"{domain}.json"
        with open(outpath, "w") as out:
            json.dump(output_json[domain], out)
            out.close()

    print(f"number of relations: {num_relations}")
    print(f"number of relations skipped: {num_skipped}")
    
    # drop the I-, B-, E- etc prefix
    print("all ners:")
    print(f"{dict(zip(ALL_NER_TAGS, range(2, len(ALL_NER_TAGS)+2)))}")

    print("\nall relation labels:")
    print(f"{dict(zip(ALL_RELATION_LABELS, range(1, len(ALL_RELATION_LABELS)+1)))}")

    print("\nall subj entity labels:")
    print(f"{dict(zip(ALL_SUBJ_ENTITY_LABELS, range(2, len(ALL_SUBJ_ENTITY_LABELS)+2)))}")

    print("\nall obj entity labels:")
    print(f"{dict(zip(ALL_OBJ_ENTITY_LABELS, range(2, len(ALL_OBJ_ENTITY_LABELS)+2)))}")
