import xml.etree.ElementTree as ET
import sys, re
import json
import nltk
from nltk import StanfordTagger
from nltk.tag import StanfordNERTagger
import stanza
from tqdm import tqdm
import argparse

stanza_parser = stanza.Pipeline(lang="en", processors="tokenize,pos,lemma,depparse")
ner_tagger = StanfordNERTagger(
    "/afs/csail.mit.edu/u/n/notadib/stanford-ner/classifiers/english.all.3class.distsim.crf.ser.gz",
    "/afs/csail.mit.edu/u/n/notadib/stanford-ner/stanford-ner-4.0.0.jar",
    encoding="utf-8",
)

entity_subtypes = {"State-or-Province", "Nation", "URL"}

tacred_entity_map = {
    "PER": "PERSON",
    "ORG": "ORGANIZATION",
    "LOC": "LOCATION",
    "VEH": "VEHICLE",
    "WEA": "WEAPON",
    "GPE": "GPE",
    "FAC": "FACILITY",
    "Contact_Info": "CONTACT_INFO",
    "State-or-Province": "STATE_OR_PROVINCE",
    "Nation": "COUNTRY",
    "URL": "URL",
    "Numeric": "NUMBER",
    "Crime": "CRIMINAL_CHARGE",
    "Job-Title": "TITLE",
    "Sentence": "MISC",
}


def preprocess_ace(apf_tree_path, sgm_tree_path):
    """
    preprocess the ace 2005 data from xml format to find text, relations
    and entities from it.
    input: the apf.xml and .sgm data paths from ace2005 dataset
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
        ne_type = (entity.attrib["TYPE"],)
        try:
            ne_subtype = (entity.attrib["SUBTYPE"],)
        except KeyError:
            ne_subtype = ne_type

        # we use the generic entity type for
        # most things, except a few special ones
        if ne_subtype in entity_subtypes:
            ne_type = ne_subtype

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
                            sys.stderr.write("duplicated entity %s\n" % (ne_id))
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
        try: 
            rel_type = relation.attrib["SUBTYPE"]
        except KeyError:
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
                sys.stderr.write("duplicated relation %s\n" % (rel_id))
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
def get_tokens(doc_stanza):
    return [
        token.text for sentence in doc_stanza.sentences for token in sentence.tokens
    ]


def get_deprels(doc_stanza):
    # dependecy relations among words
    return [word.deprel for sentence in doc_stanza.sentences for word in sentence.words]


def get_poses(doc_stanza):
    # POS of the tokens
    return [word.xpos for sentence in doc_stanza.sentences for word in sentence.words]


def get_ners(doc_stanza):
    # named entity recognition
    tokens = [
        token.text for sentence in doc_stanza.sentences for token in sentence.tokens
    ]
    return list(tuple(zip(*ner_tagger.tag(tokens)))[1])
    # return [token.ner for sentence in doc_stanza.sentences for token in sentence.tokens]


def match_entity(entity, entity_infer):
    return "".join(entity_infer).replace("&amp;", "&").replace(
        " ", ""
    ) == entity.replace(" ", "").replace("&amp;", "&")


def find_entity_position(
    text, tokens, entity_id, entity, letter_start, letter_end, filepath
):
    """
    ACE 2005 has letter position of the entities in the text
    convert that to word position for the tacred format
    """
    word_start = len(get_tokens(stanza_parser(text[:letter_start])))
    word_end = len(get_tokens(stanza_parser(text[:letter_end])))

    # this is an approximate position, because tokenizer uses ai
    # resize the window upto +- 2
    break_loop = False
    for size in range(6):
        # shift the window upto +- 3 places
        for shift in range(10):
            new_start = word_start + (-1) ** shift * shift // 2
            new_start = max(0, new_start)
            new_start = min(new_start, len(tokens) - 1)

            new_end = word_end + (-1) ** size * size // 2 + (-1) ** shift * shift // 2
            new_end = max(0, new_end)

            if match_entity(entity, tokens[new_start:new_end]):
                word_start = new_start
                word_end = new_end
                break_loop = True
                break
        if break_loop:
            break

    # assert that inferred entity is same as the given entity
    entity_infer = "".join(tokens[word_start:word_end])
    assert entity_infer.replace("&amp;", "&").replace(" ", "") == entity.replace(
        " ", ""
    ), (
        "%s <=> %s: %s, %s\n%s"
        % (entity_infer, entity, word_start, word_end, tokens[: word_end + 5])
    )

    # tacred expects interval [start, end], not [start, end)
    return word_start, word_end - 1


def to_tacred_format(doc, relations, named_entities, filepath):
    """
    given the text, relations and named entities, create a tacred formatted json file
    return the tacred formated list of datapoints
    """
    data_json = []
    doc_stanza = stanza_parser(doc)
    tokens = get_tokens(doc_stanza)
    deprels = get_deprels(doc_stanza)
    poses = get_poses(doc_stanza)
    ners = get_ners(doc_stanza)

    assert len(tokens) == len(
        deprels
    ), f"len tokens: {len(tokens)} <=> len words: {len(deprels)}"

    for (rel_id, rel_type, subj_id, obj_id) in relations.values():    
        subj_entity = named_entities[subj_id]
        obj_entity = named_entities[obj_id]

        # map the subject and object entity to tacred type
        if tacred_entity_map.get(subj_entity[1][0]) is None:
            print(f"{subj_entity[1]} is not mapped to tacred!")
        else:
            subj_entity = [
                subj_entity[0],
                tacred_entity_map.get(subj_entity[1][0]),
            ] + subj_entity[2:]

        if tacred_entity_map.get(obj_entity[1][0]) is None:
            print(f"{obj_entity[1][0]} is not mapped to tacred!")
        else:
            obj_entity = [
                obj_entity[0],
                tacred_entity_map.get(obj_entity[1][0]),
            ] + obj_entity[2:]

        # ace has entity letter position, so convert it to
        # token position for tacred
        subj_start, subj_end = find_entity_position(
            text=doc,
            tokens=tokens,
            entity_id=subj_entity[0],
            entity=subj_entity[4],
            letter_start=int(subj_entity[2]),
            letter_end=int(subj_entity[3]),
            filepath=filepath,
        )
        obj_start, obj_end = find_entity_position(
            text=doc,
            tokens=tokens,
            entity_id=obj_entity[0],
            entity=obj_entity[4],
            letter_start=int(obj_entity[2]),
            letter_end=int(obj_entity[3]),
            filepath=filepath,
        )

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

        data_json.append(relation)
    return data_json


parser = argparse.ArgumentParser()
parser.add_argument("filelist", help="list for apf and sgm files")
parser.add_argument("-s", "--start", help="index of first file in filelist to process")
parser.add_argument("-e", "--end", help="index of last file in filelist to process")
parser.add_argument(
    "-o", "--output", help="filepath for the output json", required=True
)
args = parser.parse_args()


with open(args.filelist, "r") as inp:
    filelist = [filepath.strip() for filepath in inp.readlines()]
    inp.close()

i = 0
if args.start:
    i = int(args.start)
    filelist = filelist[int(args.start) :]

if args.end:
    filelist = filelist[: int(args.end)]

output_json = []
for filepath in tqdm(filelist):
    i += 1
    text, relations, named_entities = preprocess_ace(
        filepath + ".apf.xml", filepath + ".sgm"
    )
    # try to format the text in tacred format
    # skip if any step fails
    try:
        output_json += to_tacred_format(text, relations, named_entities, filepath)
    except AssertionError:
        print(f"could not transform {filepath}")
    
    if (i+1) % 3 == 0 or (i+1) == len(filelist):
        with open(args.output[:-5] + f"_{i}.json", "w") as out:
            json.dump(output_json, out)
            out.close()
            output_json = []

    # filepath = "English/bn/timex2norm/CNNHL_ENG_20030424_123502.25"
    # text, relations, named_entities = preprocess_ace(
    #     filepath + ".apf.xml", filepath + ".sgm"
    # )
    # output_json += to_tacred_format(text, relations, named_entities, filepath)

    # with open(sys.argv[3], "w") as out:
    #     json.dump(output_json, out)
    #     out.close()
