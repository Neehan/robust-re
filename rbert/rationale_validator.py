import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AdamW, BertForTokenClassification, BertTokenizerFast

torch.manual_seed(0)
np.random.seed(0)


class Dataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels, first_token_indices):
        self.encodings = encodings  # original tokens converted to bert tokens
        self.labels = labels  # original rationale converted for bert tokens
        self.first_token_indices = (
            first_token_indices  # index of the first token in the original tokenization
        )

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        item["bert_first_token_indices"] = torch.tensor(self.first_token_indices[idx])
        return item

    def __len__(self):
        return len(self.labels)


def encode_tags(rationale, encodings, unk_rationale_id):
    """
    encode token rationale to bert token rationale by setting
    the label of the first subtoken equal to the original
    rationale id, and rest equal to unk_rationale_id
    """
    encoded_labels = []
    bert_first_token_indices = []
    for doc_labels, doc_offset in zip(rationale, encodings.offset_mapping):
        # create an empty array of args.unk_rationale_id
        doc_enc_labels = np.ones(len(doc_offset), dtype=int) * unk_rationale_id
        arr_offset = np.array(doc_offset)

        # set labels whose first offset position is 0 and the second is not 0
        first_token_indices = (arr_offset[:, 0] == 0) & (arr_offset[:, 1] != 0)
        assert sum(first_token_indices) == len(
            doc_labels
        ), f"fst tkn {sum(first_token_indices)}, doc_labels {len(doc_labels)}"
        doc_enc_labels[first_token_indices] = doc_labels
        encoded_labels.append(doc_enc_labels.tolist())
        bert_first_token_indices.append(first_token_indices)

    return encoded_labels, bert_first_token_indices


def preprocess_data(args, data_json, entity_symbol=False):
    """
    convert text into torch dataloader object
    """
    texts, rationale = [], []
    for x in tqdm(data_json):
        subj_start, subj_end = int(x["subj_start"]), int(x["subj_end"])
        obj_start, obj_end = int(x["obj_start"]), int(x["obj_end"])
        x["rationale"] = x.get("rationale", [False] * len(x["token"]))

        if entity_symbol:
            # add a $ before and after the subj
            x, subj_start, subj_end, obj_start, obj_end = insert_symbol(
                x, "$", subj_start, subj_end, obj_start, obj_end
            )
            # add a # before and after the obj
            x, obj_start, obj_end, subj_start, subj_end = insert_symbol(
                x, "#", obj_start, obj_end, subj_start, subj_end
            )

        x["subj_start"] = subj_start
        x["subj_end"] = subj_end
        x["obj_start"] = obj_start
        x["obj_end"] = obj_end

        texts.append(x["token"])
        rationale.append(x["rationale"])

    # now convert to bert tokens
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    encodings = tokenizer(
        texts,
        is_split_into_words=True,
        return_offsets_mapping=True,
        padding=True,
        truncation=True,
    )

    # transform original token rationale to bert token rationale
    # this is because in a token tagging task, we need one label for each token
    # bert tokenization splits up many tokens, so we set first subtoken
    # to have same rationale label as original token, the rest of the subtokens
    # have a distinct value, which we just ignore
    labels, first_token_indices = encode_tags(
        rationale, encodings, args.unk_rationale_id
    )

    encodings.pop("offset_mapping")

    dataset = Dataset(encodings, labels, first_token_indices)
    return dataset


def bert_label_to_rationale(labels, bert_first_token_indices):
    """
    bert predicts labels for each bert subtoken,
    but we only need prediction for each subtoken with
    offset 0 for rationale of the original tokens
    """
    rationale = []
    for i in range(len(labels)):
        rationale.append(labels[i, bert_first_token_indices[i]].tolist())
    return rationale


def view_rationale(tokens, rationale):
    assert len(tokens) == len(
        rationale
    ), f"tokens: {len(tokens)}, rationale:{len(rationale)}"
    pretty_text = ""
    for i, token in enumerate(tokens):
        if rationale[i] == 1:
            pretty_text += "[" + token + "]" + " "
        else:
            pretty_text += token + " "
    return pretty_text


def print_human_vs_model(test_json, pred_rationale):
    assert len(test_json) == len(
        pred_rationale
    ), "test rationale size and test dataset size are different"
    print_indices = np.random.choice(len(test_json), 7)
    for i in print_indices:
        text = test_json[i]["token"]
        relation = test_json[i]["relation"]
        test_rationale = test_json[i].get("rationale", [False] * len(test_json[i]))
        print("Relation: ", relation)
        print("Human: ", view_rationale(text, test_rationale))
        print("Model: ", view_rationale(text, pred_rationale[i]))


def evaluate(args, model, test_loader):
    print("evaluate")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    model.eval()
    list_of_classes = [0, 1, args.unk_rationale_id]
    correct = dict(zip(list_of_classes, [0, 0, 0]))
    total = dict(zip(list_of_classes, [0, 0, 0]))
    pred_rationale = []
    for batch in tqdm(test_loader):
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        bert_first_token_indices = batch["bert_first_token_indices"].to(device)
        with torch.no_grad():
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            preds = torch.argmax(outputs[1], dim=2)
            pred_rationale += bert_label_to_rationale(preds, bert_first_token_indices)

            for c in list_of_classes:
                correct[c] += ((preds == labels) * (labels == c)).sum().item()
                total[c] += (labels == c).sum().item()
    return total, correct, pred_rationale


def print_score(args, total, correct):
    list_of_classes = [0, 1, args.unk_rationale_id]
    print("class id\t# correct\t# total\tpcnt")
    for c in list_of_classes:
        print(
            f"{c}\t{correct[c]}\t{max(total[c], 1)}\t{correct[c]/max(total[c], 1)*100}"
        )


def insert_symbol(x, symbol, start, end, other_start, other_end):
    """
    insert a special symbol before and after an entity token
    x is an example sentence
    start and end are the positions of the target entity in the sentence
    other start and end are the positions of the other entity
    for instance subj and obj

    other entity must be positioned later in the sentence
    """

    entity = " ".join(x["token"][start : end + 1])
    other_entity = " ".join(x["token"][other_start : other_end + 1])

    x["token"] = (
        x["token"][:start]
        + [symbol]
        + x["token"][start : end + 1]
        + [symbol]
        + x["token"][end + 1 :]
    )
    x["rationale"] = (
        x["rationale"][:start]
        + [True]
        + x["rationale"][start : end + 1]
        + [True]
        + x["rationale"][end + 1 :]
    )

    if other_start > start:
        other_start += 2
        other_end += 2
    elif other_start == start:
        other_start += 1
        other_end += 1
    start += 1
    end += 1

    new_entity = " ".join(x["token"][start : end + 1])
    new_other_entity = " ".join(x["token"][other_start : other_end + 1])

    assert entity == new_entity, f"{entity}==={new_entity}, {x['token']}"
    assert (
        other_entity == new_other_entity
    ), f"{other_entity}==={new_other_entity}, {x['token']}"

    return x, start, end, other_start, other_end


def delete_symbol(x, symbol, start, end, other_start, other_end):
    """
    delete a special symbol before and after an entity token
    x is an example sentence
    start and end are the positions of the target entity in the sentence
    other start and end are the positions of the other entity
    for instance subj and obj

    other entity must be positioned later in the sentence
    """

    entity = " ".join(x["token"][start : end + 1])
    other_entity = " ".join(x["token"][other_start : other_end + 1])

    x["token"] = (
        x["token"][: start - 1] + x["token"][start : end + 1] + x["token"][end + 2 :]
    )
    x["rationale"] = (
        x["rationale"][: start - 1]
        + x["rationale"][start : end + 1]
        + x["rationale"][end + 2 :]
    )

    if other_start > start:
        other_start -= 2
        other_end -= 2
    elif other_start == start:
        other_start -= 1
        other_end -= 1
    start -= 1
    end -= 1

    new_entity = " ".join(x["token"][start : end + 1])
    new_other_entity = " ".join(x["token"][other_start : other_end + 1])

    assert entity == new_entity, f"{entity}==={new_entity}, {x['token']}"
    assert (
        other_entity == new_other_entity
    ), f"{other_entity}==={new_other_entity}, {x['token']}"

    return x, start, end, other_start, other_end


def load_datafile(filepath):
    ""

    with open(filepath, "r") as inp:
        data_json = json.load(inp)
        inp.close()
    return data_json


def train_test_split(p, data_json):
    train_json, test_json = [], []
    for x in data_json:
        # flip a coin with 0.8 success
        if np.random.binomial(1, p):  # train data
            train_json.append(x)
        else:  # test data
            test_json.append(x)
    return train_json, test_json


def train(args, train_loader, test_loader, test_json):
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print("loading bert.")
    model = BertForTokenClassification.from_pretrained("bert-base-cased", num_labels=3)
    model.to(device)
    optim = AdamW(model.parameters(), lr=args.lr)
    print("loaded. staring training.")

    # best_rationale_acc = 0
    for epoch in range(args.num_epoch):
        for batch in tqdm(train_loader):
            optim.zero_grad()
            model.train()
            model.zero_grad()
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)
            outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
            loss = outputs[0]
            loss.backward()
            optim.step()

        if epoch % 3 == 2 and not args.no_logs:
            total, correct, pred_rationale = evaluate(args, model, test_loader)
            print_score(args, total, correct)
            print_human_vs_model(test_json, pred_rationale)
    return model


def save_dataset(data_json, filepath, pred_rationale):
    # remove special symbols
    for i, x in enumerate(data_json):
        subj_start, subj_end = int(x["subj_start"]), int(x["subj_end"])
        obj_start, obj_end = int(x["obj_start"]), int(x["obj_end"])
        x["rationale"] = pred_rationale[i]

        # remove the symbols that were added for token classification
        # remove $ before and after the subj
        # x, subj_start, subj_end, obj_start, obj_end = delete_symbol(
        #     x, "$", subj_start, subj_end, obj_start, obj_end
        # )
        # # remove # before and after the obj
        # x, obj_start, obj_end, subj_start, subj_end = delete_symbol(
        #     x, "#", obj_start, obj_end, subj_start, subj_end
        # )

        x["subj_start"] = subj_start
        x["subj_end"] = subj_end
        x["obj_start"] = obj_start
        x["obj_end"] = obj_end

        assert len(x["token"]) == len(x["stanford_pos"])
        assert len(x["rationale"]) == len(x["stanford_pos"])

    with open(filepath, "w") as outp:
        json.dump(data_json, outp)
        outp.close()


def main():
    parser = argparse.ArgumentParser(description="rationale tokenizer")
    parser.add_argument(
        "--data_dir", help="path to data directory", default="./data/ace2005/final/"
    )
    parser.add_argument(
        "--ratdataset", help="rationale file name", default="rationale.json"
    )
    parser.add_argument("--testset", help="test file name", default="train.json")
    parser.add_argument("--train_pct", help="train test split", default=0.8, type=float)
    parser.add_argument(
        "--unk_rationale_id",
        help="rationale id to ignore a token",
        default=-100,
        type=int,
    )
    parser.add_argument("--batch_size", help="batch size", default=5, type=int)
    parser.add_argument("--num_epoch", help="number of epochs", default=5, type=int)
    parser.add_argument("--lr", help="learning rate", default=1e-5, type=float)
    parser.add_argument("--do_train", help="train a new model", action="store_true")
    parser.add_argument(
        "--no_logs", help="no intermediate logs (saves time)", action="store_true"
    )

    args = parser.parse_args()
    rationale_json = load_datafile(args.data_dir + args.ratdataset)
    test_json = load_datafile(args.data_dir + args.testset)

    if args.do_train:
        train_json, test_json = train_test_split(args.train_pct, rationale_json)
    else:
        train_json = rationale_json

    print("preprocessing data.")
    # rationale dataset do not have $ or # so add them
    train_loader = DataLoader(
        preprocess_data(args, train_json, entity_symbol=True), batch_size=args.batch_size, shuffle=False
    )
    # test set has $ # unless it was created from rationale.json
    test_loader = DataLoader(
        preprocess_data(args, test_json, entity_symbol=args.do_train), batch_size=args.batch_size, shuffle=False
    )

    print("training model.")
    model = train(args, train_loader, test_loader, test_json)

    total, correct, pred_rationale = evaluate(args, model, test_loader)
    print("\nTraining complete..\n")
    print_score(args, total, correct)
    print_human_vs_model(test_json, pred_rationale)

    # if generating rationale, save them.
    if not args.do_train:
        save_dataset(test_json, args.data_dir + "rationale_" + args.testset, pred_rationale)


if __name__ == "__main__":
    main()
