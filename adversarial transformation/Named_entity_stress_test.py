import random
import os
from nltk import sent_tokenize
import json
import stanza
from spacy_stanza import StanzaLanguage
from tqdm import tqdm

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "4"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4"  # export NUMEXPR_NUM_THREADS=6

entity_types_map = {"PERSON": "person", "NORP": "group", "ORG": "group", "FAC": "location", "GPE": "location",
                    "LOC": "location", "PRODUCT": "product", "EVENT": "event"}
all_entity_types = list(set([value for key, value in entity_types_map.items()]))
type_entity_dict = {key: [] for key in all_entity_types}
for key, value in entity_types_map.items():
    type_entity_dict[value].append(key)


def neg_transform(tokens, entity_span, entity_dict, add_num=1):
    new_tokens = []
    label = entity_span.label_
    start = entity_span.start
    end = entity_span.end

    entity_type = entity_types_map[label]

    # neg_random_choose_replacement from entity dict
    try_num = 2 * len(entity_dict[entity_type])
    for i in range(add_num):
        new_ent = None
        for i in range(try_num):
            tmp_ent = random.choice(entity_dict[entity_type])
            entity1 = tmp_ent.lower().strip()
            entity2 = entity_span.text.lower().strip()
            if entity1 != entity2 and not (entity1 in entity2) and not (entity2 in entity1):
                new_ent = tmp_ent
                break
        if new_ent is not None:
            new_tokens.append(
                (" ".join(tokens[:start]) + " " + new_ent + " " + " ".join(tokens[end:]), "neg.entity_random_choose"))

            if len(entity_span.text.split(" ")) == 1:
                new_tokens.append(
                    (" ".join(tokens[:start]) + " " + random.choice(
                        new_ent.split(" ")) + " " + entity_span.text + " " + " ".join(tokens[end:]),
                     "neg.entity_part_add"))
            elif len(entity_span.text.split(" ")) > 1:
                this_entity_ = entity_span.text.split(" ")
                delete_idx = random.choice([i for i in range(len(this_entity_))])
                this_entity_[delete_idx] = random.choice(new_ent.split(" "))
                if " ".join(this_entity_).strip() != entity_span.text:
                    new_tokens.append(
                        (" ".join(tokens[:start]) + " " + " ".join(this_entity_) + " " + " ".join(tokens[end:]),
                         "neg.entity_part_replace"))
    return new_tokens


def pos_transform(tokens, entity_span):
    label = entity_span.label_
    start = entity_span.start
    end = entity_span.end

    if label != "PERSON":
        return []

    new_tokens = []
    entity_tokens = entity_span.text.split(" ")
    if len(entity_tokens) <= 1:
        return []

    for _token in entity_tokens:
        new_tokens.append(
            (" ".join(tokens[:start]) + " " + _token + " " + " ".join(tokens[end:]), "pos.entity_subname"))

    return new_tokens


def entity_change(datas, save_dir):
    snlp = stanza.Pipeline(lang="en")
    nlp = StanzaLanguage(snlp)

    tmp_dir = os.path.join(save_dir, "pos.person_entity")
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    pos_entity_path = os.path.join(tmp_dir, "data-dev.jsonl")
    tmp_dir = os.path.join(save_dir, "neg.person_entity")
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    neg_entity_path = os.path.join(tmp_dir, "data-dev.jsonl")
    with open(pos_entity_path, "w") as fp, open(neg_entity_path, "w") as fn:
        for (id, docu, claim, _) in tqdm(datas):
            docu_sents = sent_tokenize(docu)
            doc_entitys = {item: [] for item in all_entity_types}
            for doc in nlp.pipe(docu_sents):
                for ent in doc.ents:
                    if ent.label_ in entity_types_map.keys():
                        doc_entitys[entity_types_map[ent.label_]].append(ent.text)
            for type_ in all_entity_types:
                doc_entitys[type_] = list(set(doc_entitys[type_]))

            claim = claim.strip()
            the_claim = nlp(claim)

            claim_tokens = [token.text for token in the_claim]
            for ent in the_claim.ents:
                if ent.label_ in entity_types_map.keys():
                    for new_claim, tag in neg_transform(claim_tokens, ent, doc_entitys):
                        fn.write(
                            json.dumps(
                                {"id": id, "text": docu, "claim": new_claim, "label": "INCORRECT", "tag": tag,
                                 "origin_claim": claim}) + "\n")
                    for new_claim, tag in pos_transform(claim_tokens, ent):
                        fp.write(
                            json.dumps(
                                {"id": id, "text": docu, "claim": new_claim, "label": "CORRECT", "tag": tag,
                                 "origin_claim": claim}) + "\n")

    print("Entity swap test data in {} done!".format(save_dir))
