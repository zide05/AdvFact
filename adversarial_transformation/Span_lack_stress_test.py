import spacy
import os
from benepar.spacy_plugin import BeneparComponent
import json
from tqdm import tqdm

clause_tags = ["S", "SBAR"]
phrase_tags = ["PP"]
conj_tags = ["CONJP"]

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "4"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4"  # export NUMEXPR_NUM_THREADS=6


def get_item_from_generator(the_generator, idx):
    for tmp_idx, tmp in enumerate(the_generator):
        if tmp_idx == idx:
            return tmp
    return None


def get_labels(span):
    if span is None:
        return None
    if len(span._.labels) > 0:
        return span._.labels[-1]
    return None


def pos_transform(tokens, spacy_claim, reorder):
    new_tokens = []
    for idx, child in enumerate(spacy_claim._.children):
        if len(child._.labels) > 0:
            # a tuple of labels for the given span.A span may have multiple labels when there are unary chains in the parse tree, thus choosing lastlabel
            label = child._.labels[-1]

            father_label = get_labels(spacy_claim)
            if father_label and father_label == "NP":
                sibling1 = get_item_from_generator(spacy_claim._.children, idx + 1)
                sibling2 = get_item_from_generator(spacy_claim._.children, idx + 2)
                sibling3 = get_item_from_generator(spacy_claim._.children, idx + 3)

                label2 = get_labels(sibling2)
                # delete detailed description for noun phrase
                # e.g.,Phil Rudd, the drummer for legendary hard rock band AC/DC, has pleaded guilty to charges of threatening to kill and possession of drugs in a New Zealand court
                # ----->Phil Rudd has pleaded guilty to charges of threatening to kill and possession of drugs in a New Zealand court
                if sibling1 and sibling2 and sibling3 and len(sibling1._.labels) == 0 and sibling1.text == "," and len(
                        sibling3._.labels) == 0 and sibling3.text == "," and label2 and label2 == "NP":
                    new_claim = " ".join(tokens[:child.end] + tokens[sibling3.end:]).strip(",").strip()
                    tag = "pos.delete_detailed_NP"
                    new_tokens.append((new_claim, tag))

                # delete detailed description for clause
                # e.g.,Heather Mack, 19, who gave birth to her own daughter just weeks ago, was found guilty....
                # ----->Heather Mack, 19, was found guilty....
                if sibling1 and sibling2 and sibling3 and sibling1.text == "," and sibling3.text == "," and label2 and label2 in clause_tags:
                    new_claim = " ".join(tokens[:child.end] + tokens[sibling3.end:]).strip(",").strip()
                    tag = "pos.delete_detailed_{}".format(label2)
                    new_tokens.append((new_claim, tag))

                # delete detailed description for clause
                # although the clause is not surrounded by two ,
                if (not (sibling1 and sibling2 and sibling3) or not (
                        sibling1.text == "," and sibling3.text == ",")) and label in clause_tags:

                    # e.g., farouk younis , the imam of the mosque used by relatives of hassan munshi and talha asmal , said muslims must talk to other teenagers who ` might be looking at them and thinking this is the way ' .
                    before = get_item_from_generator(spacy_claim._.children, idx - 1)
                    if before and get_labels(before) == "WHNP":
                        new_claim = " ".join(tokens[:before.start] + tokens[child.end:]).strip(",").strip()
                        tag = "pos.delete_detailed_{}".format("WHNP")
                        new_tokens.append((new_claim, tag))
                    else:
                        new_claim = " ".join(tokens[:child.start] + tokens[child.end:]).strip(",").strip()
                        tag = "pos.delete_detailed_{}".format(label)
                        new_tokens.append((new_claim, tag))

            if label in phrase_tags:
                after = get_item_from_generator(spacy_claim._.children, idx + 1)
                if after and len(after._.labels) > 0:
                    after_label = after._.labels[-1]
                    if after_label == label:
                        if reorder:
                            # swap the neighboring two PP
                            new_claim = " ".join(
                                tokens[:child.start] + [after.text, child.text] + tokens[after.end:]).strip(",").strip()
                            tag = "pos.reorder_neighboring_{}".format(label)
                            new_tokens.append((new_claim, tag))

                        # skip one PP if there exist two neighboring PP
                        new_claim = " ".join(tokens[:child.start] + tokens[after.start:]).strip(",").strip()
                        tag = "pos.lack_{}_{}".format(label, child.text)
                        new_tokens.append((new_claim, tag))

                        new_claim = " ".join(tokens[:child.end] + tokens[after.end:]).strip(",").strip()
                        tag = "pos.lack_{}_{}".format(label, child.text)
                        new_tokens.append((new_claim, tag))

            # swap the conjunction part, and delete one part of conjunction
            if label in conj_tags:
                before = get_item_from_generator(spacy_claim._.children, idx - 1)
                after = get_item_from_generator(spacy_claim._.children, idx + 1)
                if before and after:
                    if reorder:
                        # swap the position of phrase around CONJP
                        new_claim = " ".join(
                            tokens[:before.start] + [after.text, child.text, before.text] + tokens[after.end:]).strip(
                            ",").strip()
                        tag = "pos.reorder_around_{}".format(label)
                        new_tokens.append((new_claim, tag))

                    # delete one part of conjunction
                    new_claim = " ".join(
                        tokens[:before.start] + [before.text] + tokens[after.end:]).strip(",").strip()
                    tag = "pos.delete_after_{}".format(label)
                    new_tokens.append((new_claim, tag))
                    new_claim = " ".join(
                        tokens[:before.start] + [after.text] + tokens[after.end:]).strip(",").strip()
                    tag = "pos.delete_before_{}".format(label)
                    new_tokens.append((new_claim, tag))

        if len(child) == 1:
            # token level pos tagging for CCONJ word e.g., and, which will not be included in CONJP
            token = child[0]
            if token.pos_ == "CCONJ":
                before = get_item_from_generator(spacy_claim._.children, idx - 1)
                after = get_item_from_generator(spacy_claim._.children, idx + 1)
                if before and after:
                    if reorder:
                        # swap the position of phrase around CCONJ
                        new_claim = " ".join(
                            tokens[:before.start] + [after.text, child.text, before.text] + tokens[after.end:]).strip(
                            ",").strip()
                        tag = "pos.reorder_around_{}".format("CCONJ")
                        new_tokens.append((new_claim, tag))

        new_tokens.extend(pos_transform(tokens, child, reorder))
    return new_tokens


def span_change(datas, save_dir, reorder=True):
    nlp = spacy.load('en_core_web_sm')
#     nlp.add_pipe(BeneparComponent('benepar_en'))
    nlp.add_pipe('benepar', config={'model': 'benepar_en3'})

    if reorder:
        tmp_dir = os.path.join(save_dir, "pos.span_lack_and_reorder")
    else:
        tmp_dir = os.path.join(save_dir, "pos.span_lack")
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    pos_path = os.path.join(tmp_dir, "data-dev.jsonl")
    claim_set = []
    with open(pos_path, "w") as fp:
        for (id, docu, claim, _) in tqdm(datas):
            claim = claim.strip()
            the_claim = nlp(claim)
            claim_tokens = [token.text for token in the_claim]
            # here claim must be single sentence
            the_claim = list(the_claim.sents)[0]
            for new_claim, tag in pos_transform(claim_tokens, the_claim, reorder):
                # reduce redundency
                if new_claim not in claim_set:
                    claim_set.append(new_claim)
                    fp.write(
                        json.dumps(
                            {"id": id, "text": docu, "claim": new_claim, "label": "CORRECT", "tag": tag,
                             "origin_claim": claim}) + "\n")

    if reorder:
        print("Span lack and reorder test data in {} done!".format(save_dir))
    else:
        print("Span lack test data in {} done!".format(save_dir))
