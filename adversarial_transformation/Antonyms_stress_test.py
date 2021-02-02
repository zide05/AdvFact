from nltk.corpus import wordnet
from spacy_stanza import StanzaLanguage
import json
import stanza
import os
import random
from tqdm import tqdm

os.environ["OMP_NUM_THREADS"] = "6"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "6"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "6"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "6"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "6"  # export NUMEXPR_NUM_THREADS=6


def get_antonomys(word, pos_tag):
    antonyms = []
    for syn in wordnet.synsets(word, pos=pos_tag):
        for l in syn.lemmas():
            if l.antonyms():
                antonyms.append(l.antonyms()[0].name())
    return list(set(antonyms))


def get_first_hypernym_then_hyponym(word, pos_tag):
    result = []
    for syn in wordnet.synsets(word, pos=pos_tag):
        hyperset = syn.hypernyms()
        for hypersyn in hyperset:
            if hypersyn.pos() == pos_tag:
                hyposet = hypersyn.hyponyms()
                for hyposyn in hyposet:
                    if hyposyn.pos() == pos_tag:
                        for l in hyposyn.lemmas():
                            result.append(l.name())
    return result


pos_types = ["VERB", "ADJ"]
pos_orig_wordnet_trans = {"VERB": wordnet.VERB, "ADJ": wordnet.ADJ}


def verb_adj_change(datas, save_dir, try_num=4):
    snlp = stanza.Pipeline(lang="en")
    nlp = StanzaLanguage(snlp)

    tmp_dir = os.path.join(save_dir, "neg.verb_adj_change")
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    neg_path = os.path.join(tmp_dir, "data-dev.jsonl")
    with open(neg_path, "w") as fn:
        for (id, docu, claim, _) in tqdm(datas):
            claim = claim.strip()
            the_claim = nlp(claim)

            claim_tokens = [token.text for token in the_claim]

            for idx, token in enumerate(the_claim):
                if token.pos_ in pos_types:
                    antonomys = get_antonomys(token.text, pos_orig_wordnet_trans[token.pos_])
                    if len(antonomys) > 0:
                        anto = random.choice(antonomys).replace("_", " ")
                        # when the return is phrase
                        new_claim = " ".join(claim_tokens[:idx]) + " {} ".format(anto) + " ".join(
                            claim_tokens[idx + 1:])
                        tag = "neg.{}_antonomy_{}_to_{}".format(token.pos_.lower(), token.text, anto)

                        fn.write(json.dumps(
                            {"id": id, "text": docu, "claim": new_claim, "label": "INCORRECT", "tag": tag,
                             "origin_claim": claim}) + "\n")

    print("Verb and Adj antonomy test data in {} done!".format(save_dir))
