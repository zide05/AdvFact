import stanza
from spacy_stanza import StanzaLanguage
import json

import os
from nltk import sent_tokenize
import math
import random
from tqdm import tqdm

os.environ["OMP_NUM_THREADS"] = "4"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "4"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "4"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "4"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "4"  # export NUMEXPR_NUM_THREADS=6


def read_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            data.append(json.loads(line))
    return data


numerical_types = ["DATE", "TIME", "PERCENT", "MONEY", "QUANTITY", "ORDINAL", "CARDINAL"]


from sutime import SUTime
import datetime
from datetime import date

month_map = {"1": "January", "2": "February", "3": "March", "4": "April", "5": "May", "6": "June", "7": "July",
             "8": "August", "9": "September", "10": "October", "11": "November", "12": "December"}
reference_date = "5050-12-31"
reference_year = 5050 - 10
reference_month = 12
reference_day = 31

number_for_random_choice = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]


def pos_time_transform(origin_text, time_result):
    last_results = []
    for item in time_result:
        value = item["value"]
        text = item["text"]
        start = item["start"]
        end = item["end"]


        year, month, day, hour, minute, second = None, None, None, None, None, None
        if "last" in text.lower() or "next" in text.lower() or "today" in text or "tomorrow" in text or "yesterday" in text or "this" in text:
            # skipping relative date
            continue
        if item["type"] == 'DATE':
            if "W" in value:
                # skipping time about week, e.g, last week
                continue
            if len(value.split("-")) == 1:
                try:
                    year = int(value)
                except ValueError:
                    print("Value error: {}", format(value))
            if len(value.split("-")) == 2:
                try:
                    year = int(value.split("-")[0])
                    month = int(value.split("-")[1].lstrip("0"))
                except ValueError:
                    print("Value error: {}", format(value))
            if len(value.split("-")) == 3:
                try:
                    year = int(value.split("-")[0])
                    month = int(value.split("-")[1].lstrip("0"))
                    day = int(value.split("-")[2].lstrip("0"))
                except ValueError:
                    print("Value error: {}", format(value))
            if not (year is not None and year < reference_year) and month == reference_month:
                continue

        # only year
        if year is not None and month is None and day is None and year < reference_year:
            if origin_text.lower()[:start].endswith("in "):
                # add before
                tmp_year_sep = random.choice(number_for_random_choice)
                new_year = year + tmp_year_sep
                last_results.append((origin_text[:start - 3] + "before {} ".format(new_year) + origin_text[end:],
                                     "pos.in {} to before {}".format(year, new_year)))
                last_results.append(
                    (origin_text[:start - 3] + "{} years before {} ".format(tmp_year_sep, new_year) + origin_text[end:],
                     "pos.in {} to {} years before {}".format(year, tmp_year_sep, new_year)))

                # add after
                tmp_year_sep = random.choice(number_for_random_choice)
                new_year = year - tmp_year_sep
                last_results.append((origin_text[:start - 3] + "after {} ".format(new_year) + origin_text[end:],
                                     "pos.in {} to after {}".format(year, new_year)))
                last_results.append(
                    (origin_text[:start - 3] + "{} years after {} ".format(tmp_year_sep, new_year) + origin_text[end:],
                     "pos.in {} to {} years after {}".format(year, tmp_year_sep, new_year)))

            if origin_text.lower()[:start].endswith("before "):
                # add before
                tmp_year_sep = random.choice(number_for_random_choice)
                new_year = year + tmp_year_sep
                last_results.append((origin_text[:start - 7] + "before {} ".format(new_year) + origin_text[end:],
                                     "pos.before {} to before {}".format(year, new_year)))
            if origin_text.lower()[:start].endswith("after "):
                # add after
                tmp_year_sep = random.choice(number_for_random_choice)
                new_year = year - tmp_year_sep
                last_results.append((origin_text[:start - 6] + "after {} ".format(new_year) + origin_text[end:],
                                     "pos.after {} to after {}".format(year, new_year)))

        # year and month
        if year is not None and year < reference_year and month is not None and day is None:
            if origin_text.lower()[:start].endswith("in "):
                # add before
                tmp_month_sep = random.choice(number_for_random_choice)
                tmp_month = month + tmp_month_sep
                if tmp_month / 12 > 1:
                    new_year = year + 1
                    new_month = tmp_month % 12
                else:
                    new_year = year
                    new_month = tmp_month
                last_results.append((origin_text[:start - 3] + "before {}, {} ".format(month_map[str(new_month)],
                                                                                       new_year) + origin_text[end:],
                                     "pos.in {} to before {}, {}".format(text, month_map[str(new_month)], new_year)))
                last_results.append((origin_text[:start - 3] + "{} months before {}, {} ".format(tmp_month_sep,
                                                                                                 month_map[
                                                                                                     str(new_month)],
                                                                                                 new_year) + origin_text[
                                                                                                             end:],
                                     "pos.in {} to {} months before {}, {}".format(text, tmp_month_sep,
                                                                                   month_map[str(new_month)],
                                                                                   new_year)))

                # add after
                tmp_month_sep = random.choice(number_for_random_choice)
                tmp_month = month - tmp_month_sep
                if tmp_month <= 0:
                    new_year = year - 1
                    new_month = 12 + tmp_month
                else:
                    new_year = year
                    new_month = tmp_month
                last_results.append((origin_text[:start - 3] + "after {}, {} ".format(month_map[str(new_month)],
                                                                                      new_year) + origin_text[end:],
                                     "pos.in {} to after {}, {}".format(text, month_map[str(new_month)], new_year)))
                last_results.append((origin_text[:start - 3] + "{} months after {}, {} ".format(tmp_month_sep,
                                                                                                month_map[
                                                                                                    str(new_month)],
                                                                                                new_year) + origin_text[
                                                                                                            end:],
                                     "pos.in {} to {} months after {}, {}".format(text, tmp_month_sep,
                                                                                  month_map[str(new_month)],
                                                                                  new_year)))
            if origin_text.lower()[:start].endswith("before "):
                # add before
                tmp_month_sep = random.choice(number_for_random_choice)
                tmp_month = month + tmp_month_sep
                if tmp_month / 12 > 1:
                    new_year = year + 1
                    new_month = tmp_month % 12
                else:
                    new_year = year
                    new_month = tmp_month
                last_results.append((origin_text[:start - 7] + "before {}, {} ".format(month_map[str(new_month)],
                                                                                       new_year) + origin_text[end:],
                                     "pos.before {} to before {}, {}".format(text, month_map[str(new_month)],
                                                                             new_year)))

            if origin_text.lower()[:start].endswith("after "):
                # add after
                tmp_month_sep = random.choice(number_for_random_choice)
                tmp_month = month - tmp_month_sep
                if tmp_month <= 0:
                    new_year = year - 1
                    new_month = 12 + tmp_month
                else:
                    new_year = year
                    new_month = tmp_month
                last_results.append((origin_text[:start - 6] + "after {}, {} ".format(month_map[str(new_month)],
                                                                                      new_year) + origin_text[end:],
                                     "pos.after {} to after {}, {}".format(text, month_map[str(new_month)], new_year)))

        # only month
        if not (year is not None and year < reference_year) and month is not None and day is None:
            if month == 12 or month == 1:
                continue
            if origin_text.lower()[:start].endswith("in "):
                # add before
                tmp_month_sep = random.choice([i for i in range(1, 13 - month)])
                new_month = month + tmp_month_sep
                last_results.append(
                    (origin_text[:start - 3] + "before {} ".format(month_map[str(new_month)]) + origin_text[end:],
                     "pos.in {} to before {}".format(text, month_map[str(new_month)])))
                last_results.append((origin_text[:start - 3] + "{} months before {} ".format(tmp_month_sep, month_map[
                    str(new_month)]) + origin_text[end:],
                                     "pos.in {} to {} months before {}".format(text, tmp_month_sep,
                                                                               month_map[str(new_month)])))

                # add after
                tmp_month_sep = random.choice([i for i in range(1, month)])
                new_month = month - tmp_month_sep
                last_results.append(
                    (origin_text[:start - 3] + "after {} ".format(month_map[str(new_month)]) + origin_text[end:],
                     "pos.in {} to after {}".format(text, month_map[str(new_month)])))
                last_results.append((origin_text[:start - 3] + "{} months after {} ".format(tmp_month_sep, month_map[
                    str(new_month)]) + origin_text[end:],
                                     "pos.in {} to {} months after {}".format(text, tmp_month_sep,
                                                                              month_map[str(new_month)])))
            if origin_text.lower()[:start].endswith("before "):
                # add before
                tmp_month_sep = random.choice([i for i in range(1, 13 - month)])
                new_month = month + tmp_month_sep
                last_results.append(
                    (origin_text[:start - 7] + "before {} ".format(month_map[str(new_month)]) + origin_text[end:],
                     "pos.before {} to before {}".format(text, month_map[str(new_month)])))

            if origin_text.lower()[:start].endswith("after "):
                # add after
                tmp_month_sep = random.choice([i for i in range(1, month)])
                new_month = month - tmp_month_sep
                last_results.append(
                    (origin_text[:start - 6] + "after {} ".format(month_map[str(new_month)]) + origin_text[end:],
                     "pos.after {} to after {}".format(text, month_map[str(new_month)])))

        # year month and day
        if year is not None and year < reference_year and month is not None and day is not None:
            if origin_text.lower()[:start].endswith("on "):
                # add before
                tmp_date = date(year, month, day)
                tmp_day_sep = random.choice([i for i in range(1, 32)])
                new_date = tmp_date + datetime.timedelta(days=tmp_day_sep)

                last_results.append(
                    (origin_text[:start - 3] + "before {} {}, {} ".format(month_map[str(new_date.month)], new_date.day,
                                                                          new_date.year) + origin_text[end:],
                     "pos.in {} to before {} {}, {}".format(text, month_map[str(new_date.month)], new_date.day,
                                                            new_date.year)))
                last_results.append((origin_text[:start - 3] + "{} days before {} {}, {} ".format(tmp_day_sep,
                                                                                                  month_map[
                                                                                                      str(
                                                                                                          new_date.month)],
                                                                                                  new_date.day,
                                                                                                  new_date.year) + origin_text[
                                                                                                                   end:],
                                     "pos.in {} to {} days before {} {}, {}".format(text, tmp_day_sep,
                                                                                    month_map[str(new_date.month)],
                                                                                    new_date.day,
                                                                                    new_date.year)))
                # add after
                tmp_date = date(year, month, day)
                tmp_day_sep = random.choice([i for i in range(1, 32)])
                new_date = tmp_date + datetime.timedelta(days=-tmp_day_sep)
                last_results.append(
                    (origin_text[:start - 3] + "after {} {}, {} ".format(month_map[str(new_date.month)], new_date.day,
                                                                         new_date.year) + origin_text[end:],
                     "pos.in {} to after {} {}, {}".format(text, month_map[str(new_date.month)], new_date.day,
                                                           new_date.year)))
                last_results.append((origin_text[:start - 3] + "{} days after {} {}, {} ".format(tmp_day_sep,
                                                                                                 month_map[
                                                                                                     str(
                                                                                                         new_date.month)],
                                                                                                 new_date.day,
                                                                                                 new_date.year) + origin_text[
                                                                                                                  end:],
                                     "pos.in {} to {} days after {} {}, {}".format(text, tmp_day_sep,
                                                                                   month_map[str(new_date.month)],
                                                                                   new_date.day,
                                                                                   new_date.year)))
            if origin_text.lower()[:start].endswith("before "):
                # add before
                tmp_date = date(year, month, day)
                tmp_day_sep = random.choice([i for i in range(1, 32)])
                new_date = tmp_date + datetime.timedelta(days=tmp_day_sep)

                last_results.append(
                    (origin_text[:start - 7] + "before {} {}, {} ".format(month_map[str(new_date.month)], new_date.day,
                                                                          new_date.year) + origin_text[end:],
                     "pos.before {} to before {} {}, {}".format(text, month_map[str(new_date.month)], new_date.day,
                                                                new_date.year)))

            if origin_text.lower()[:start].endswith("after "):
                # add after
                tmp_date = date(year, month, day)
                tmp_day_sep = random.choice([i for i in range(1, 32)])
                new_date = tmp_date + datetime.timedelta(days=-tmp_day_sep)
                last_results.append(
                    (origin_text[:start - 6] + "after {} {}, {} ".format(month_map[str(new_date.month)], new_date.day,
                                                                         new_date.year) + origin_text[end:],
                     "pos.after {} to after {} {}, {}".format(text, month_map[str(new_date.month)], new_date.day,
                                                              new_date.year)))

    return last_results


def neg_transform(tokens, entity_span, entity_dict):
    new_tokens = []
    label = entity_span.label_
    start = entity_span.start
    end = entity_span.end

    # neg_random_choose_replacement from entity dict
    try_num = len(entity_dict[label]) * 2
    new_ent = None
    for i in range(try_num):
        tmp_ent = random.choice(entity_dict[label])
        if tmp_ent.strip().lower() != entity_span.text.strip().lower():
            new_ent = tmp_ent
            break
    if new_ent is not None:
        new_tokens.append(
            ((" ".join(tokens[:start]) + " " + new_ent + " " + " ".join(tokens[end:])).strip(),
             "neg.numerical_random_choose"))

    # add more_than, less than and so_on
    if label in ["DATE", "TIME"]:
        add_tokens = []
        if start - 1 >= 0:
            if tokens[start - 1].lower() == "in" or tokens[start - 1].lower() == "on" or tokens[
                start - 1].lower() == "at":
                add_tokens = ["before", "after"]
            elif tokens[start - 1].lower() == "before" or tokens[start - 1].lower() == "by" or tokens[
                start - 1].lower() == "till" or tokens[start - 1].lower() == "until":
                add_tokens = ["after", "since", "from"]
            elif tokens[start - 1].lower() == "after" or tokens[start - 1].lower() == "since" or tokens[
                start - 1].lower() == "from":
                add_tokens = ["before", "by", "till", "until"]

        for add_token in add_tokens:
            new_tokens.append(
                ((" ".join(tokens[:start - 1]) + " " + add_token + " " + " ".join(tokens[start:])).strip(),
                 "neg.numerical_add_" + add_token))

    if label in ["PERCENT", "MONEY", "QUANTITY"]:
        add_tokens = []
        tmp_end = None

        # if the tokens before the number is around, less than and so on, then skip the result of transformation
        span_text = " ".join(tokens[start:end]).lower().strip()
        if span_text.startswith("around") or span_text.startswith(
                "nearly") or span_text.startswith("about") or span_text.startswith(
            "less than") or span_text.startswith("more than"):
            return new_tokens

        if start - 1 >= 0:
            if tokens[start - 1] == "$":
                add_tokens = ["less than", "more than"]
                for add_token in add_tokens:
                    new_tokens.append(
                        ((" ".join(tokens[:start - 1]) + " " + add_token + " " + " ".join(tokens[start - 1:])).strip(),
                         "neg.numerical_add_" + add_token))
                return new_tokens

            if start - 2 >= 0:
                if " ".join(tokens[start - 2:]).lower() == "more than":
                    add_tokens = ["less than"]
                    tmp_end = start - 2
                elif " ".join(tokens[start - 2:]).lower() == "less than":
                    add_tokens = ["more than"]
                    tmp_end = start - 2

        if tmp_end is None:
            add_tokens = ["less than", "more than"]
            tmp_end = start

        for add_token in add_tokens:
            new_tokens.append(
                ((" ".join(tokens[:tmp_end]) + " " + add_token + " " + " ".join(tokens[start:])).strip(),
                 "neg.numerical_add_" + add_token))
    return new_tokens


def is_integer_number(s):
    try:
        tmp = int(s)
        if tmp - float(s) == 0:
            return True, int(s)
        else:
            return False, 0
    except ValueError:
        pass
    return False, 0


def pos_transform(tokens, entity_span, add_upper_bound=10):
    new_tokens = []
    label = entity_span.label_
    start = entity_span.start
    end = entity_span.end

    # if the tokens before the number is around, less than and so on, then skip the result of transformation
    span_text = " ".join(tokens[start:end]).lower().strip()
    if span_text.startswith("around") or span_text.startswith(
            "nearly") or span_text.startswith("about") or span_text.startswith(
        "less than") or span_text.startswith("more than"):
        return new_tokens

    if label in ["PERCENT", "MONEY", "QUANTITY"]:
        span_tokens = [str(token) for token in entity_span]
        for idx, token in enumerate(entity_span):
            if token.pos_ == "NUM":
                tag, number = is_integer_number(token.text)
                if tag and number > 1:
                    # add more than
                    sep_number = random.choice([i for i in range(1, math.floor(number) + 1)])
                    new_number = number - sep_number
                    new_tokens.append(
                        (" ".join(tokens[:start]) + " more than {} ".format(
                            " ".join(span_tokens[:idx] + [str(new_number)] + span_tokens[idx + 1:])) + " ".join(
                            tokens[end:]),
                         "pos.{} to more than {}".format(number, new_number)))

                    # add less than
                    sep_number = random.choice([i for i in range(1, add_upper_bound)])
                    new_number = number + sep_number
                    new_tokens.append(
                        (" ".join(tokens[:start]) + " less than {} ".format(
                            " ".join(span_tokens[:idx] + [str(new_number)] + span_tokens[idx + 1:])) + " ".join(
                            tokens[end:]),
                         "pos.{} to less than {}".format(number, new_number)))
    return new_tokens


def numerical_inference_change(datas, save_dir):
    snlp = stanza.Pipeline(lang="en")
    nlp = StanzaLanguage(snlp)
    sutime = SUTime(mark_time_ranges=False, include_range=False)

    tmp_dir = os.path.join(save_dir, "pos.numerical")
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    pos_numerical_path = os.path.join(tmp_dir, "data-dev.jsonl")
    tmp_dir = os.path.join(save_dir, "neg.numerical")
    if not os.path.exists(tmp_dir):
        os.mkdir(tmp_dir)
    neg_numerical_path = os.path.join(tmp_dir, "data-dev.jsonl")

    with open(pos_numerical_path, "w") as fp, open(neg_numerical_path, "w") as fn:
        for (id, docu, claim, _) in tqdm(datas):
            docu_sents = sent_tokenize(docu)
            doc_entitys = {item: [] for item in numerical_types}
            for doc in nlp.pipe(docu_sents):
                for ent in doc.ents:
                    if ent.label_ in numerical_types:
                        doc_entitys[ent.label_].append(ent.text)
            for type_ in numerical_types:
                doc_entitys[type_] = list(set(doc_entitys[type_]))

            claim = claim.strip()
            the_claim = nlp(claim)
            time_result = sutime.parse(claim, reference_date)

            claim_tokens = [token.text for token in the_claim]
            for new_claim, tag in pos_time_transform(claim, time_result):
                fp.write(
                    json.dumps(
                        {"id": id, "text": docu, "claim": new_claim, "label": "CORRECT", "tag": tag,
                         "origin_claim": claim}) + "\n")

            for ent in the_claim.ents:
                if ent.label_ in numerical_types:
                    for new_claim, tag in pos_transform(claim_tokens, ent):
                        fp.write(
                            json.dumps(
                                {"id": id, "text": docu, "claim": new_claim, "label": "CORRECT", "tag": tag,
                                 "origin_claim": claim}) + "\n")
                    for new_claim, tag in neg_transform(claim_tokens, ent, doc_entitys):
                        fn.write(
                            json.dumps(
                                {"id": id, "text": docu, "claim": new_claim, "label": "INCORRECT", "tag": tag,
                                 "origin_claim": claim}) + "\n")

    print("Numerical inference test data in {} done!".format(save_dir))
