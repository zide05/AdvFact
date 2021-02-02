import json
import argparse
from Antonyms_stress_test import verb_adj_change
from Named_entity_stress_test import entity_change
from Numerical_inference_strss_test import numerical_inference_change
from Span_lack_stress_test import span_change
import os

trans_map = {"verb_adj_change": verb_adj_change, "entity_change": entity_change,
             "numerical_inference_change": numerical_inference_change, "span_change": span_change}


def read_jsonl(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            line = line.strip()
            tmp = json.loads(line)
            data.append((tmp['id'], tmp['text'], tmp['claim'], tmp['label']))
    return data


def transform_main(path, save_dir, trans_type, reorder):
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    data = read_jsonl(path)
    if trans_type == "all":
        for key in trans_map.keys():
            if key == "span_change":
                trans_map[key](data, save_dir, reorder)
            else:
                trans_map[key](data, save_dir)
    else:
        if trans_type == "span_change":
            trans_map[trans_type](data, save_dir, reorder)
        else:
            trans_map[trans_type](data, save_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("-path", required=True,
                        help="file path with factcc format (jsonl with keys: 'id', 'text','claim','label', attention: here the claim need to be one sentence) to readin")
    parser.add_argument("-save_dir", required=True, help="directory to save generated data")
    parser.add_argument("-trans_type", required=True, help="determine transformation type",
                        choices=["verb_adj_change", "entity_change", "numerical_inference_change",
                                 "span_change", "all"])
    parser.add_argument("-reorder", nargs="?", const=True, default=False,
                        help="it is needed only when span_change, if need reorder transformation then set it as true others false")
    args = parser.parse_args()
    transform_main(args.path, args.save_dir, args.trans_type, args.reorder)
