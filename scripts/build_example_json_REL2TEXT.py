import argparse
import json
import pathlib
import re
from collections import Counter, defaultdict
import pandas as pd
import requests
import os


RDF_EXAMPLE_FILE_NAME = "rdf_examples_for_each_pid.json"
RDF_REFERENCE_FILE_NAME = "rdf_ref_for_each_pid.json"
SUBJECT = "<subject>"
OBJECT = "<object>"

GITHUB_URL = "https://raw.githubusercontent.com/kasnerz/rel2text/main/data/desc_cat/rel2text"
RAW_TSV_URL = "https://raw.githubusercontent.com/kasnerz/rel2text/main/data/orig/rel2text/rel2text_raw_annotated.tsv"


def download_file(url, target_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(target_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def ensure_data_files_exist(source_folder):
    files = ["test.json", "train.json", "dev.json"]
    for file in files:
        file_path = source_folder.joinpath(file)
        if not file_path.is_file():
            download_file(f"{GITHUB_URL}/{file}", file_path)


def ensure_raw_tsv_exists(raw_tsv_path):
    if not os.path.isfile(raw_tsv_path):
        download_file(RAW_TSV_URL, raw_tsv_path)


def extract_parts_from_data(data_dict: dict):
    data = data_dict['data']

    result = []
    for item in data:
        head = re.search(r'<head>(.*?)<rel>', item['in']).group(1).strip()
        rel = re.search(r'<rel>(.*?)<tail>', item['in']).group(1).strip()
        tail = re.search(r'<tail>(.*?)<rel_desc>', item['in']).group(1).strip()
        rel_desc = re.search(r'<rel_desc>(.*?)$', item['in']).group(1).strip()
        out = item['out'].strip()

        result.append({"head": head, "rel": rel, "tail": tail, "rel_desc": rel_desc, "out": out})

    return result


def _extract_rel_examples_from_raw(raw_df: pd.DataFrame):
    l = raw_df.loc[:, ["relation", "response", "response_delex"]]
    examples = [(row[1][0], row[1][1], row[1][2]) for row in l.iterrows()]
    d = defaultdict(list)
    d_delex = defaultdict(set)
    for example in examples:
        d[example[0]].append(example[1])
        d_delex[example[0]].add(example[2])

    return d, d_delex


def find_dict_entry(dicts, value_list):
    for key, examples in dicts.items():
        for val in value_list:
            if val in map(lambda d: d["lab"], examples):
                return key
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-folder', help="path to input folder with source rel2text data jsons")
    parser.add_argument('--splits', nargs='+', default=["test", "train", "dev"], help="source data splits to include")
    parser.add_argument('--rel2text-raw-tsv', default="rel2text_raw_annotated.tsv")
    parser.add_argument('--output-folder', default="path to output folder")

    args = parser.parse_args()

    splits = args.splits
    source_folder = pathlib.Path(args.input_folder)
    raw_tsv_path = source_folder.joinpath(args.rel2text_raw_tsv)
    dump_folder = pathlib.Path(args.output_folder)

    source_folder.mkdir(parents=True, exist_ok=True)
    dump_folder.mkdir(parents=True, exist_ok=True)

    # ensure that the data files exist, if not, download them
    ensure_data_files_exist(source_folder)
    # ensure that the raw tsv file exists, if not, download it
    ensure_raw_tsv_exists(raw_tsv_path)

    list_of_extracted_triples_and_labels = []
    for split in splits:
        json_source_file = source_folder.joinpath(f"{split}.json")

        data_dict = json.load(json_source_file.open())
        list_of_extracted_triples_and_labels.extend(extract_parts_from_data(data_dict))

    rdf_example_dict = dict()
    example_count = Counter()
    for entry in list_of_extracted_triples_and_labels:
        rel = entry["rel"]
        entry_rdf = {"s": entry["head"], "r": rel, "o": entry["tail"],
                     "desc": entry["rel_desc"],
                     "lab": entry["out"]}

        if rel not in rdf_example_dict:
            rdf_example_dict[rel] = [entry_rdf]
        else:
            rdf_example_dict[rel].append(entry_rdf)

        example_count[rel] += 1

    _dump_file = dump_folder.joinpath(f"({'|'.join(splits)}){RDF_EXAMPLE_FILE_NAME}")
    json.dump(rdf_example_dict, _dump_file.open("w"), indent=2)
    print(f"REL2TEXT relation example file saved to: {_dump_file}")

    df_rel2text_raw = pd.read_table(raw_tsv_path)
    raw_examples_for_each_rel, delex_labels_for_each_rel = _extract_rel_examples_from_raw(df_rel2text_raw)

    idx2rel_dict = {}
    for source_rel, examples in raw_examples_for_each_rel.items():
        if source_rel in rdf_example_dict.keys():
            idx2rel_dict[source_rel] = source_rel
        else:
            rel = find_dict_entry(rdf_example_dict, examples)
            idx2rel_dict[rel if rel is not None else source_rel] = source_rel

    assert len(idx2rel_dict) == len(raw_examples_for_each_rel)
    assert len(idx2rel_dict) == len(delex_labels_for_each_rel)
    assert set(raw_examples_for_each_rel.keys()) == set(idx2rel_dict.values())

    json.dump(idx2rel_dict, dump_folder.joinpath("index_rel_dict.json").open("w"), indent=2)

    references = {}
    for target_key, raw_key in idx2rel_dict.items():
        references[target_key] = [delex_sent.replace("[HEAD]", SUBJECT).replace("[TAIL]", OBJECT) for delex_sent in delex_labels_for_each_rel[raw_key]]

    json.dump(references, dump_folder.joinpath("reference.json").open("w"), indent=2)

    print(f"Preprocessed Rel2Text data were generated at {dump_folder}")
