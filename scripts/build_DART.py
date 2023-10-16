import argparse
import json
import pathlib
from collections import Counter, defaultdict
import requests
import os


RDF_EXAMPLE_FILE_NAME = "rdf_examples_for_each_pid.json"

TEMP_JSON_URL = "https://raw.githubusercontent.com/szxiangjn/any-shot-data2text/main/temp.json"
DART_DATA_URL = "https://raw.githubusercontent.com/szxiangjn/any-shot-data2text/main/data/dart"


def download_file(url, target_path):
    response = requests.get(url, stream=True)
    response.raise_for_status()
    with open(target_path, 'wb') as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def ensure_temp_json_exists(temp_json_path):
    if not os.path.isfile(temp_json_path):
        download_file(TEMP_JSON_URL, temp_json_path)


def ensure_dart_data_exists(dart_folder):
    # Here, you would need to list all the files to be downloaded
    files = ["test_both.json", "train.json", "val.json"]
    for file in files:
        file_path = dart_folder.joinpath(file)
        if not file_path.is_file():
            download_file(f"{DART_DATA_URL}/{file}", file_path)


def map_relation_to_dart(relation_label):
    """transformation between `relation_label` and `relation_label_uppercase_w_underscores`"""
    # Remove leading and trailing whitespace
    relation_label = relation_label.strip()
    # Replace spaces with underscores
    relation_label = relation_label.replace(' ', '_').replace('__', '_')
    # Convert to uppercase
    relation_label = relation_label.upper()
    # print(relation_label)

    return relation_label


def load_jsonl_to_list_of_dicts(file_path: pathlib.Path):
    return [json.loads(row) for row in file_path.open().readlines()]


def extract_triples_from_dart_string(dart_rdf_string: str):
    # Split the string into segments based on the <H> marker
    segments = dart_rdf_string.split('<H>')

    # Remove empty segments
    segments = [segment.strip() for segment in segments if segment.strip()]

    triples = []

    for segment in segments:
        # Split the segment into components based on the <R> and <T> markers
        components = segment.split('<R>', 1)[1].split('<T>', 1)

        # Trim whitespace from each component
        components = [component.strip() for component in components]

        # Split the subject out from the relation and object components
        subject = segment.split('<R>', 1)[0].strip()

        triples.append((subject, components[0], components[1]))

    return triples


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-folder', help="path to input folder with source temp.json from ASDOT github")
    parser.add_argument('--output-folder', default="path to output folder")

    args = parser.parse_args()

    source_folder = pathlib.Path(args.input_folder)
    asdot_template_jsonl = source_folder.joinpath("temp.json")     # source: https://raw.githubusercontent.com/szxiangjn/any-shot-data2text/main/temp.json
    dart_folder = source_folder.joinpath("data").joinpath("dart")  # source: https://github.com/szxiangjn/any-shot-data2text/tree/main/data/dart
    max_examples = None
    excluded_subjects = ["[TABLECONTEXT]"]
    excluded_relations = ["[TITLE]"]

    dump_folder = pathlib.Path(args.output_folder)

    dart_folder.mkdir(parents=True, exist_ok=True)
    dump_folder.mkdir(parents=True, exist_ok=True)

    # ensure that the temp.json file exists, if not, download it
    ensure_temp_json_exists(asdot_template_jsonl)

    # ensure that the dart data files exist, if not, download them
    ensure_dart_data_exists(dart_folder)

    templates = load_jsonl_to_list_of_dicts(asdot_template_jsonl)
    examples = [load_jsonl_to_list_of_dicts(fl) for fl in dart_folder.glob("*.json")]
    examples = [e for ex in examples for e in ex]
    single_examples = [ex for ex in examples if ex["triple"].count("<H>") == 1]

    r2dart_map = {}
    for row in templates:
        r_label, template = list(row.items())[0]
        r2dart_map[r_label] = map_relation_to_dart(r_label)
        # r_label, template = row.items()
        # print(r_label)
    # print(r2dart_map)

    dart2r_map = {v: k for k, v in r2dart_map.items()}
    # print(dart2r_map)

    example_triples = []
    relation_aggregated_example_dict = defaultdict(set)
    # relation_aggregated_example_dict = {k: set() for k in dart2r_map.keys()}  # dict.fromkeys(r2dart_map.keys(), [[] for _ in r2dart_map])  # {key: [] for key in r2dart_map.keys()}
    # print(relation_aggregated_example_dict)

    json.dump(r2dart_map, dump_folder.joinpath("rlabel2dart_map.json").open("w"), indent=2)

    count_missing_templates = Counter()
    for example in examples:
        # print(example["triple"])
        # print(extract_triples_from_dart_string(example["triple"]))
        triples = extract_triples_from_dart_string(example["triple"])
        # example_triples.append(triples)
        for triple in triples:
            s, r, o = triple
            r = map_relation_to_dart(r)
            if s in excluded_subjects:
                continue
            if r in excluded_relations:
                continue
            if len(relation_aggregated_example_dict[r]) == max_examples:
                continue

            try:
                r_lower = dart2r_map[r]
            except KeyError:
                count_missing_templates.update([r])
                print(triple)
                r_lower = r.lower()
                dart2r_map[r] = r_lower
                # print(f"r: {r} has no examples")

            relation_aggregated_example_dict[r].add((s, r_lower, o))

    json.dump(dart2r_map, dump_folder.joinpath("dart2rlabel_map.json").open("w"), indent=2)

    print(f"Unique relations: {len(relation_aggregated_example_dict)}")

    rel_rdf_example_dict = dict()
    for k in dart2r_map.keys():
        rdf_set = sorted(relation_aggregated_example_dict[k], key=lambda entr: len(entr[0]), reverse=True)
        for entry in rdf_set:
            if k not in rel_rdf_example_dict.keys():
                rel_rdf_example_dict[k] = [{"s": entry[0], "r": entry[1], "o": entry[2]}]
            else:
                rel_rdf_example_dict[k].append({"s": entry[0], "r": entry[1], "o": entry[2]})

    # rel_rdf_example_dict = {k:  for k, rdf_set in relation_aggregated_example_dict.items() for entry in rdf_set}
    json.dump(rel_rdf_example_dict, dump_folder.joinpath(RDF_EXAMPLE_FILE_NAME).open("w"), indent=2)

    print(f"Preprocessed DART data were generated at {dump_folder}")

    # test consistency of mappings:
    template_dict = {}
    [template_dict.update(ent) for ent in templates]
    for i, (key, val) in enumerate(template_dict.items()):
        assert key in r2dart_map
        # print(f"i: {i} | {key}: {r2dart_map[key]}")


