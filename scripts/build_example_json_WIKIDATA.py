import argparse
import json
import pathlib
import random
from collections import Counter

import requests
from tqdm import tqdm

from typing import Sequence


DEFAULT_FOLDER = pathlib.Path("../data/wikidata")
SUBJECT = "<subject>"
OBJECT = "<object>"


def _fetch_entities(pid, props="claims|labels|descriptions|aliases"):
    url = 'https://www.wikidata.org/w/api.php'

    params = {
        'action': 'wbgetentities',
        'ids': pid,
        'languages': 'en',
        'props': props,
        'format': 'json'
    }

    response = requests.get(url, params=params)
    return response.json()


def _fetch_claims(sid, pid):
    url = 'https://www.wikidata.org/w/api.php'
    params = {
        'action': 'wbgetclaims',
        'entity': sid,
        'property': pid,
        'format': 'json'
    }
    response = requests.get(url, params=params)
    return response.json()


def _uppercase_sequence(sequence: Sequence[str], tp):
    if not isinstance(sequence, tp):
        return sequence

    new_sequence = []
    for ent in sequence:
        try:
            new_sequence.append(ent.upper())
        except AttributeError:
            raise AttributeError("Entity in sequence is not a string!")
    return tp(new_sequence)


def uppercase(f):

    def wrap(entry, *args, **kwargs):
        if isinstance(entry, str):
            entry = entry.upper()
        else:
            entry = _uppercase_sequence(entry, list)
            entry = _uppercase_sequence(entry, tuple)

        return f(entry, *args, **kwargs)

    return wrap


def _get_english_label_from_wikidata(eid: str):
    data = _fetch_entities(eid, props="labels")
    return data['entities'][eid]['labels']['en']['value']


def _get_all_labels_from_wikidata(eid: str):
    data = _fetch_entities(eid, props="labels")
    return data['entities'][eid]['labels']


@uppercase
def get_label(gid: str, id2label_dict: dict = None, memory: dict = None):
    """Generic get label for given gid in given index if it exists"""
    label = None
    if memory is not None:
        try:
            label = memory[gid]
        except KeyError:
            print(f" x", end="")  # entity/relation with {gid} is not in memory yet.

    if label is None and id2label_dict is not None:
        # print("Fetching from id2label_dict ... ")
        try:
            label = id2label_dict[gid]
        except KeyError:
            print(f"x", end="")  # entity/relation with {gid} is not in given entity dictionary (id2label_dict).

    if label is None:
        # print("Fetching from Wikidata ...")
        try:
            label = _get_english_label_from_wikidata(gid)
        except Exception as err:
            print(f"{gid}: {repr(err)}", end="")
            label = None

    if memory is not None and label is not None:
        memory.update({gid: label})

    return label


def get_wikidata_property_example(pid):

    data = _fetch_entities(pid)
    label = ""
    description = ""
    aliases = []
    triples = []

    if 'entities' not in data:
        # print(data)
        return label, description, aliases, triples

    if pid not in data['entities']:
        # print(data['entities'])
        return label, description, aliases, triples

    entity = data['entities'][pid]
    label = entity['labels']['en']['value'] if 'labels' in entity and 'en' in entity['labels'] else label
    description = entity['descriptions']['en']['value'] if 'descriptions' in entity and 'en' in entity['descriptions'] else description
    aliases = [alias['value'] for alias in entity['aliases']['en']] if 'aliases' in entity and 'en' in entity['aliases'] else aliases

    if 'claims' not in entity or 'P1855' not in entity['claims']:
        return label, description, aliases, triples

    examples = entity['claims']['P1855']
    # print(examples)
    # exit()
    for example in examples:
        if 'mainsnak' in example and 'datavalue' in example['mainsnak']:
            datavalue = example['mainsnak']['datavalue']

            if datavalue['type'] == 'wikibase-entityid':
                subject_entity = datavalue['value']['id']
                data = _fetch_claims(subject_entity, pid)

                if pid in data['claims']:
                    for claim in data['claims'][pid]:
                        if 'mainsnak' in claim and 'datavalue' in claim['mainsnak']:
                            datavalue = claim['mainsnak']['datavalue']
                            try:
                                if datavalue['type'] == 'wikibase-entityid':
                                    object_entity = datavalue['value']['id']
                                elif datavalue['type'] == 'quantity':
                                    amount = datavalue['value']['amount']
                                    if datavalue['value']['unit'] == '1':
                                        unit_label = ""
                                    else:
                                        unit_id = datavalue['value']['unit'].split("/")[-1]
                                        data = _fetch_entities(unit_id, props="labels")
                                        unit_label = data['entities'][unit_id]['labels']['en']['value']
                                    object_entity = f"{amount} {unit_label}"
                                    # print(f"amount: {object_entity}", end="")
                                elif datavalue['type'] == 'time':
                                    time = datavalue['value']['time']
                                    object_entity = time
                                    # print(f"time: {object_entity}", end="")
                                elif datavalue['type'] == 'string':
                                    object_entity = datavalue['value']
                                    # print(f"string: {object_entity}", end="")
                                elif datavalue['type'] == 'monolingualtext':
                                    text = datavalue['value']['text']
                                    language = datavalue['value']['language']
                                    object_entity = f"{text} ({language})"
                                    # print(f"text: {object_entity}", end="")
                                else:
                                    object_entity = datavalue
                                    # print(f"other: {object_entity}")
                            except KeyError as err:
                                object_entity = f"<<<{repr(err)}>>> {datavalue}"
                                print(f"{object_entity} (fix manually in `rel_metadata.json`)")
                            triples.append((subject_entity, pid, object_entity))

    return label, description, aliases, triples


def _random_id(length=4):
    return "".join(str(random.randint(0, 9)) for _ in range(length))


def main(args):
    input_folder = pathlib.Path(args.input_folder)
    rel_dict_path = input_folder.joinpath("index_rel_dict.json")
    assert rel_dict_path.exists(), f"Place `index_rel_dict.json` to `{input_folder}`"
    rel_dict = json.load(rel_dict_path.open())
    output_folder = pathlib.Path(args.output_folder)
    output_folder.mkdir(parents=True, exist_ok=True)
    if args.labels_file != "":
        labels_dict_path = pathlib.Path(args.labels_file)
        assert labels_dict_path.exists(), f"labels-file argument was provided, but it does not lead to existing file."
        template_labels_dict = json.load(labels_dict_path.open())
    else:
        template_labels_dict = None

    # Phase 1: Fetch metadata for each unique pid in WIKIDATA as defined in `index_rel_dict.json`
    output_rel_metadata_json_path = output_folder.joinpath("rel_metadata.json")
    rel_metadata_dict = dict()
    try:
        rel_metadata_dict = json.load(output_rel_metadata_json_path.open())
    except Exception:
        print("Can not load `rel_metadata.json` ... building a new one:")
        for pid, plabel in tqdm(list(rel_dict.items()), desc="Fetching metadata from Wikidata"):
            label, description, aliases, triples = get_wikidata_property_example(pid)

            rel_metadata_dict[pid] = {"csqa_label": plabel,
                                      "wikidata_label": label,
                                      "description": description,
                                      "aliases": aliases,
                                      "examples": triples}

        json.dump(rel_metadata_dict, output_rel_metadata_json_path.open("w"), indent=2)
        print(f"Relation metadata dictionary saved @`{output_rel_metadata_json_path}`")

    # print(rel_metadata_dict)

    # Phase 2: Build rdf_examples_for_each_pid.json
    id2label_dict_path = input_folder.joinpath("items_wikidata_n.json")
    assert id2label_dict_path.exists(), f"Place `items_wikidata_n.json` to `{input_folder}`"
    id2label_dict = json.load(id2label_dict_path.open())
    path_to_example_json = output_folder.joinpath("rdf_examples_for_each_pid.json")
    path_to_label_memory = output_folder.joinpath("label_memory.json")
    path_to_no_example_json = output_folder.joinpath("no_examples.json")

    rdf_examples_dict = {}
    no_examples = {}
    if path_to_label_memory.exists():
        label_memory = json.load(path_to_label_memory.open())
    else:
        label_memory = {}
    for pid, metadata in tqdm(list(rel_metadata_dict.items()), desc="Building examples for each PID"):
        csqa_label = metadata["csqa_label"]
        wiki_label = metadata["wikidata_label"]
        desc = metadata["description"]
        example_list = []
        if template_labels_dict is not None:
            lab_template = template_labels_dict[csqa_label][0]  # only take the first template if there is more
        else:
            lab_template = ""
        for example in metadata["examples"]:
            s, r, o = example

            s_label = get_label(s, id2label_dict, label_memory)
            o_label = get_label(o, id2label_dict, label_memory) if o.startswith("Q") else o

            if s_label is None or o_label is None:
                print(f"@{r}: skipping example {(s_label, csqa_label, o_label, desc, pid)} (missing labels for subject or object).")
                continue

            lab = lab_template.replace(SUBJECT, s_label).replace(OBJECT, o_label)
            d_entry = {"s": s_label, "r": csqa_label, "o": o_label, "desc": desc, "id": pid, "wiki_label": wiki_label, "lab": lab}

            example_list.append(d_entry)

        if not example_list:
            no_examples.update({pid: csqa_label})
            subj = "Q"+_random_id(5)
            obj = "Q"+_random_id(5)
            lab = lab_template.replace(SUBJECT, subj).replace(OBJECT, obj)
            example_list = [{"s": subj, "r": csqa_label, "o": obj, "desc": desc, "id": pid, "wiki_label": wiki_label, "lab": lab}]

        rdf_examples_dict[csqa_label] = example_list

    json.dump(label_memory, path_to_label_memory.open("w"), indent=2)
    json.dump(no_examples, path_to_no_example_json.open("w"), indent=2)
    json.dump(rdf_examples_dict, path_to_example_json.open("w"), indent=2)

    # Phase 3: Analyze and Validate
    rdf_examples_from_disk = json.load(path_to_example_json.open("r"))

    assert set(rel_dict.values()) == set(rdf_examples_from_disk.keys())
    print(f"rel_dict: {len(rel_dict)} | metadata_dict: {len(rel_metadata_dict)} | example_dict: {len(rdf_examples_from_disk)}")
    # c = Counter({k: len(v) for k, v in rdf_examples_from_disk.items()})

    print(f"Preprocessed Wikidata examples were generated at {path_to_example_json.parent}")


if __name__ == "__main__":
    path_to_dataset = DEFAULT_FOLDER

    parser = argparse.ArgumentParser()
    parser.add_argument('--input-folder', help="path to input folder with source rel2text data jsons", type=str, default=path_to_dataset)
    parser.add_argument('--output-folder', help="path to output folder", type=str, default=path_to_dataset)
    parser.add_argument('--labels-file', help="(optional) file with label templates.", type=str, default="")

    main(parser.parse_args())
