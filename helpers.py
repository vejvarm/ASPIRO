import json
import logging
import pathlib
from typing import Union, Sequence

from ordered_set import OrderedSet
from aenum import extend_enum

from flags import RDFExampleFormat, DatasetChoice, RDF_SEPARATOR
from flags import Templates, ModelChoices, ConsistencyTemplateNames


# Define a function that will load and validate the config
def load_and_validate_config(config_file):
    with open(config_file, 'r') as f:
        config = json.load(f)

    config['file_name'] = pathlib.Path(config_file).stem

    # Validate the values against the Enums
    config['dataset_choice'] = DatasetChoice[config['dataset_choice'].upper()]
    config['initial_template'] = Templates[config['initial_template'].upper()]
    config['llm_stack'] = _parse_model_choices(config['llm_stack'], config['file_name'])
    config['consist_val_model'] = _parse_model_choices([config['consist_val_model']], "CV")[0]
    config['example_format'] = RDFExampleFormat[config['example_format'].upper()]
    config['cv_template'] = ConsistencyTemplateNames[config['cv_template'].upper()]

    return config


def _parse_model_choices(model_list: list[str], model_name: str):
    llm_stack = []
    for model in model_list:
        if pathlib.Path(model).exists():
            path_to_data = pathlib.Path(model)
            extend_enum(ModelChoices, model_name.upper(), path_to_data)
            llm_stack.append(ModelChoices[model_name.upper()])
        elif model.upper() in [m.name for m in ModelChoices]:
            llm_stack.append(ModelChoices[model.upper()])
        elif model.lower() in ModelChoices:
            llm_stack.append(ModelChoices(model.lower()))
        else:
            raise NotImplementedError("Given model is not a valid choice. Refer to class `ModelChoices` in flags.py")
    return llm_stack


def setup_logger(name=__name__, loglevel=logging.DEBUG, handlers=None, output_log_file: pathlib.Path or str = None):
    if handlers is None:
        handlers = [logging.StreamHandler()]
    if output_log_file:
        file_handler = logging.FileHandler(output_log_file, mode="w", encoding="utf-8")
        handlers.append(file_handler)

    logger = logging.getLogger(name)
    logger.setLevel(loglevel)

    formatter = logging.Formatter('%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                                   datefmt='%d/%m/%Y %I:%M:%S %p')

    for handler in handlers:
        handler.setLevel(loglevel)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    return logger


def _uppercase_sequence(sequence: Union[Sequence[str], OrderedSet[str]], tp):
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
            entry = _uppercase_sequence(entry, OrderedSet)
            entry = _uppercase_sequence(entry, list)
            entry = _uppercase_sequence(entry, tuple)

        return f(entry, *args, **kwargs)

    return wrap


def format_rdf_entries(entries: list[tuple[str, str, str]], output_type: RDFExampleFormat, sep: str):
    if len(entries) == 0:
        return None
    if output_type == RDFExampleFormat.TABLE:
        return "\\n".join([f"Table: {e[0]} {sep} {e[1]} {sep} {e[2]}" for e in entries])
        # entries.append(f"Table: {s_label} {sep} {r_label} {sep} {o_label}")
    elif output_type == RDFExampleFormat.JSON:
        return json.dumps(entries)
    else:  # RDFExampleFormat.DEFAULT
        return entries


def parse_rdf_list_to_examples(rdf_list: list[dict[str: str]], max_examples: int, output_type: RDFExampleFormat,
                               dataset_choice: DatasetChoice, sep="|") -> tuple[str, list[str], list[str], list[str]]:
    """
    :param rdf_list: (list[dict]) list[{'sid': '', 'rid': '', 'oid': ''}, {'sid': '', 'rid': '', 'oid': ''}, ...]
    :param max_examples: (int) maximum number of examples to parse into the final string
    :param output_type: (RDFExampleFormat) how examples are formatted at the output
    :param dataset_choice: (DatasetChoice[Enum]) which dataset are we loading the examples from
    :param sep: (str) separator between entities for parsed Table string entries
    :return: (tuple[str, list[str], list[str], list[str]]) ("Table: s_label | r_label | o_label\nTable: s_label | r_label | o_label\n ...", s_label, o_label)
    """
    entries = []
    s_labels = []
    r_labels = []
    o_labels = []

    if output_type not in RDFExampleFormat:
        raise NotImplementedError(f"output_type expected to be instance of `RDFExampleFormat` enum (got {output_type})")

    for rdf in rdf_list[:max_examples]:
        if dataset_choice in [DatasetChoice.DART, DatasetChoice.DART_TEST]:
            s_label = rdf["s"]
            r_label = rdf["r"]
            o_label = rdf["o"]
        elif dataset_choice in [DatasetChoice.REL2TEXT, DatasetChoice.REL2TEXT_TEST]:
            s_label = rdf["s"]
            r_label = rdf["r"]
            o_label = rdf["o"]
        elif dataset_choice in [DatasetChoice.WIKIDATA, DatasetChoice.WIKIDATA_TEST]:
            s_label = rdf["s"]
            r_label = rdf["r"]
            o_label = rdf["o"]
        else:
            raise NotImplementedError("Chosen `DatasetChoice` is not supported yet.")

        entries.append((s_label, r_label, o_label))

        s_labels.append(s_label)
        r_labels.append(r_label)
        o_labels.append(o_label)

    output = format_rdf_entries(entries, output_type, sep)

    return output, s_labels, r_labels, o_labels


def load_examples(path_to_example_json: pathlib.Path, max_examples_per_pid: int, output_type: RDFExampleFormat,
                  dataset_choice: DatasetChoice):
    if path_to_example_json.exists():
        fetched_example_dict = json.load(path_to_example_json.open())
    else:
        raise NotImplementedError(f"{path_to_example_json.name} is missing! Please run `scripts/dataset_builders/build` script for the respective dataset first")

    pid_examples_dict = {pid: parse_rdf_list_to_examples(rdf_list, max_examples_per_pid, output_type, dataset_choice, sep=RDF_SEPARATOR)
                         for pid, rdf_list in fetched_example_dict.items()}

    return pid_examples_dict


def make_json_compliant(json_str: str):
    # # Ensure that keys and string values are wrapped with double quotes
    # json_str = re.sub(r'(?:(?<=\{)|(?<=,))\s*(\'[^:\']+\')\s*:', r' \1 :',
    #                   json_str)  # if the key is in single quotes
    # json_str = re.sub(r'(?:(?<=\{)|(?<=,))\s*([^:\'"]+)\s*:', r' "\1" :',
    #                   json_str)  # if the key is without any quotes
    # json_str = re.sub(r':\s*\'([^,\']*)\'', r': "\1"', json_str)  # if the value is in single quotes
    # json_str = re.sub(r':\s*([^\',"{}]*)\s*(?=[,\}])', r': "\1"', json_str)  # if the value is without any quotes
    #
    # # Remove trailing commas
    # json_str = re.sub(r',\s*}', '}', json_str)
    # json_str = re.sub(r',\s*]', ']', json_str)
    # print(json_str)
    json_str = json_str.lstrip('{`\\n\n"').rstrip('.`\\n\n"}')

    # Add '{' and '}' at start and end if not exist
    # print(json_str)
    json_str = '{"' + json_str
    json_str = json_str + '."}'
    # print(json_str)

    # Remove '{' and '}' within the string (not at start or end)
    # json_str = json_str[0] + json_str[1:-1].replace('{', '').replace('}', '') + json_str[-1]

    return json_str