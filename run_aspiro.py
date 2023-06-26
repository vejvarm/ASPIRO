import argparse
import json
import logging
import pathlib

from langchain import PromptTemplate, LLMChain
from langchain.base_language import BaseLanguageModel
from langchain.chat_models import ChatOpenAI
from langchain.llms import OpenAI
from langchain.llms.fake import FakeListLLM
from langchain.schema import OutputParserException
from tqdm import tqdm

from error_analysis import analyze_and_save_errors
from flags import (LOG_ROOT, Templates, TemplateErrors, ERROR_MESSAGES, ModelChoices,
                   OPENAI_REQUEST_TIMEOUT, RDF_EXAMPLE_FILE_NAME, BACKUP_TEMPLATE)
from helpers import setup_logger, load_examples, load_and_validate_config
from parsing import (build_output_dict, prepare_prompt, MultiRetryParser, ConsistencyValidator, TextOutputParser)

LOGFILE_PATH = LOG_ROOT.joinpath(pathlib.Path(__file__).name.removesuffix(".py")+".log")
LOGGER = setup_logger(__name__, loglevel=logging.WARNING, output_log_file=LOGFILE_PATH)


def initialize_chains(model_choices: list[ModelChoices], prompt: PromptTemplate, max_tokens_to_generate: int,
                     temperature: float, stop_sequences: list[str], template_file: Templates) -> tuple[list[BaseLanguageModel], list[LLMChain]]:
    models = []
    chains = []
    for model_choice in model_choices:
        if model_choice not in ModelChoices:
            raise NotImplementedError(f"Choose one of {list(ModelChoices)} for model Variant")

        if isinstance(model_choice.value, pathlib.Path):
            llm = FakeListLLM(responses=build_llm_inputs(template_file, model_choice.value))
        elif model_choice in [ModelChoices.G3P5, ModelChoices.G3]:
            model_id = model_choice.value
            llm = OpenAI(model_name=model_id, temperature=temperature, max_tokens=max_tokens_to_generate,
                         top_p=1, frequency_penalty=0, request_timeout=OPENAI_REQUEST_TIMEOUT, stop=stop_sequences)
        elif model_choice in [ModelChoices.G3P5T, ModelChoices.G3P5T_0301, ModelChoices.G4, ModelChoices.G4_0314]:
            model_id = model_choice.value
            llm = ChatOpenAI(model_name=model_id, temperature=temperature, max_tokens=max_tokens_to_generate,
                             stop=stop_sequences, request_timeout=OPENAI_REQUEST_TIMEOUT)
        else:
            raise NotImplementedError(f"Choose one of {list(ModelChoices)} for model Variant")

        models.append(llm)
        chains.append(LLMChain(prompt=prompt, llm=llm))

    return models, chains


def dehalucinate(dc: ConsistencyValidator, text: str, metadata: dict, keep_result_with_better_score):
    text = dc.run(text, metadata, keep_result_with_better_score)
    return text


def build_llm_inputs(template_file: Templates, path_to_data: pathlib.Path, remove_empty_lines=True):

    if path_to_data.suffix == ".txt":
        inputs_list = path_to_data.open().readlines()
    elif path_to_data.suffix == ".json":
        json_dict = json.load(path_to_data.open())
        inputs_list = [v["output"] for v in json_dict.values()]
    elif path_to_data.suffix == ".jsonl":
        list_of_jsons = path_to_data.open().readlines()
        inputs_list = [list(json.loads(row).values())[0][0] for row in list_of_jsons]
    else:
        raise NotImplementedError("provided path to saved model outputs is not in valid format.")

    if remove_empty_lines:
        inputs_list = [row for row in inputs_list if row != "\n"]

    if "json" in template_file.value.name:
        metadata = json.load(template_file.value.with_suffix(".json").open())
        fake_inputs = [json.dumps({metadata["first_key"]: "", metadata["output_key"]: row}) for row in inputs_list]
    else:
        fake_inputs = inputs_list

    return fake_inputs


# debugging
START_FROM, BREAK_AFTER = (None, None)   # @debug param: start from example num. `START_FROM` and stop at num. `BREAK_AFTER`


def _to_results(result_entry_dict: dict, final_output_dict: dict, intermediate_result_file: pathlib.Path):
    with intermediate_result_file.open("a") as f_intermediate_result:
        f_intermediate_result.write(json.dumps(result_entry_dict)+"\n")
    final_output_dict.update(result_entry_dict)


def main(args):

    config = load_and_validate_config(args.config)
    n_runs = config["n_runs"]  # @param
    dataset_choice = config["dataset_choice"]  # @param
    template_file = config["initial_template"]  # @param (NOTE! two-shot only works with V10 or above)
    model_choices = config["llm_stack"]  # @param
    max_retry_shots = config["max_retry_shots"]  # @param (0 ... zero-shot, 1 ... one-shot, ....)
    consist_val_model = config["consist_val_model"]  # @param  (if None, don't use dehalucination)
    example_format = config["example_format"]  # @param
    max_fetched_examples_per_pid = config["max_fetched_examples_per_pid"]  # @param
    error_dump_subfolder = config["error_dump_subfolder"]  # @param (if None, errors are not saved)
    use_backup = config["use_backup"]  # @param (to change backup template, change flags.BACKUP_TEMPLATE)

    # GENERAL MODEL HYPERPARAMS
    max_tokens_to_generate = config["max_tokens_to_generate"]  # @param
    temperature = config["temperature"]  # @param
    stop_sequences = config["stop_sequences"]  # @param

    # Consistency Validation HYPERPARAMS
    cv_metric = ConsistencyValidator.Metrics.PARENT
    cv_threshold = config["cv_threshold"]  # about 157 prompts in the original
    cv_template = config["cv_template"]
    cv_keep_better = config["cv_keep_better"]  # @param

    dataset_folder = dataset_choice.value
    assert all(model in ModelChoices for model in model_choices)
    assert consist_val_model in ModelChoices

    # INITIALIZE PROMPT and PARSER
    prompt, output_parser = prepare_prompt(template_file, example_format)
    # LOGGER.warning(prompt.format(examples="hi\nyou"))

    # INITIALIZE LANGCHAIN with specified `model_choices` and `prompt`
    models, llm_chains = initialize_chains(model_choices, prompt, max_tokens_to_generate, temperature,
                                        stop_sequences, template_file)
    retry_parser = MultiRetryParser.from_llms(parser=output_parser, llms=models)

    # INITIALIZE dehalucination chain class
    dc_prompt_version = f"_{cv_template.name.lower()}"
    consistency_validator_log = LOG_ROOT.joinpath(dataset_choice.name).joinpath(template_file.name).joinpath(
        f"{','.join(m.name for m in model_choices)}({max_retry_shots}shot)({consist_val_model.name}){dc_prompt_version}.jsonl")  # @param

    if consist_val_model.value is not None:
        consistency_validator_log.parent.mkdir(parents=True, exist_ok=True)
        prompt_template = cv_template.value.open().read()
        prompt_metadata = json.load(cv_template.value.with_suffix(".json").open())
        # TODO make work for versions below v4
        dc = ConsistencyValidator(cv_metric, cv_threshold, consist_val_model, prompt_template,
                                  source_data_key=prompt_metadata["source_data_key"],
                                  first_key=prompt_metadata["first_key"],
                                  output_key=prompt_metadata["output_key"],
                                  stop=prompt_metadata["stop"],
                                  path_to_jsonl_results_file=consistency_validator_log)
    else:
        dc = None

    # Load pids and rdfs examples from path_to_fetched_example_json
    path_to_fetched_example_json = dataset_folder.joinpath(RDF_EXAMPLE_FILE_NAME)
    pid_examples_dict = load_examples(path_to_fetched_example_json, max_fetched_examples_per_pid, example_format,
                                      dataset_choice)

    run = 0
    runs_left = n_runs
    while runs_left > 0:
        # Define files for results
        retry_models = ",".join([mch.name for mch in model_choices[1:]]) if (max_retry_shots > 0
                                                                             and len(model_choices) > 1) else "NONE"
        path_to_output_template_json = dataset_folder.joinpath(
            f"{template_file.name}").joinpath(
            f"{model_choices[0].name}({retry_models})({max_retry_shots}shot)({consist_val_model.name or 'NONE'})").joinpath(
            f"max_examples{max_fetched_examples_per_pid}").joinpath(f"run{run:02d}").joinpath(
            f"templates-{dataset_choice.name.lower()}_{template_file.name}.json")
        if path_to_output_template_json.parent.exists():
            print(f"RUN NUMBER: {run} (EXISTS)")
            run += 1
            continue

        runs_left -= 1
        path_to_output_template_json.parent.mkdir(parents=True, exist_ok=True)
        print(f"RUN NUMBER: {run} (left: {runs_left})")

        output_pid_template_dict = {}
        intermediate_result_file = path_to_output_template_json.with_suffix(".jsonl")
        k = 0
        while intermediate_result_file.exists():
            k += 1
            LOGGER.warning(f"(k={k}):\n\t"
                           f"The intermediate results file already exists at path: {intermediate_result_file}")
            intermediate_result_file = intermediate_result_file.with_stem(f"{intermediate_result_file.stem}({k})")
        backup_count = 0
        for i, (pid, example) in tqdm(enumerate(pid_examples_dict.items()), total=len(list(pid_examples_dict.keys()))):
            rdf_example, subj_labs, rel_labs, obj_labs = example

            # debugging purposes
            if START_FROM is not None and i < START_FROM:
                for mdl in model_choices:
                    if isinstance(mdl.value, pathlib.Path):
                        _ = llm_chains[0].run(rdf_example)
                continue

            # debugging purposes
            if BREAK_AFTER is not None and i == BREAK_AFTER:
                break

            if not rdf_example:
                # TODO: try with generic example
                err = ERROR_MESSAGES[TemplateErrors.NA]
                LOGGER.warning(f"({pid}) {TemplateErrors.NA.value}: {err}']")
                out_dict = {pid: build_output_dict("", [TemplateErrors.NA.value], [err], rdf_example, subj_labs, obj_labs)}
                _to_results(out_dict, output_pid_template_dict, intermediate_result_file)
                continue

            unique_rel_labs = list(set(rel_labs))
            if len(unique_rel_labs) == 1:
                rel_lab = unique_rel_labs[0]
            else:
                raise NotImplementedError("Example structures must have only 1 unique relation in all their entries")

            metadata = {"data": rdf_example, "reference": rel_lab, "relation_label": rel_lab,
                        "rdf_example": rdf_example, "subj_labels": subj_labs, "obj_labels": obj_labs}

            # Zero-shot
            try:
                answer = llm_chains[0].run(rdf_example)
            except Exception as err:
                LOGGER.warning(f"({pid}) {TemplateErrors.API.value}: {err}.")
                out_dict = {pid: build_output_dict("", [TemplateErrors.API.value], [repr(err)],
                                                                        rdf_example, subj_labs, obj_labs)}
                _to_results(out_dict, output_pid_template_dict, intermediate_result_file)
                continue

            # parse the answer
            shot = 0
            try:
                shot, output_dict = retry_parser.parse_with_prompt(answer, prompt.format_prompt(examples=rdf_example),
                                                                    shot=shot, max_shots=max_retry_shots, metadata=metadata)
            except OutputParserException as err:
                LOGGER.info(f'({pid}) {TemplateErrors.PARSING.value}: {err}')
                shot = max_retry_shots
                output_dict = json.loads(str(err))

            if use_backup:
                if not ("<subject>" in output_dict["output"] and "<object>" in output_dict["output"]):
                    output_dict["output"] = BACKUP_TEMPLATE.format("<subject>", rel_lab, "<object>")
                    backup_count += 1

            output_dict = build_output_dict(output=output_dict["output"],
                                        error_codes=output_dict["error_codes"],
                                        error_messages=output_dict["error_messages"],
                                        rdf_example=rdf_example,
                                        subj_labels=subj_labs, obj_labels=obj_labs, shot=shot)
            # dehalucinate
            if dc is not None:
                text = dehalucinate(dc, output_dict["output"], metadata, cv_keep_better)
                output_dict["output"] = text

            final_templates = {pid: output_dict}
            _to_results(final_templates, output_pid_template_dict, intermediate_result_file)

        json.dump(output_pid_template_dict, path_to_output_template_json.open("w"), indent=2)
        print(f"Output saved into {path_to_output_template_json}")

        if error_dump_subfolder is not None:
            _folder_to_dump_error_jsons = path_to_output_template_json.parent.joinpath(error_dump_subfolder)
            _folder_to_dump_error_jsons.mkdir(parents=True, exist_ok=True)
            analyze_and_save_errors(path_to_output_template_json, _folder_to_dump_error_jsons, parser=TextOutputParser())
            err_counts_file = _folder_to_dump_error_jsons.joinpath("errCOUNTS.json")
            err_counts_dict = json.load(err_counts_file.open("r"))
            err_counts_dict["BACKUPS"] = backup_count
            json.dump(err_counts_dict, err_counts_file.open("w"), indent=2)
            print(f"Error analysis saved into: {_folder_to_dump_error_jsons}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Process the path to the configuration file.')
    parser.add_argument('--config', type=str, default="setups/json_default.json", help='Path to the configuration file.')
    main(parser.parse_args())
