import json
import pathlib
from abc import ABC, abstractmethod
from functools import partial
from typing import Protocol, Type
from dataclasses import dataclass

from langchain import PromptTemplate, LLMChain, OpenAI, HuggingFacePipeline
from langchain.base_language import BaseLanguageModel
from langchain.chat_models import ChatOpenAI
from langchain.llms import FakeListLLM
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline

from flags import ModelChoices, Templates, OPENAI_REQUEST_TIMEOUT


MODEL_CATEGORIES = {"OpenAI": [ModelChoices.G3, ModelChoices.G3P5],
                    "ChatOpenAI": [ModelChoices.G3P5T, ModelChoices.G3P5T_0301, ModelChoices.G3P5T_0613,
                                   ModelChoices.G4, ModelChoices.G4_0314, ModelChoices.G4_0613],
                    "HFacePipeline": [ModelChoices.FALCON_7B, ModelChoices.FALCON_40B]}


class NShotGenerator:
    pass

    def run(self):
        pass


class LLMBuilder:
    # defaults
    TEMPERATURE = 0.
    MAX_NEW_TOKENS = 128
    STOP_SEQUENCES = ["\n"]
    TOP_K = 10
    LOAD_IN_8BIT = False
    LOAD_IN_4BIT = False



    def __init__(self):
        self.chains = []
        self.llms = []
        self.initialized_local_models = dict()
        self.cv_llm = None

    @staticmethod
    def _parse_argument(config: dict, key: str, default_val: any):
        return config[key] if key in config.keys() else default_val

    def _init_config(self, config):
        conf_out = {}
        _parse_config = partial(self._parse_argument, config)
        conf_out["temperature"] = _parse_config("temperature", self.TEMPERATURE)
        conf_out["max_tokens"] = _parse_config("max_tokens_to_generate", self.MAX_NEW_TOKENS)
        conf_out["stop_sequences"] = _parse_config("stop_sequences", self.STOP_SEQUENCES)
        conf_out["top_k"] = _parse_config("top_k", self.TOP_K)
        conf_out["load_in_8bit"] = _parse_config("load_in_8bit", self.LOAD_IN_8BIT)
        conf_out["load_in_4bit"] = _parse_config("load_in_4bit", self.LOAD_IN_4BIT)

        if conf_out["load_in_8bit"] & conf_out["load_in_4bit"]:
            raise NotImplementedError("load_in_8bit and load_in_4bit CAN NOT be both True at once!")

        return conf_out

    def initialize_chains(self, model_choices: list[ModelChoices], prompt: PromptTemplate,
                          template_file: Templates, **config):
        # config = self._init_config(config)

        for model_choice in model_choices:
            if model_choice not in ModelChoices:
                raise NotImplementedError(f"Choose one of {list(ModelChoices)} for model Variant")

            model_id = model_choice.value

            if isinstance(model_id, pathlib.Path):
                llm = FakeListLLM(responses=self._build_llm_inputs(template_file, model_id))
            else:
                llm = self.initialize_llm(model_choice, **config)
            # elif model_choice in MODEL_CATEGORIES["OpenAI"]:
            #     llm = OpenAI(model_name=model_id, temperature=config["temperature"], max_tokens=config["max_tokens"],
            #                  stop=config["stop_sequences"], request_timeout=OPENAI_REQUEST_TIMEOUT)
            # elif model_choice in MODEL_CATEGORIES["ChatOpenAI"]:
            #     llm = ChatOpenAI(model_name=model_id, temperature=config["temperature"], max_tokens=config["max_tokens"],
            #                      stop=config["stop_sequences"], request_timeout=OPENAI_REQUEST_TIMEOUT)
            # elif model_choice in MODEL_CATEGORIES["HFacePipeline"]:
            #     llm = self._init_hf_llm(model_choice, **config)
            # else:
            #     raise NotImplementedError(f"Choose one of {list(ModelChoices)} for model Variant")

            self.llms.append(llm)
            self.chains.append(LLMChain(prompt=prompt, llm=llm))

    def initialize_llm(self, model_choice: ModelChoices, **config):
        config = self._init_config(config)

        if model_choice in MODEL_CATEGORIES["HFacePipeline"]:
            return self._init_hf_llm(model_choice, **config)

        if model_choice in MODEL_CATEGORIES["ChatOpenAI"]:
            llm_class = ChatOpenAI
        elif model_choice in MODEL_CATEGORIES["OpenAI"]:
            llm_class = OpenAI
        else:
            raise NotImplementedError(f"Choose one of {list(ModelChoices)} for model API")
        return llm_class(model_name=model_choice.value, temperature=config["temperature"],
                         max_tokens=config["max_tokens"], stop=config["stop_sequences"],
                         request_timeout=OPENAI_REQUEST_TIMEOUT)

    def _init_hf_llm(self, model_choice: ModelChoices, **config):
        if model_choice in self.initialized_local_models.keys():
            # TODO: disambiguate if config is different
            tokenizer, model = self.initialized_local_models[model_choice]
        else:
            model_id = model_choice.value
            tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(model_id,
                                                         temperature=config["temperature"] + 1e-6,
                                                         load_in_8bit=config["load_in_8bit"],
                                                         load_in_4bit=config["load_in_4bit"],
                                                         trust_remote_code=True,
                                                         device_map="auto")
            self.initialized_local_models[model_choice] = (tokenizer, model)

        # tokenizer.pad_token_id = tokenizer.eos_token_id
        pipe = pipeline("text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens=config["max_tokens"],
                        top_k=config["top_k"],
                        num_return_sequences=1,
                        stop_sequence=config["stop_sequences"][0])
        llm = HuggingFacePipeline(pipeline=pipe)

        return llm

    @staticmethod
    def _build_llm_inputs(template_file: Templates, path_to_data: pathlib.Path, remove_empty_lines=True):
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
