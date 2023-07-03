import json
import pathlib
from abc import ABC, abstractmethod
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
                    "HFacePipeline": [ModelChoices.FALCON_7B]}


class NShotGenerator:
    pass

    def run(self):
        pass


class LLMBuilder:

    def __init__(self):
        self.chains = []
        self.llms = []
        self.initialized_local_models = dict()
        self.cv_llm = None

    def initialize_chains(self, model_choices: list[ModelChoices], prompt: PromptTemplate, max_tokens_to_generate: int,
                      temperature: float, stop_sequences: list[str], template_file: Templates):
        for model_choice in model_choices:
            if model_choice not in ModelChoices:
                raise NotImplementedError(f"Choose one of {list(ModelChoices)} for model Variant")

            model_id = model_choice.value

            if isinstance(model_id, pathlib.Path):
                llm = FakeListLLM(responses=self._build_llm_inputs(template_file, model_id))
            elif model_choice in MODEL_CATEGORIES["OpenAI"]:
                llm = OpenAI(model_name=model_id, temperature=temperature, max_tokens=max_tokens_to_generate,
                             top_p=1, frequency_penalty=0, request_timeout=OPENAI_REQUEST_TIMEOUT, stop=stop_sequences)
            elif model_choice in MODEL_CATEGORIES["ChatOpenAI"]:
                llm = ChatOpenAI(model_name=model_id, temperature=temperature, max_tokens=max_tokens_to_generate,
                                 stop=stop_sequences, request_timeout=OPENAI_REQUEST_TIMEOUT)
            elif model_choice in MODEL_CATEGORIES["HFacePipeline"]:
                llm = self._init_hf_llm(model_choice, temperature=temperature, load_in_8bit=True, load_in_4bit=False,
                                        top_k=10, max_tokens_to_generate=max_tokens_to_generate,
                                        stop_sequences=stop_sequences)
            else:
                raise NotImplementedError(f"Choose one of {list(ModelChoices)} for model Variant")

            self.llms.append(llm)
            self.chains.append(LLMChain(prompt=prompt, llm=llm))

    def initialize_cv_llm(self, model_choice: ModelChoices, **config):
        temperature = config["temperature"]
        max_tokens = config["max_tokens"]
        stop = config["stop_sequences"]

        if model_choice in MODEL_CATEGORIES["HFacePipeline"]:
            return self._init_hf_llm(model_choice, temperature=temperature, load_in_8bit=True, load_in_4bit=False,
                                     top_k=10, max_tokens_to_generate=max_tokens, stop_sequences=stop)

        if model_choice in MODEL_CATEGORIES["ChatOpenAI"]:
            llm_class = ChatOpenAI
        elif model_choice in MODEL_CATEGORIES["OpenAI"]:
            llm_class = OpenAI
        else:
            raise NotImplementedError(f"Choose one of {list(ModelChoices)} for model API")
        return llm_class(model_name=model_choice.value, temperature=temperature, max_tokens=max_tokens, stop=stop,
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
                                                         )
            self.initialized_local_models[model_choice] = (tokenizer, model)

        # tokenizer.pad_token_id = tokenizer.eos_token_id
        pipe = pipeline("text-generation",
                        model=model,
                        tokenizer=tokenizer,
                        max_new_tokens=config["max_tokens_to_generate"],
                        top_k=config["top_k"],  # 10
                        num_return_sequences=1,
                        stop_sequence=config["stop_sequences"][0],
                        device_map="auto", )
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
