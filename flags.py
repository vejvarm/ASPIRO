import os
import pathlib
from aenum import Enum

os.environ["TOKENIZERS_PARALLELISM"] = "true"


PROJECT_ROOT = pathlib.Path(os.path.dirname(__file__))
LOG_ROOT = PROJECT_ROOT.joinpath("logs")
PROMPT_TEMPLATES_FOLDER = PROJECT_ROOT.joinpath("prompt_templates")

DATA_ROOT = pathlib.Path("data/")
RDF_DATASET_AGGREGATOR_DUMP_ROOT = DATA_ROOT.joinpath("raw")

OPENAI_REQUEST_TIMEOUT = 120


RDF_EXAMPLE_FILE_NAME = "rdf_examples_for_each_pid.json"
RDF_REFERENCE_FILE_NAME = "rdf_ref_for_each_pid.json"


class DatasetChoice(Enum):
    DART = DATA_ROOT.joinpath("dart")
    DART_TEST = DATA_ROOT.joinpath("dart_test")
    REL2TEXT = DATA_ROOT.joinpath("rel2text")
    REL2TEXT_TEST = DATA_ROOT.joinpath("rel2text_test")
    WEBNLG = DATA_ROOT.joinpath("webnlg")
    WEBNLG_TEST = DATA_ROOT.joinpath("webnlg_test")
    WIKIDATA = DATA_ROOT.joinpath("wikidata")
    WIKIDATA_TEST = DATA_ROOT.joinpath("wikidata_test")


RDF_SEPARATOR = "<&SEP>"  # '<&SEP>'  # NOTE: legacy '|' has problems as it is present in some entity labels
NOT_SEPARATOR_FLAG = "<&NOT>"


API_KEY_JSON = PROJECT_ROOT.joinpath("certs/api_keys.json")
SUBJECT = "<subject>"
RELATION = "<relation>"
OBJECT = "<object>"
ENTITY = "<entity>"
CONSTANT_PLACEHOLDER = "<constant>"  # <value>, <constant>, or <object>
REGEX_TEMPLATE_PATTERN = rf"\s*([^\.]*<(?:subject|object|{CONSTANT_PLACEHOLDER[1:-1]})>[^\.]*<(?:object|subject|{CONSTANT_PLACEHOLDER[1:-1]})>[^\.]*\.)\s*?\n*?"  # rf"(.*<subject>.*<object>.*\.)\s*\n"
BACKUP_TEMPLATE = "{}'s {} is {}"  # can be used if all parsing fails. (--use-backup-temps)
BACKUP_TEMPLATE_REGEX = r"^<subject>'s (.*?) is <object>\.?"  # for detecting the above backup templates retroactively


class Templates(Enum):
    DEFAULT = PROMPT_TEMPLATES_FOLDER.joinpath("default.tmp")
    ASDOT = PROMPT_TEMPLATES_FOLDER.joinpath("v0_ASDOT.tmp")
    JSON = PROMPT_TEMPLATES_FOLDER.joinpath("v19_json.tmp")  # v18 without format_instructions
    JSONV19B = PROMPT_TEMPLATES_FOLDER.joinpath("v19b_json.tmp")  # v19 with only one example
    JSONV20 = PROMPT_TEMPLATES_FOLDER.joinpath("v20_json.tmp")  # v19b with pre-filled subject rel object fiels in output

    # old versions
    V5 = PROMPT_TEMPLATES_FOLDER.joinpath("old/v5.tmp")
    V6 = PROMPT_TEMPLATES_FOLDER.joinpath("old/v6.tmp")
    V7 = PROMPT_TEMPLATES_FOLDER.joinpath("old/v7.tmp")
    V8 = PROMPT_TEMPLATES_FOLDER.joinpath("old/v8.tmp")
    V9 = PROMPT_TEMPLATES_FOLDER.joinpath("old/v9.tmp")     # removing <value> as possible output instead of <object> and replacing algorithmically
    V10 = PROMPT_TEMPLATES_FOLDER.joinpath("old/v10.tmp")   # two-shot, output_parsing, {format_instructions}
    V11 = PROMPT_TEMPLATES_FOLDER.joinpath("old/v11.tmp")   # restructuring, handling INFORMATION_LEAK in {format_instructions}
    V12 = PROMPT_TEMPLATES_FOLDER.joinpath("old/v12.tmp")   # linguistic knowledge, zero-shot (no examples), no {format_instructions}
    V13 = PROMPT_TEMPLATES_FOLDER.joinpath("old/v13.tmp")   # 1-shot (1 correct example)
    V17 = PROMPT_TEMPLATES_FOLDER.joinpath("old/v17_json.tmp")   # output as json structure
    V18 = PROMPT_TEMPLATES_FOLDER.joinpath("old/v18_json.tmp")


class ConsistencyTemplateNames(Enum):
    V4 = PROMPT_TEMPLATES_FOLDER.joinpath("hallucination_prompt_v4.tmp")

    # old versions
    V1 = PROMPT_TEMPLATES_FOLDER.joinpath("old/hallucination_prompt.tmp")
    V2 = PROMPT_TEMPLATES_FOLDER.joinpath("old/hallucination_prompt_v2.tmp")
    V3 = PROMPT_TEMPLATES_FOLDER.joinpath("old/hallucination_prompt_v3.tmp")


class TemplateErrors(Enum):
    NA = "<<err-n/a>>"  # error with fetching examples from rdf index (no examples exist in rdf index for given pid)
    API = "<<err-api>>"  # error with langchain (or underlying llm) api
    PARSING = "<<err-parsing-general>>"  # general parsing error
    JSON_PARSING = "<<err-parsing>>"  # error with parsing output from llm
    NO_SUBJECT = "<<err-no-subject>>"  # there must be exactly one subject!
    MULTIPLE_SUBJECTS = "<<err-multiple-subjects>>"  # there must be exactly one subject!
    NO_OBJECT = "<<err-no-object>>"  # there must be exactly one <object> (or <value>)
    MULTIPLE_OBJECTS = "<<err-multiple-objects>>"  # there must be exactly one <object> (or <value>)
    OBJECT_XOR_VALUE = "<<err-obj-xor-val>>"  # there must be one of <object> or <value> but not both at once
    MISPLACED_VALUE = "<<err-misplaced-value>>"  # e.g. <value> where <object> should be)
    ILLEGAL_PLACEHOLDER = "<<err-illegal-placeholder>>"  # e.g. there is a placeholder (<...>) that is not <subject>, <object> or <value>)
    INFORMATION_LEAK = "<<err-information-leak>>"  # specific information from subject or object is present in the template
    ERR = "<<err>>"  # generic error for errors that were not handled above


ERROR_MESSAGES = {
    TemplateErrors.NA: "No examples exist in rdf index for the given pid.",
    TemplateErrors.API: "Error with langchain (or underlying llm) API.",
    TemplateErrors.PARSING: "General parsing error with output from llm.",
    TemplateErrors.JSON_PARSING: "Output is not a valid JSON.",
    TemplateErrors.NO_SUBJECT: "Output is missing <subject> placeholder.",
    TemplateErrors.MULTIPLE_SUBJECTS: "Output has more than one <subject> placeholders. Expected exactly one <subject>.",
    TemplateErrors.NO_OBJECT: "Output is missing <object> placeholder.",
    TemplateErrors.MULTIPLE_OBJECTS: "Output has more than one <object> placeholders. Expected exactly one <object>.",
    TemplateErrors.OBJECT_XOR_VALUE: f"Output has both <object> and {CONSTANT_PLACEHOLDER} placeholders. Expected only one or the other.",
    TemplateErrors.MISPLACED_VALUE: f"Misplaced {CONSTANT_PLACEHOLDER} placeholder. Expected <object> placeholder instead.",
    TemplateErrors.ILLEGAL_PLACEHOLDER: f"Output contains illegal placeholder. Allowed placeholders: [<subject>, <object>].",
    TemplateErrors.INFORMATION_LEAK: "Specific information from subject or object is present in the template.",
    TemplateErrors.ERR: "Output has an unspecified error.",
}


class RDFExampleFormat(Enum):
    DEFAULT = "examples"
    TABLE = "example_table_str"
    JSON = "example_rdf_list"


class ModelChoices(Enum):
    G3 = "davinci"
    G3P5T_0301 = "gpt-3.5-turbo-0301"  # turbo snapshot from March 1st
    G3P5T_0613 = "gpt-3.5-turbo-0613"  # turbo snapshot from June 13th
    G3P5T = "gpt-3.5-turbo"  # the default turbo model as per OpenAI documentation
    G3P5 = "text-davinci-003"
    G4_0314 = "gpt-4-0314"  # GPT-4 snapshot from March 14th
    G4_0613 = "gpt-4-0613"  # GPT-4 snapshot from June 13th
    G4 = "gpt-4"  # the default GPT-4 model as per OpenAI documentation
    FALCON_7B = "tiiuae/falcon-7b-instruct"  # instruct-falcon-llm from Technology Innovation Institute
    FALCON_40B = "tiiuae/falcon-40b-instruct"  # instruct-falcon-llm from Technology Innovation Institute
    NONE = None  # defined for omitting Consistency Validation step


MODEL_CATEGORIES = {"OpenAI": [ModelChoices.G3, ModelChoices.G3P5],
                    "ChatOpenAI": [ModelChoices.G3P5T, ModelChoices.G3P5T_0301, ModelChoices.G3P5T_0613,
                                   ModelChoices.G4, ModelChoices.G4_0314, ModelChoices.G4_0613],
                    "HFacePipeline": [ModelChoices.FALCON_7B, ModelChoices.FALCON_40B]}

sTOTAL_ERRORS = "TOTAL_ERRORS"
sTOTAL_PIDs_w_ERROR = "TOTAL_PIDs_w_ERROR"

LOG_ROOT.mkdir(parents=True, exist_ok=True)
