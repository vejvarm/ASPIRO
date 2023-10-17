ASPIRO: Parse RDF triples to natural language templates
---

Robust data-to-text generation from RDF triples using multi-shot reprompting of Large language models.

![Full pipeline for ASPIRO on general data input.](images/ASPIRO.webp?raw=true "ASPIRO pipeline")

**Influenced by**: \
https://github.com/kasnerz/zeroshot-d2t-pipeline/tree/main \
https://github.com/szxiangjn/any-shot-data2text

This repository contains code for reproducing the paper results accepted to the _Findings of the ACL: EMNLP2023_.

    Martin Vejvar & Yasutaka Fujimoto: ASPIRO: Any-shot Structured Parsing-error-Induced ReprOmpting for Consistent Data-to-Text Generation. In: Findings of the Association for Computational Linguistics: EMNLP 2023.

Link for the paper: (todo)

# Requirements
To run ASPIRO with OpenAI models only:
``` bash
pip install -r requirements.txt
```

Run the following as well if you want to run local models (such as Falcon-llm-7B):
```bash
pip install -r requirements-local.txt
```

# Execution
## 1) API key
make sure you have your API key in the system path `OPENAI_API_KEY` variable.
On Linux, you can run the following from console:
```
export OPENAI_API_KEY="your-private-api-key"
```
To generate new API key, visit [OpenAI API keys](https://platform.openai.com/account/api-keys) page.


## 2) Check Config
You will need to provide path to config when running ASPIRO. 
Some default configs are in the `setups` folder

## 3) Run ASPIRO
### Run from scratch:
``` bash
python run_aspiro.py --config path/to/config
```
NOTE: if no `--config` argument is provided `setups/json_default.json` is chosen as default 

#### example:
```
python run_aspiro.py --config setups/json_default.json --output "outputs/d2t/aspiro_test/"
```

### Run from Data of previous run
You can replace the 0shot (0th model in `llmstack`) model in config file with `path/to/previous/result/json`.
It will act as if it was a normal LLM from the stack. This is useful when you already ran the 0shot model and now you want to test how 1shot or Consistency Validation changes it.

See `setups/json_from_data.json` config for example:
```
python run_aspiro --config setups/json_from_data.json
```


# Preparing datasets
In these source files the data is already built and ready in the `data` folder.

In case you want to rebuild the data, you can use the `build_[dataset-name].py` scripts in the `scripts` folder.

## Build REL2TEXT:
### 1) generate data
``` bash
python scripts/build_REL2TEXT.py --input-folder sources/rel2text --output-folder sources/rel2text/data
```
### 2) move generated data
To use generated data in ASPIRO, rename the generated file in `--output-folder` to simply `rdf_examples_for_each_pid.json` (remove the leading `'(split|split|...)'`) file and place it into `data/rel2text` folder.


## Build DART:
### 1) generate data
```bash
python scripts/build_DART.py --input-folder sources/dart --output-folder sources/dart/data
```
### 2) move generated data
If you want to use the generated data the pipeline, move [dart2rlabel_map.json](sources%2Fdart%2Fdata%2Fdart2rlabel_map.json), [rdf_examples_for_each_pid.json](sources%2Fdart%2Fdata%2Frdf_examples_for_each_pid.json)
and [rlabel2dart_map.json](sources%2Fdart%2Fdata%2Frlabel2dart_map.json) to `data/dart` folder.
## Build WebNLG:
``` bash
cd scripts
python build_WebNLG.py
```