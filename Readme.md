# Requirements:
## OpenAI models only:
``` bash
pip install -r requirements.txt
```

## Local addon:
If you want to run local models (such as Falcon-llm-7B), run the following as well:
```bash
pip install -r requirements-local.txt
```

# Execution:
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


### Run from Data of previous run
You can replace the 0shot (0th model in `llmstack`) model in config file with `path/to/previous/result/json`.
It will act as if it was a normal LLM from the stack. This is useful when you already ran the 0shot model and now you want to test how 1shot or Consistency Validation changes it.

See `setups/json_from_data.json` config for example:
```
python run_aspiro --config setups/json_from_data.json
```


# Building dataset scripts
In these source files the data is already built in the `data` folder.

But in case you want to build the data again, you can use the `build_` scripts in the `scripts` folder.

## Build REL2TEXT:
``` bash
python scripts/build_example_json_REL2TEXT.py --input-folder sources/rel2text --output-folder sources/rel2text/data
```

If you want to use the generated data the pipeline, remove the '(split|split|...)' prefix from the generated `(splits)rdf_examples_for_each_pid.json` file and place it into `data/rel2text` folder.


## Build DART:
```bash
python scripts/build_example_json_DART.py --input-folder sources/dart --output-folder sources/dart/data
```

If you want to use the generated data the pipeline, move [dart2rlabel_map.json](sources%2Fdart%2Fdata%2Fdart2rlabel_map.json), [rdf_examples_for_each_pid.json](sources%2Fdart%2Fdata%2Frdf_examples_for_each_pid.json)
and [rlabel2dart_map.json](sources%2Fdart%2Fdata%2Frlabel2dart_map.json) to `data/dart` folder.