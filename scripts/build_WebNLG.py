import random
import pathlib
import json
import requests
import subprocess
import os
import shutil

DEFAULT_FOLDER = "../data/webnlg"
KASNER2022_TEMPLATE_FILE_URL = "https://raw.githubusercontent.com/kasnerz/zeroshot-d2t-pipeline/9ddc978d6caef98fe11153d8ded72e907c65bae5/templates/templates-webnlg.json"


def download_file(url, output_file: pathlib.Path):
    response = requests.get(url, stream=True)
    response.raise_for_status()  # Raise an HTTPError if the HTTP request returned an unsuccessful status code

    with output_file.open("wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)


def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    if process.returncode != 0:
        print(f'Command failed with error: {stderr.decode()}')
    else:
        print(f'\t{stdout.decode()}')


def prepare_webnlg(ver: str, data_folder: pathlib.Path, splits: list[str]):
    repo_name = "WebNLG_Reader"
    for split in splits:
        if data_folder.joinpath(f"{split}.json").exists():
            print(f"Some split.json files already exist in `{data_folder}`.")
            inp = input("overwrite? y/N\n")
            if "y" in inp:
                print("Overwriting old files.")
                break
            else:
                print(f"Exitting script. Please check files in `{data_folder}` manually.")
                exit()

    # Clone the repo
    if not pathlib.Path.cwd().joinpath(repo_name).exists():
        print(f"Cloning {repo_name} repository ...")
        run_command(f'git clone https://github.com/vejvarm/{repo_name}.git')
    os.chdir(f'{repo_name}')

    # Download specified version of the WebNLG data
    print(f"Downloading WebNLG version {ver} ...")
    run_command(f'python data/webnlg/reader.py --version {ver}')

    # Move the generated json files to the desired location
    print(f"Moving ...")
    for split in splits:
        print(f"\t `{split}.json` to `{data_folder}`")
        shutil.move(f'data/webnlg/{split}.json', data_folder.joinpath(f'{split}.json'))

    # Clean up
    print(f"Cleaning up ...")
    os.chdir('..')
    shutil.rmtree(f'{repo_name}')
    print(f"WebNLG processed.")


def parse_examples(example_list: list[tuple["s", "r", "o"]], label: str) -> list[dict["s": str,
                                                                                      "r": str,
                                                                                      "o": str,
                                                                                      "lab": str]]:
    parsed_examples = []
    for s, r, o in example_list:
        label_template = label.replace("<subject>", s).replace("<object>", o)
        parsed_examples.append({"s": s, "r": r, "o": o, "lab": label_template})
    return parsed_examples


def main():
    splits = ["valid", "test", "train"]
    data_folder = pathlib.Path(DEFAULT_FOLDER).resolve()
    webnlg_ver = "1.4"
    max_examples = 5
    random.seed(42)

    path_to_webnlg_data = [data_folder.joinpath(f"{split}.json") for split in splits]
    path_to_zeroshot_d2t_webnlg_manual_templates = data_folder.joinpath("templates-webnlg.json")
    path_to_example_output_json = data_folder.joinpath("rdf_examples_for_each_pid.json")

    # download required files
    if not path_to_zeroshot_d2t_webnlg_manual_templates.exists():
        print(f"downloading {path_to_zeroshot_d2t_webnlg_manual_templates.name}")
        download_file(KASNER2022_TEMPLATE_FILE_URL, path_to_zeroshot_d2t_webnlg_manual_templates)

    # download and prepare webnlg data
    prepare_webnlg(webnlg_ver, data_folder, splits)

    # step 1: aggregate webnlg_data based on rel_label
    webnlg_data_rel_aggregated_dict = dict()
    for pth in path_to_webnlg_data:
        webnlg_data_list = json.load(pth.open())
        for entry in webnlg_data_list:
            triple_list = entry["triples"]
            for triple in triple_list:
                s, r, o = triple
                if r not in webnlg_data_rel_aggregated_dict.keys():
                    webnlg_data_rel_aggregated_dict[r] = {tuple(triple)}
                else:
                    webnlg_data_rel_aggregated_dict[r].add(tuple(triple))

    counts = {rel: len(example_list) for rel, example_list in webnlg_data_rel_aggregated_dict.items()}

    print(min(counts.values()))  # 'chairperson': 1

    # step 2: randomly sample max_examples per rel_label in zeroshot_webnlg_templates
    zeroshot_webnlg_templates = json.load(path_to_zeroshot_d2t_webnlg_manual_templates.open())
    rel_rdf_sample_dict = {}
    for rel_label, rel_template in zeroshot_webnlg_templates.items():
        examples = webnlg_data_rel_aggregated_dict[rel_label]
        template_label = rel_template[0]
        if len(examples) < max_examples:
            rel_rdf_sample_dict[rel_label] = parse_examples(list(examples), template_label)
        else:
            rel_rdf_sample_dict[rel_label] = parse_examples(random.sample(list(examples), max_examples), template_label)

    json.dump(rel_rdf_sample_dict, path_to_example_output_json.open("w", encoding="utf8"), indent=2)
    print("Done!")


if __name__ == "__main__":
    main()
