#!/bin/bash

setups=("setups/wikidata/json_gpt3p5_0shot.json" "setups/wikidata/json_gpt3p5_1shot_3p5.json" "setups/wikidata/json_gpt3p5_2shot_3p5-gpt4.json" "setups/wikidata/json_gpt3p5_5shot.json" "setups/wikidata/json_gpt3p5_5shot_cv.json" "setups/wikidata/json_gpt4_0shot.json")

output_dir="/media/freya/kubuntu-data/datasets/d2t/csqa-d2t/aspiro_test/"

for setup in "${setups[@]}"
do
    echo "Running with setup $setup"
    python run_aspiro.py --config "$setup" --output "$output_dir"
    echo "Completed with setup $setup"
done

echo "All tasks completed."
