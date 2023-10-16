#!/bin/bash

setups=("setups/rel2text/falcon/json_0shot.json" "setups/rel2text/falcon/json_0shot_cv.json" "setups/rel2text/falcon/json_1shot.json" "setups/rel2text/falcon/json_1shot_gpt3p5t.json" "setups/rel2text/falcon/json_5shot.json")

output_dir="/media/freya/kubuntu-data/datasets/d2t/csqa-d2t/aspiro_test/"

for setup in "${setups[@]}"
do
    echo "Running with setup $setup"
    python run_aspiro.py --config "$setup" --output "$output_dir"
    echo "Completed with setup $setup"
done

echo "All tasks completed."
