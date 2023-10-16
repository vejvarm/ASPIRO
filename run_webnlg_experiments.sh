#!/bin/bash

setups=("setups/webnlg/asdot_gpt3p5t_0shot.json" "setups/webnlg/asdot_gpt3p5t_0shot_cv.json" "setups/webnlg/asdot_gpt3p5t_1shot.json" "setups/webnlg/asdot_gpt3p5t_2shot.json" "setups/webnlg/asdot_gpt3p5t_3shot.json" "setups/webnlg/asdot_gpt3p5t_4shot.json" "setups/webnlg/asdot_gpt3p5t_5shot.json" "setups/webnlg/asdot_gpt3p5t_5shot_cv.json" "setups/webnlg/json_gpt3p5t_0shot.json" "setups/webnlg/json_gpt3p5t_0shot_cv.json" "setups/webnlg/json_gpt3p5t_1shot.json" "setups/webnlg/json_gpt3p5t_2shot.json" "setups/webnlg/json_gpt3p5t_3shot.json" "setups/webnlg/json_gpt3p5t_4shot.json" "setups/webnlg/json_gpt3p5t_5shot.json" "setups/webnlg/json_gpt3p5t_5shot_cv.json")
output_dir="/media/freya/kubuntu-data/datasets/d2t/csqa-d2t/EMNLP2023_rebuttal/experiments/"

for setup in "${setups[@]}"
do
    echo "Running with setup $setup"
    python run_aspiro.py --config "$setup" --output "$output_dir"
    echo "Completed with setup $setup"
done

echo "All tasks completed."