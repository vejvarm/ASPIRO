#!/bin/bash

setups=("setups/webnlg/gpt3p5/g3p5_0shot.json" "setups/webnlg/gpt3p5/g3p5_0shot_cv.json" "setups/webnlg/gpt3p5/g3p5_1shot.json" "setups/webnlg/gpt3p5/g3p5_2shot.json" "setups/webnlg/gpt3p5/g3p5_3shot.json" "setups/webnlg/gpt3p5/g3p5_4shot.json" "setups/webnlg/gpt3p5/g3p5_5shot.json" "setups/webnlg/gpt3p5/g3p5_5shot_cv.json")
output_dir="/media/freya/kubuntu-data/datasets/d2t/csqa-d2t/EMNLP2023_rebuttal/experiments/g3p5"

for setup in "${setups[@]}"
do
    echo "Running with setup $setup"
    python run_aspiro.py --config "$setup" --output "$output_dir"
    echo "Completed with setup $setup"
done

echo "All tasks completed."