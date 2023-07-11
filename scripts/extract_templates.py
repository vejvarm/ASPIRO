"""
Extract templates from ASPIRO outputs into a template file, which only
contains the templates without any additional entries.

i.e.
INPUT FORMAT:
{
  "diameter": {
    "output": "The diameter of <subject> is <object>.",
    "error_codes": [],
    "error_messages": [],
    "input_data": "[[\"London Eye eye\", \"diameter\", \"+120 metre\"]]",
    "subjects": [
      "London Eye eye"
    ],
    "objects": [
      "+120 metre"
    ],
    "shot": 0
  },
  ...
}

OUTPUT FORMAT:
{
    "diameter": [
        "The diameter of <subject> is <object>."
    ],
}
...
"""
import json
import argparse


def convert_structure(input_file, output_file):
    with open(input_file, 'r') as infile:
        data = json.load(infile)

    output_data = {}
    for key, value in data.items():
        output_data[key] = [value['output']]

    with open(output_file, 'w') as outfile:
        json.dump(output_data, outfile, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert file structures.")
    parser.add_argument('--input-file', type=str, required=True, help="Input JSON file path (ASPIRO output format).")
    parser.add_argument('--output-file', type=str, required=True, help="Output JSON file path (templates only).")

    args = parser.parse_args()

    convert_structure(args.input_file, args.output_file)
