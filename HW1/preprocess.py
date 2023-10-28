import os
import sys
import argparse
import json
from pathlib import Path
from code_base.inference.preprocess_func import source_separate_and_remove_silence_and_segment

def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--target_dir", type=str, help="Dircetory that stores your input wavs")
    parser.add_argument("--output_dir", type=str, help="Dircetory that stores your output wavs")
    parser.add_argument("--json_prefix", type=str, default="inference", help="Prefix of your .json file in output_dif")

    return parser.parse_args(argv)

def main(argv):
    args = parseArgs(argv)
    source_separate_and_remove_silence_and_segment(
        args.target_dir,
        args.output_dir
    )

    # create json file of your data in output_dir
    output_path_list = [ str(f) for f in Path(args.output_dir).glob("**/*.wav") ]
    data_list = [ {"path": p, "song_id": int(p.split("/")[-2])} for p in output_path_list ]
    with open(os.path.join(args.output_dir, f"{args.json_prefix}.json"), "w") as f:
        json.dump(data_list, f, indent=4)


if __name__ == "__main__":
    main(sys.argv[1:])