import os
import sys
import argparse
from code_base.inference.predict import predict_singer_csv

def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset_root", type=str, help="Path to your directory containing your json file of your data")
    parser.add_argument("--ckpt", type=str, help="Path of the model checkpoints")
    parser.add_argument("--output_csv", type=str, help="Path of the predicting csv")
    parser.add_argument("--json_prefix", type=str, default="inference", help="Prefix of your json file")
    parser.add_argument("--batch_size", type=int, default=6)

    return parser.parse_args(argv)


def main(argv):
    args = parseArgs(argv)
    predict_singer_csv(
        **vars(args)
    )

if __name__ == "__main__":
    main(sys.argv[1:])