from code_base.model import TransformerMusicLM
from transformers import GenerationConfig
import argparse
import sys
import os

def parseArgs(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, default="exp/submission.ckpt")
    parser.add_argument('--output_dir', type=str)
    parser.add_argument('--generation_number', type=int, default=20)
    parser.add_argument("--do_sample", type=bool, default=False)
    parser.add_argument("--num_beams", type=int, default=1)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=float, default=50)
    parser.add_argument("--top_p", type=float, default=1.0)
    parser.add_argument("--max_length", type=int, default=1024)
    return parser.parse_args(argv)

def main(argv):
    args = parseArgs(argv)
    model = TransformerMusicLM.load_from_checkpoint(args.ckpt)
    model.eval()
    model.generate(generation_num=args.generation_number, output_dir=args.output_dir, generation_config=vars(args))


if __name__ == "__main__":
    main(sys.argv[1:])