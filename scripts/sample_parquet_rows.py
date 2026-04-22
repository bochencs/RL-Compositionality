from argparse import ArgumentParser
import os

from datasets import load_dataset


def main(args):
    dataset = load_dataset("parquet", data_files=args.input)["train"]
    total = len(dataset)
    if args.num_rows > total:
        raise ValueError(f"Requested {args.num_rows} rows from {args.input}, but only {total} rows are available.")

    if args.num_rows == total:
        sampled = dataset
    else:
        sampled = dataset.shuffle(seed=args.seed).select(range(args.num_rows))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    sampled.to_parquet(args.output)
    print(f"Saved {len(sampled)} rows to {args.output}")


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--num_rows", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    main(parser.parse_args())
