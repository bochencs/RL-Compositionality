from argparse import ArgumentParser
from datasets import Dataset, load_dataset

def main(args):
    rows = []
    for path in args.data:
        dataset = load_dataset('parquet', data_files=path)['train']
        rows.extend(dataset.to_list())

    merged = Dataset.from_list(rows)
    merged.to_parquet(args.output_path)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--data', type=str, required=True, nargs='+')
    parser.add_argument('--output-path', type=str, required=True)
    args = parser.parse_args()

    main(args)
