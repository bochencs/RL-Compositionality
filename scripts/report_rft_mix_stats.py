from argparse import ArgumentParser
import csv
import os
import re

from datasets import load_dataset


ATOMIC_RE = re.compile(r"^codeio-forward-incomplete-depth(?P<depth>\d+)$")
EXPLICIT_DAG_RE = re.compile(r"^codeio-forward-explicit-dag-branch(?P<num_branch>\d+)-depth(?P<depth>\d+)$")


def parse_data_source(data_source):
    atomic_match = ATOMIC_RE.match(data_source)
    if atomic_match:
        return {
            "category": "atomic",
            "num_branch": "",
            "depth": atomic_match.group("depth"),
            "template": "atomic",
        }

    dag_match = EXPLICIT_DAG_RE.match(data_source)
    if dag_match:
        num_branch = dag_match.group("num_branch")
        depth = dag_match.group("depth")
        return {
            "category": "explicit_dag",
            "num_branch": num_branch,
            "depth": depth,
            "template": f"branch{num_branch}_depth{depth}",
        }

    return {
        "category": "unknown",
        "num_branch": "",
        "depth": "",
        "template": data_source,
    }


def collect_split_rows(split_name, parquet_path):
    dataset = load_dataset("parquet", data_files=parquet_path)["train"]
    total = len(dataset)
    counts = {}
    for row in dataset:
        data_source = row["data_source"]
        counts[data_source] = counts.get(data_source, 0) + 1

    rows = []
    for data_source, count in sorted(counts.items()):
        parsed = parse_data_source(data_source)
        rows.append(
            {
                "split": split_name,
                "data_source": data_source,
                "category": parsed["category"],
                "template": parsed["template"],
                "num_branch": parsed["num_branch"],
                "depth": parsed["depth"],
                "count": count,
                "ratio": f"{count / total:.6f}" if total else "0.000000",
            }
        )

    rows.append(
        {
            "split": split_name,
            "data_source": "__TOTAL__",
            "category": "",
            "template": "",
            "num_branch": "",
            "depth": "",
            "count": total,
            "ratio": "1.000000" if total else "0.000000",
        }
    )
    return rows


def main(args):
    rows = []
    if args.train and os.path.exists(args.train):
        rows.extend(collect_split_rows("train", args.train))
    if args.test and os.path.exists(args.test):
        rows.extend(collect_split_rows("test", args.test))

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "split",
                "data_source",
                "category",
                "template",
                "num_branch",
                "depth",
                "count",
                "ratio",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(args.output)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--train", type=str, default=None)
    parser.add_argument("--test", type=str, default=None)
    parser.add_argument("--output", type=str, required=True)
    main(parser.parse_args())
