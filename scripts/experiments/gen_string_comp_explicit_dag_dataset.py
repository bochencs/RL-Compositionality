#!/usr/bin/env python3
"""Generate compositional data with explicit intermediate-state DAG programs."""

from argparse import ArgumentParser
from datasets import Dataset
import inspect
import json
import os
from pathlib import Path
import random
import re
import string
import sys

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from examples.data_preprocess import string_data as base


# Keep operators stable to avoid uncontrolled output explosion.
SAFE_NO_PARAM_OPS = [
    "deterministic_shuffle",
    "remove_vowels",
    "sort_chars",
    "reverse_words",
    "mirror_str",
    "alternate_case",
    "vowel_to_number",
    "compress_repeats",
    "recursive_reverse",
    "loop_filter_nonalpha",
    "verify_even_length",
    "run_length_encode",
    "sort_by_frequency",
    "checksum_rotate",
]

SAFE_PARAM_OPS = [
    "add_prefix",
    "add_suffix",
    "rotate_str",
    "shift_chars",
    "insert_separator",
    "while_rotate",
]

# Binary merge operators are reserved for schema structure, not unary atomics.
BINARY_ONLY_OPS = {"interlace_str", "recursive_interlace"}


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--stage", type=int, choices=[1, 2], required=True)
    parser.add_argument("--dataset_split", type=str, choices=["train", "test"], required=True)
    parser.add_argument("--num_branch", type=int, required=True, help="Number of branches expanded from the shared ancestor.")
    parser.add_argument("--depth", type=int, required=True, help="Unary-chain length for each branch.")
    parser.add_argument("--data_num", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ability", type=str, default="reasoning")
    parser.add_argument("--max_output_len", type=int, default=256)
    return parser.parse_args()


def sample_literal(k_low=2, k_high=4):
    return "".join(random.choices(string.ascii_lowercase, k=random.randint(k_low, k_high)))


def select_custom_functions(stage, dataset_split):
    if stage == 1:
        return base.all_set
    if dataset_split == "train":
        return base.train_set
    return base.eval_set


def available_ops(custom_functions):
    no_param = [op for op in SAFE_NO_PARAM_OPS if op in custom_functions and op not in BINARY_ONLY_OPS]
    param = [op for op in SAFE_PARAM_OPS if op in custom_functions and op not in BINARY_ONLY_OPS]
    if not no_param and not param:
        raise ValueError("No available unary operators after filtering.")
    return no_param, param


def build_unary_call(input_expr, op, used_funcs):
    used_funcs.add(op)

    if op == "add_prefix":
        return f"add_prefix({input_expr}, '{sample_literal()}')"
    if op == "add_suffix":
        return f"add_suffix({input_expr}, '{sample_literal()}')"
    if op == "rotate_str":
        return f"rotate_str({input_expr}, {random.randint(1, 3)})"
    if op == "shift_chars":
        return f"shift_chars({input_expr}, {random.randint(1, 5)})"
    if op == "insert_separator":
        return f"insert_separator({input_expr}, '{random.choice(['-', '_', '|'])}')"
    if op == "while_rotate":
        return f"while_rotate({input_expr}, {random.randint(1, 3)})"

    return f"{op}({input_expr})"


def sample_unary_expr(input_expr, no_param_ops, param_ops, used_funcs):
    if no_param_ops and param_ops:
        use_no_param = random.random() < 0.5
    else:
        use_no_param = bool(no_param_ops)
    op = random.choice(no_param_ops if use_no_param else param_ops)
    return build_unary_call(input_expr, op, used_funcs)


def build_explicit_dag_program(num_branch, depth, no_param_ops, param_ops):
    # Explicit-state DAG template with shared ancestor:
    # t0 = U(x)
    # branch_0 from t0
    # branch_1 from t0
    # ...
    # branch_(num_branch-1) from t0
    # merge all branch leaves in a single + expression when num_branch > 1
    # t_out = U(t_merge)
    used_funcs = set()
    lines = []
    t_idx = 0

    shared_expr = sample_unary_expr("x", no_param_ops, param_ops, used_funcs)
    shared = f"t{t_idx}"
    lines.append(f"    {shared} = {shared_expr}")
    t_idx += 1

    branch_leaves = []
    for _ in range(num_branch):
        branch_prev = shared
        for _ in range(depth):
            expr = sample_unary_expr(branch_prev, no_param_ops, param_ops, used_funcs)
            branch_prev = f"t{t_idx}"
            lines.append(f"    {branch_prev} = {expr}")
            t_idx += 1
        branch_leaves.append(branch_prev)

    if num_branch == 1:
        merged = branch_leaves[0]
        num_merge_nodes = 0
    else:
        merged = f"t{t_idx}"
        merged_expr = " + ".join(branch_leaves)
        lines.append(f"    {merged} = ({merged_expr})")
        t_idx += 1
        num_merge_nodes = 1

    out = f"t{t_idx}"
    out_expr = sample_unary_expr(merged, no_param_ops, param_ops, used_funcs)
    lines.append(f"    {out} = {out_expr}")

    return lines, out, used_funcs, {
        "num_unary_calls": 1 + num_branch * depth + 1,
        "num_merge_nodes": num_merge_nodes,
        "num_program_lines": len(lines),
        "structure_schema": "dag_explicit",
        "num_branch": num_branch,
        "branch_depth": depth,
    }


def add_stage1_function_defs(main_code, custom_functions, used_funcs):
    if not used_funcs:
        return main_code
    defs = []
    for name in sorted(used_funcs):
        defs.append(inspect.getsource(custom_functions[name]))
    return "\n\n".join(defs) + "\n\n" + main_code


def rename_to_func_ids(code):
    out = code
    for func_name, mapped_name in base.func_name_mapping.items():
        out = re.sub(rf"\b{re.escape(func_name)}\b", mapped_name, out)
    return out


def build_program_code(lines, out_var):
    body = "\n".join(lines + [f"    return {out_var}"])
    return f"def main_solution(x):\n{body}"


def build_callable(program_code, custom_functions):
    namespace = dict(custom_functions)
    namespace["__builtins__"] = {}
    exec(program_code, namespace)
    return namespace["main_solution"]


def generate_feasible_input(func, max_output_len, attempts=500, min_len=3, max_len=10):
    for _ in range(attempts):
        candidate = "".join(random.choices(string.ascii_lowercase, k=random.randint(min_len, max_len)))
        try:
            result = func(candidate)
        except Exception:
            continue
        if isinstance(result, str) and len(result) <= max_output_len:
            return candidate, result
    return None, None


def make_one_sample(args, custom_functions, no_param_ops, param_ops):
    lines, out_var, used_funcs, stats = build_explicit_dag_program(
        args.num_branch, args.depth, no_param_ops, param_ops
    )
    program_code = build_program_code(lines, out_var)
    func = build_callable(program_code, custom_functions)
    input_x, output = generate_feasible_input(func, max_output_len=args.max_output_len)
    if input_x is None:
        return None

    if args.stage == 1:
        full_code = add_stage1_function_defs(program_code, custom_functions, used_funcs)
    else:
        full_code = program_code

    full_code = rename_to_func_ids(full_code)

    sample = {
        "data_source": f"codeio-forward-explicit-dag-branch{args.num_branch}-depth{args.depth}",
        "prompt": [{
            "role": "user",
            "content": base.FORWARD_PROMPT.format(code=full_code, input=input_x),
        }],
        "ability": args.ability,
        "reward_model": {
            "style": "rule",
            "ground_truth": json.dumps(
                {
                    "ref_input": {"x": input_x},
                    "ref_output": output,
                    "ref_code": full_code,
                    "funcname": "main_solution",
                },
                ensure_ascii=True,
            ),
        },
        "extra_info": {
            "index": 0,
            "split": args.dataset_split,
            "dataset_split": args.dataset_split,
            "num_branch": args.num_branch,
            "depth": args.depth,
            "stage": args.stage,
            "schema": "dag",
            **stats,
            "max_output_len": args.max_output_len,
        },
    }
    return sample


def main():
    args = parse_args()
    if args.num_branch < 1:
        raise ValueError("--num_branch must be >= 1 for explicit DAG generation.")
    if args.depth < 1:
        raise ValueError("--depth must be >= 1.")

    random.seed(args.seed)

    custom_functions = select_custom_functions(args.stage, args.dataset_split)
    no_param_ops, param_ops = available_ops(custom_functions)

    all_samples = []
    dedup = set()
    target = args.data_num
    count = 0
    attempts = 0
    max_attempts = max(target * 300, 3000)
    print(f"Generating explicit DAG: num_branch={args.num_branch}, depth={args.depth}, target={target}")

    while count < target and attempts < max_attempts:
        attempts += 1
        sample = make_one_sample(args, custom_functions, no_param_ops, param_ops)
        if sample is None:
            continue

        dedup_key = sample["prompt"][0]["content"]
        if dedup_key in dedup:
            continue
        dedup.add(dedup_key)

        sample["extra_info"]["index"] = count
        all_samples.append(sample)
        count += 1

    if count < target:
        raise RuntimeError(
            f"num_branch={args.num_branch}, depth={args.depth}: generated {count}/{target} samples after {attempts} attempts"
        )

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    dataset = Dataset.from_list(all_samples)
    dataset.to_parquet(args.save_path)

    print(f"Saved {len(dataset)} samples to {args.save_path}")
    print(f"schema=dag, num_branch={args.num_branch}, depth={args.depth}, dataset_split={args.dataset_split}")


if __name__ == "__main__":
    main()
