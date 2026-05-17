#!/usr/bin/env python3
"""Generate controllable string compositional datasets for RL-Compositionality."""

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


NO_PARAM_FUNCS = [
    "deterministic_shuffle",
    "remove_vowels",
    "sort_chars",
    "reverse_words",
    "mirror_str",
    "alternate_case",
    "vowel_to_number",
    "duplicate_every_char",
    "fancy_brackets",
    "compress_repeats",
    "recursive_reverse",
    "loop_filter_nonalpha",
    "verify_even_length",
    "run_length_encode",
    "sort_by_frequency",
    "checksum_rotate",
]

PARAM_FUNCS = [
    "repeat_str",
    "add_prefix",
    "add_suffix",
    "rotate_str",
    "shift_chars",
    "insert_separator",
    "while_rotate",
    "loop_concat",
    "backchain_add_digit",
    "backchain_palindrome",
]

BUILTIN_OPS = ["upper", "lower", "capitalize", "swapcase"]


def parse_args():
    parser = ArgumentParser()
    parser.add_argument("--save_path", required=True)
    parser.add_argument("--stage", type=int, choices=[1, 2], required=True)
    parser.add_argument("--min_level", type=int, required=True)
    parser.add_argument("--max_level", type=int, required=True)
    parser.add_argument("--data_num", type=int, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--allowed_funcs",
        type=str,
        default="",
        help="Comma-separated allowed function names. Empty means all functions in base.all_set.",
    )
    parser.add_argument(
        "--binary_prob",
        type=float,
        default=0.2,
        help="Probability of generating binary composition at each node.",
    )
    parser.add_argument(
        "--allow_constants",
        type=int,
        choices=[0, 1],
        default=0,
        help="Whether random leaf constants are allowed.",
    )
    parser.add_argument(
        "--allow_builtin_methods",
        type=int,
        choices=[0, 1],
        default=0,
        help="Whether built-in string methods (upper/lower/...) are allowed.",
    )
    parser.add_argument("--ability", type=str, default="reasoning")
    parser.add_argument("--split", type=str, default="dummy")
    return parser.parse_args()


def parse_allowed_funcs(raw_value, all_funcs):
    if raw_value.strip() == "":
        return sorted(all_funcs.keys())

    allowed = [item.strip() for item in raw_value.split(",") if item.strip()]
    unknown = sorted(set(allowed) - set(all_funcs.keys()))
    if unknown:
        raise ValueError(f"Unknown functions in --allowed_funcs: {unknown}")
    return allowed


def random_literal():
    return "'" + ''.join(random.choices(string.ascii_lowercase, k=random.randint(3, 6))) + "'"


def build_param_expr(op, sub_expr):
    if op == "repeat_str":
        n = random.randint(2, 4)
        return f"repeat_str({sub_expr}, {n})"
    if op == "add_prefix":
        pre = ''.join(random.choices(string.ascii_lowercase, k=random.randint(2, 4)))
        return f"add_prefix({sub_expr}, '{pre}')"
    if op == "add_suffix":
        suf = ''.join(random.choices(string.ascii_lowercase, k=random.randint(2, 4)))
        return f"add_suffix({sub_expr}, '{suf}')"
    if op == "rotate_str":
        n = random.randint(1, 3)
        return f"rotate_str({sub_expr}, {n})"
    if op == "shift_chars":
        shift_val = random.randint(1, 5)
        return f"shift_chars({sub_expr}, {shift_val})"
    if op == "insert_separator":
        sep = random.choice(['-', '_', '|'])
        return f"insert_separator({sub_expr}, '{sep}')"
    if op == "while_rotate":
        n = random.randint(1, 3)
        return f"while_rotate({sub_expr}, {n})"
    if op == "loop_concat":
        n = random.randint(2, 4)
        return f"loop_concat({sub_expr}, {n})"
    if op == "backchain_add_digit":
        depth = random.randint(1, 3)
        return f"backchain_add_digit({sub_expr}, {depth})"
    if op == "backchain_palindrome":
        depth = random.randint(1, 3)
        return f"backchain_palindrome({sub_expr}, {depth})"
    return sub_expr


def random_expr(depth, custom_functions, binary_prob, allow_constants, allow_builtin_methods):
    if depth == 0:
        if allow_constants and random.random() < 0.5:
            return random_literal()
        return "x"

    binary_candidates = ["plus"]
    if "interlace_str" in custom_functions:
        binary_candidates.append("interlace")
    if "recursive_interlace" in custom_functions:
        binary_candidates.append("recursive_interlace")

    if random.random() < binary_prob and len(binary_candidates) > 0:
        left = random_expr(depth - 1, custom_functions, binary_prob, allow_constants, allow_builtin_methods)
        right = random_expr(depth - 1, custom_functions, binary_prob, allow_constants, allow_builtin_methods)
        op = random.choice(binary_candidates)
        if op == "plus":
            return f"({left} + {right})"
        if op == "interlace":
            return f"interlace_str({left}, {right})"
        return f"recursive_interlace({left}, {right})"

    if allow_builtin_methods and random.random() < 0.05:
        op = random.choice(BUILTIN_OPS)
        sub_expr = random_expr(depth - 1, custom_functions, binary_prob, allow_constants, allow_builtin_methods)
        return f"({sub_expr}).{op}()"

    no_param_ops = [name for name in NO_PARAM_FUNCS if name in custom_functions]
    param_ops = [name for name in PARAM_FUNCS if name in custom_functions]

    if not no_param_ops and not param_ops:
        return "x"

    use_no_param = no_param_ops and (not param_ops or random.random() < 0.5)
    sub_expr = random_expr(depth - 1, custom_functions, binary_prob, allow_constants, allow_builtin_methods)

    if use_no_param:
        op = random.choice(no_param_ops)
        return f"{op}({sub_expr})"

    op = random.choice(param_ops)
    return build_param_expr(op, sub_expr)


def generate_full_code(expr, custom_functions, stage, func_name_mapping):
    used_funcs = []
    for func_name in custom_functions.keys():
        if re.search(rf"\b{re.escape(func_name)}\b", expr):
            used_funcs.append(func_name)

    if stage == 1:
        code_parts = []
        for func_name in sorted(used_funcs):
            code_parts.append(inspect.getsource(custom_functions[func_name]))
        full_code = "\n\n".join(code_parts)
        if full_code:
            full_code += "\n\n"
        full_code += f"def main_solution(x):\n    return {expr}"
    else:
        full_code = f"def main_solution(x):\n    return {expr}"

    for func_name, mapped_name in func_name_mapping.items():
        full_code = re.sub(rf"\b{re.escape(func_name)}\b", mapped_name, full_code)

    return full_code


def generate_feasible_input(func, attempts=200, min_len=3, max_len=10):
    for _ in range(attempts):
        length = random.randint(min_len, max_len)
        candidate = ''.join(random.choices(string.ascii_lowercase, k=length))
        try:
            result = func(candidate)
            if isinstance(result, str):
                return candidate
        except Exception:
            continue
    return None


def generate_one_sample(depth, custom_functions, stage, args):
    expr = random_expr(
        depth=depth,
        custom_functions=custom_functions,
        binary_prob=args.binary_prob,
        allow_constants=bool(args.allow_constants),
        allow_builtin_methods=bool(args.allow_builtin_methods),
    )

    eval_globals = dict(custom_functions)
    eval_globals["__builtins__"] = {}
    func = eval(f"lambda x: {expr}", eval_globals)
    input_x = generate_feasible_input(func)
    if input_x is None:
        return None

    output = func(input_x)
    if not isinstance(output, str):
        return None

    full_code = generate_full_code(
        expr=expr,
        custom_functions=custom_functions,
        stage=stage,
        func_name_mapping=base.func_name_mapping,
    )

    sample = {
        "data_source": f"codeio-forward-incomplete-depth{depth}",
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
            "split": args.split,
            "depth": depth,
            "stage": stage,
        },
    }
    return expr, sample


def main():
    args = parse_args()

    if args.min_level > args.max_level:
        raise ValueError("--min_level must be <= --max_level")
    if not (0.0 <= args.binary_prob <= 1.0):
        raise ValueError("--binary_prob must be in [0, 1]")

    random.seed(args.seed)

    allowed_func_names = parse_allowed_funcs(args.allowed_funcs, base.all_set)
    custom_functions = {name: base.all_set[name] for name in allowed_func_names}

    depths = list(range(args.min_level, args.max_level + 1))
    if args.data_num % len(depths) != 0:
        raise ValueError(
            f"--data_num {args.data_num} should be divisible by number of depths {len(depths)}"
        )

    num_per_depth = args.data_num // len(depths)
    all_samples = []
    generated_exprs = set()

    for depth in depths:
        target = num_per_depth
        count = 0
        attempts = 0
        max_attempts = max(target * 200, 2000)
        print(f"Generating depth={depth}, target={target}")

        while count < target and attempts < max_attempts:
            attempts += 1
            try:
                result = generate_one_sample(depth, custom_functions, args.stage, args)
            except Exception:
                continue

            if result is None:
                continue

            _, sample = result
            dedup_key = (depth, sample["prompt"][0]["content"])
            if dedup_key in generated_exprs:
                continue
            generated_exprs.add(dedup_key)

            sample["extra_info"]["index"] = count
            all_samples.append(sample)
            count += 1

        if count < target:
            raise RuntimeError(
                f"Depth {depth}: generated {count}/{target} samples after {attempts} attempts"
            )

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)
    dataset = Dataset.from_list(all_samples)
    dataset.to_parquet(args.save_path)

    print(f"Saved {len(dataset)} samples to {args.save_path}")
    print(f"Allowed functions: {sorted(custom_functions.keys())}")
    print(f"Depth range: {args.min_level}-{args.max_level}")


if __name__ == "__main__":
    main()
