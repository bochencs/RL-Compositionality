# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Generate responses given a dataset of prompts
"""

import csv
import ray
import numpy as np
import hydra
import os
import json
from datetime import datetime
from tabulate import tabulate
from datasets import load_dataset

os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ['TORCH_COMPILE_DISABLE'] = '1'

from verl.utils.model import compute_position_id_with_mask

import pandas as pd

from transformers import AutoTokenizer

from verl import DataProto
from verl.utils.fs import copy_to_local
from verl.workers.fsdp_workers import ActorRolloutRefWorker
from verl.utils.hdfs_io import makedirs
from verl.single_controller.ray import RayClassWithInitArgs, RayResourcePool, RayWorkerGroup
from verl.utils.reward_score import _default_compute_score


@hydra.main(config_path="config", config_name="generation", version_base=None)
def main(config):
    run_generation(config)


def run_generation(config) -> None:
    if not ray.is_initialized():
        # this is for local ray cluster
        ray.init(runtime_env={"env_vars": {"TOKENIZERS_PARALLELISM": "true", "NCCL_DEBUG": "WARN"}})

    ray.get(main_task.remote(config))


@ray.remote(num_cpus=1)
def main_task(config):
    from pprint import pprint
    from omegaconf import OmegaConf
    from verl.utils.import_utils import import_external_libs

    resolved_config = OmegaConf.to_container(config, resolve=True)
    pprint(resolved_config)  # resolve=True will eval symbol values
    OmegaConf.resolve(config)

    wandb_run = None
    wandb_module = None
    if os.environ.get("WANDB_ENABLE", "0") == "1":
        try:
            import wandb as wandb_module  # type: ignore

            project = os.environ.get("WANDB_PROJECT", "rlcomp-inference")
            run_name = os.environ.get("WANDB_RUN_NAME", os.path.basename(str(config.data.output_path)))
            init_kwargs = {
                "project": project,
                "name": run_name,
                "config": resolved_config,
                "reinit": True,
            }
            entity = os.environ.get("WANDB_ENTITY")
            group = os.environ.get("WANDB_GROUP")
            tags = [tag.strip() for tag in os.environ.get("WANDB_TAGS", "").split(",") if tag.strip()]
            notes = os.environ.get("WANDB_NOTES")
            if entity:
                init_kwargs["entity"] = entity
            if group:
                init_kwargs["group"] = group
            if len(tags) > 0:
                init_kwargs["tags"] = tags
            if notes:
                init_kwargs["notes"] = notes

            wandb_run = wandb_module.init(**init_kwargs)
            wandb_run.log({
                "meta/run_start_time": datetime.utcnow().isoformat(),
                "meta/model_path": config.model.path,
                "meta/data_path": config.data.path,
                "meta/n_samples": int(config.data.n_samples),
                "meta/temperature": float(config.rollout.temperature),
            })
        except Exception as e:
            print(f"[WANDB] init failed, continue without wandb: {e}")
            wandb_run = None
            wandb_module = None

    if os.path.exists(config.data.output_path) and not config.data.overwrite:
        print(f"Output file {config.data.output_path} already exists. Skipping generation and proceeding to evaluation.")
        # dataset = pd.read_parquet(config.data.output_path)
        dataset = load_dataset("parquet", data_files=config.data.output_path)['train']
    else:
        local_path = copy_to_local(config.model.path)
        import_external_libs(config.model.get("external_lib", None))
        from verl.utils import hf_tokenizer

        tokenizer = hf_tokenizer(local_path)

        if config.rollout.temperature == 0.0:
            assert config.data.n_samples == 1, "When temperature=0, n_samples must be 1."

        # read dataset. Note that the dataset should directly contain chat template format (e.g., a list of dictionary)
        # dataset = pd.read_parquet(config.data.path)
        dataset = load_dataset("parquet", data_files=config.data.path)['train']
        chat_lst = dataset[config.data.prompt_key]

        chat_lst = [chat for chat in chat_lst]

        tokenizer.padding_side = "left"
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        ray_cls_with_init = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=config, role="rollout")
        resource_pool = RayResourcePool(process_on_nodes=[config.trainer.n_gpus_per_node] * config.trainer.nnodes)
        wg = RayWorkerGroup(resource_pool=resource_pool, ray_cls_with_init=ray_cls_with_init)
        wg.init_model()

        total_samples = len(dataset)
        # real_batch_size = data.batch['input_ids'].shape[0]
        config_batch_size = config.data.batch_size
        dispatch_dp_size = wg.world_size
        num_batch = -(-total_samples // config_batch_size)
        output_lst = [[] for _ in range(config.data.n_samples)]

        for batch_idx in range(num_batch):
            print(f"[{batch_idx + 1}/{num_batch}] Start to process.")
            batch_chat_lst = chat_lst[batch_idx * config_batch_size : (batch_idx + 1) * config_batch_size]
            if config.data.zero:
                from verl.utils.dataset.rl_dataset import ZERO_TEMPLATE

                inputs = tokenizer(
                    [ZERO_TEMPLATE.format(prompt=chat) for chat in batch_chat_lst],
                    padding=True,
                    truncation=True,
                    max_length=config.rollout.prompt_length,
                    return_tensors="pt",
                )
            else:
                inputs = tokenizer.apply_chat_template(
                    batch_chat_lst,
                    add_generation_prompt=True,
                    padding=True,
                    truncation=True,
                    max_length=config.rollout.prompt_length,
                    return_tensors="pt",
                    return_dict=True,
                    tokenize=True,
                )
            input_ids = inputs['input_ids']
            attention_mask = inputs['attention_mask']
            position_ids = compute_position_id_with_mask(attention_mask)

            batch_dict = {'input_ids': input_ids, 'attention_mask': attention_mask, 'position_ids': position_ids}

            data = DataProto.from_dict(batch_dict)
            real_batch_size = data.batch['input_ids'].shape[0]
            if real_batch_size % dispatch_dp_size != 0:
                dummy_data_size = dispatch_dp_size - real_batch_size % dispatch_dp_size
                if dummy_data_size <= real_batch_size:
                    dummy_data = data[:dummy_data_size]
                else:
                    dummy_data = data.repeat(-(-dummy_data_size // real_batch_size))[:dummy_data_size]
                data = DataProto.concat([data, dummy_data])
                print(
                    f'real_batch_size {real_batch_size} is not divisible by dispatch_dp_size {dispatch_dp_size}, add {dummy_data_size} dummy data'
                )

            batch_size = data.batch['input_ids'].shape[0]
            assert batch_size % dispatch_dp_size == 0, f'batch_size {batch_size} is not divisible by dispatch_dp_size {dispatch_dp_size}'

            print(f'[{batch_idx+1}/{num_batch}] Start to generate.')
            # START TO GENERATE FOR n_samples TIMES
            for i in range(config.data.n_samples):
                print(f"Generating {i+1}/{config.data.n_samples}")
                output = wg.generate_sequences(data)
                # remove dummy data
                output = output[:real_batch_size]
                output_text = tokenizer.batch_decode(output.batch['input_ids'][:, -config.rollout.response_length:],
                                                    skip_special_tokens=False)

                # remove the padding
                pad_token = tokenizer.pad_token
                output_text_unpad = []
                for text in output_text:
                    output_text_unpad.append(text.replace(pad_token, ''))

                output_dir = os.path.dirname(config.data.output_path)
                makedirs(output_dir, exist_ok=True)
                with open(os.path.join(output_dir, f'gen_{i}.json'), 'w') as f:
                    json.dump(output_text_unpad, f)

                output_lst[i].extend(output_text_unpad)
                if wandb_run is not None:
                    try:
                        wandb_run.log(
                            {
                                "progress/generation_sample_index": int(i + 1),
                                "progress/generation_sample_total": int(config.data.n_samples),
                                "progress/generation_sample_ratio": float(i + 1) / float(config.data.n_samples),
                            },
                            step=int(i + 1),
                        )
                    except Exception as e:
                        print(f"[WANDB] progress log failed: {e}")

        # convert output_lst from (n_samples, n_data) to (n_data, n_sampels)
        output_lst = np.array(output_lst, dtype=object)
        output_lst = np.transpose(output_lst, axes=(1, 0)).tolist()

        # add to the data frame
        # dataset[f'responses'] = output_lst
        dataset = dataset.add_column("responses", output_lst)

        # write to a new parquet
        output_dir = os.path.dirname(config.data.output_path)
        makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "gen_all.json"), 'w') as f:
            json.dump(output_lst, f)
        dataset.to_parquet(config.data.output_path)

    output_dir = os.path.dirname(config.data.output_path)
    # Compute evaluation metrics
    responses = dataset['responses']  # Using the generated responses
    data_sources = dataset[config.data.data_source_key]
    reward_model_data = dataset[config.data.reward_model_key]

    def _normalize_score(score):
        # Some reward fns return tuple/list, e.g., (reward, acc).
        # Prefer the explicit accuracy slot when available.
        if isinstance(score, (tuple, list)):
            if len(score) >= 2 and isinstance(score[1], (int, float, bool)):
                return float(score[1])
            if len(score) >= 1 and isinstance(score[0], (int, float, bool)):
                return float(score[0])
            return 0.0
        if isinstance(score, (int, float, bool)):
            return float(score)
        return 0.0

    passes = 0
    total = len(dataset)
    total_scores = []
    
    for i in range(total):
        response_lst = responses[i]
        if isinstance(response_lst, np.ndarray):
            response_lst = response_lst.tolist()
        elif isinstance(response_lst, tuple):
            response_lst = list(response_lst)
        elif not isinstance(response_lst, list):
            # Keep non-list case (including n_samples=1) consistent with n_samples>1.
            response_lst = [response_lst]
        data_source = data_sources[i]
        reward_data = reward_model_data[i]
        ground_truth = reward_data['ground_truth']
        score_lst = []
        for r in response_lst:
            score = _normalize_score(_default_compute_score(data_source, r, ground_truth))
            score_lst.append(score)
        max_score = np.max(score_lst) if len(score_lst) > 0 else 0.0
        total_scores.append(score_lst)
        if max_score >= 1.0:
            passes += 1

    n_samples = config.data.n_samples
    pass_at_n = passes / total
    pass_at_1 = np.mean([scores[0] if len(scores) > 0 else 0.0 for scores in total_scores])

    # Save metrics to CSV
    csv_path = os.path.join(output_dir, 'pass.csv')
    
    # Prepare the row data
    # Extract the dataset name from the path
    dataset_name = os.path.basename(config.data.path)
    row_data = {
        'model_path': config.model.path,
        'dataset': dataset_name,
        'pass@1': pass_at_1,
        f'pass@{n_samples}': pass_at_n
    }

    # Check if file exists
    file_exists = os.path.isfile(csv_path)
    
    # Write to CSV
    with open(csv_path, mode='a', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=row_data.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row_data)

    # Convert the row data into a list of lists format for tabulate
    table_data = [[k, v] for k, v in row_data.items()]
    
    # Print table
    print(tabulate(table_data, headers=['Metric', 'Value'], tablefmt='grid'))

    if wandb_run is not None and wandb_module is not None:
        try:
            wandb_run.log({
                "metrics/pass_at_1": float(pass_at_1),
                f"metrics/pass_at_{n_samples}": float(pass_at_n),
                "metrics/num_prompts": int(total),
                "meta/output_path": config.data.output_path,
                "meta/dataset_name": dataset_name,
            })
            if os.path.exists(config.data.output_path):
                artifact_name = f"gen_{os.path.basename(config.data.output_path).replace('.', '_')}_{n_samples}"
                artifact = wandb_module.Artifact(artifact_name, type="inference-results")
                artifact.add_file(config.data.output_path)
                if os.path.exists(csv_path):
                    artifact.add_file(csv_path)
                wandb_run.log_artifact(artifact)
        except Exception as e:
            print(f"[WANDB] finalize log failed: {e}")
        finally:
            try:
                wandb_run.finish()
            except Exception:
                pass

# Add the select_reward_fn from main_eval.py
# def select_reward_fn(data_source):
#     if data_source == 'lighteval/MATH':
#         from verl.utils.reward_score import math
#         return math.compute_score
#     else:
#         from deepscaler.rewards.math_reward import deepscaler_reward_fn
#         return deepscaler_reward_fn


if __name__ == "__main__":
    main()
