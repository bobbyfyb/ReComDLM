"""
Run LLaDA inference on a custom JSONL dataset.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List

import torch
import transformers
from datasets import load_dataset
from tqdm import tqdm

from dllm.core.samplers.mdlm import MDLMSampler, MDLMSamplerConfig
from dllm.pipelines.dream.sampler import DreamSampler, DreamSamplerConfig
from dllm.utils.models import get_model, get_tokenizer
from dllm.utils.sampling import decode_trim
from dllm.utils.utils import resolve_with_base_env
from dllm.utils.visualizers import TerminalVisualizer


@dataclass
class ScriptArguments:
    infer_backend: str = "llada"  # llada or dream
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Instruct"
    adapter_path: str = ""  # LoRA adapter dir (e.g., checkpoint-final)
    split_jsonl_path: str = ""  # If set, use this single file for inference
    train_jsonl_path: str = "data/train.jsonl"
    valid_jsonl_path: str = "data/valid.jsonl"
    test_jsonl_path: str = "data/test.jsonl"
    split: str = "test"
    output_jsonl_path: str = "outputs/llada_predictions.jsonl"
    batch_size: int = 8
    max_samples: int = 0  # 0 = all
    seed: int = 42
    visualize: bool = False
    is_debug: bool = False

    def __post_init__(self):
        self.model_name_or_path = resolve_with_base_env(
            self.model_name_or_path, "BASE_MODELS_DIR"
        )
        if self.adapter_path:
            self.adapter_path = resolve_with_base_env(
                self.adapter_path, "BASE_MODELS_DIR"
            )


@dataclass
class SamplerConfig:
    # Shared
    steps: int = 128
    max_new_tokens: int = 128
    temperature: float = 0.0
    # LLaDA
    block_size: int = 32
    remasking: str = "low_confidence"
    # Dream
    top_p: float = 0.95
    top_k: int = 50
    alg: str = "entropy"
    alg_temp: float = 0.0


def _build_prompt(example: Dict[str, Any]) -> str:
    concept = str(example.get("concept", example.get("src", "")))
    retrieved = example.get("retrieved_passages")
    if isinstance(retrieved, list) and retrieved:
        passages = "\n".join(str(p) for p in retrieved)
        return (
            "Given several concepts together with serveral reference sentences, "
            "write a short and simple sentence that contains *all* the required words. "
            "Only give me the sentence and do not output any other words. "
            "The sentence should describe a common scene in daily life, and the concepts "
            "should be used in a natural way.\n"
            f"concept: {concept}\n"
            f"reference sentences:\n{passages}"
        )
    return (
        "Given a concept, write a short and simple sentence that contains *all* the required words. "
        "Only give me the sentence and do not output any other words. "
        "The sentence should describe a common scene in daily life, and the concepts "
        "should be used in a natural way.\n"
        f"concept: {concept}"
    )


def _load_split(args: ScriptArguments):
    if args.split_jsonl_path:
        raw = load_dataset("json", data_files={"data": args.split_jsonl_path})
        ds = raw["data"]
    else:
        data_files = {
            "train": args.train_jsonl_path,
            "validation": args.valid_jsonl_path,
            "test": args.test_jsonl_path,
        }
        raw = load_dataset("json", data_files=data_files)
        split = args.split
        if split not in raw:
            raise ValueError(f"Split '{split}' not found. Available: {list(raw.keys())}")
        ds = raw[split]
    if args.max_samples and args.max_samples > 0:
        ds = ds.select(range(min(args.max_samples, len(ds))))
    return ds


def _batch_iter(items: List[Dict[str, Any]], batch_size: int):
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def main():
    parser = transformers.HfArgumentParser((ScriptArguments, SamplerConfig))
    args, sampler_config = parser.parse_args_into_dataclasses()
    args.infer_backend = args.infer_backend.lower().strip()
    if args.infer_backend not in {"llada", "dream"}:
        raise ValueError("infer_backend must be either 'llada' or 'dream'.")
    transformers.set_seed(args.seed)

    model = get_model(model_args=args).eval()
    tokenizer = get_tokenizer(model_args=args)

    # If a LoRA adapter is provided, load it on top of the base model.
    if args.adapter_path:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, args.adapter_path).eval()

    if args.infer_backend == "dream":
        sampler = DreamSampler(model=model, tokenizer=tokenizer)
        backend_sampler_config = DreamSamplerConfig(
            steps=sampler_config.steps,
            max_new_tokens=sampler_config.max_new_tokens,
            temperature=sampler_config.temperature,
            top_p=sampler_config.top_p,
            top_k=sampler_config.top_k,
            alg=sampler_config.alg,
            alg_temp=sampler_config.alg_temp,
        )
    else:
        sampler = MDLMSampler(model=model, tokenizer=tokenizer)
        backend_sampler_config = MDLMSamplerConfig(
            steps=sampler_config.steps,
            max_new_tokens=sampler_config.max_new_tokens,
            block_size=sampler_config.block_size,
            temperature=sampler_config.temperature,
            remasking=sampler_config.remasking,
        )

    visualizer = TerminalVisualizer(tokenizer=tokenizer) if args.visualize else None

    ds = _load_split(args)
    items = [ds[i] for i in range(len(ds))]

    os.makedirs(os.path.dirname(args.output_jsonl_path) or ".", exist_ok=True)
    total = len(items)
    with open(args.output_jsonl_path, "w", encoding="utf-8") as f:
        for batch in tqdm(
            _batch_iter(items, args.batch_size),
            total=(total + args.batch_size - 1) // args.batch_size,
            desc="Inferencing",
        ):
            prompts = [_build_prompt(ex) for ex in batch]
            messages = [[{"role": "user", "content": p}] for p in prompts]

            input_ids = tokenizer.apply_chat_template(
                messages, add_generation_prompt=True, tokenize=True
            )

            outputs = sampler.sample(input_ids, backend_sampler_config, return_dict=True)
            texts = decode_trim(tokenizer, outputs.sequences.tolist(), input_ids)

            for ex, prompt, pred in zip(batch, prompts, texts):
                if args.is_debug:
                    print(f"pred: {pred}")
                out = {
                    "prompt": prompt,
                    "prediction": pred,
                }
                if "tgt" in ex:
                    out["reference"] = ex.get("tgt")
                if "retrieved_passages" in ex:
                    out["supported_contexts"] = ex.get("retrieved_passages")
                f.write(json.dumps(out, ensure_ascii=True) + "\n")

            if visualizer is not None:
                visualizer.visualize(outputs.histories, rich=True)

    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.barrier()


if __name__ == "__main__":
    main()
