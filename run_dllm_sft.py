"""
End-to-end SFT runner for LLaDA using dllm:
- load JSONL
- build prompts and messages
- save to disk
- train via MDLMTrainer
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from functools import partial
from typing import Any, Dict

import accelerate
import transformers
from datasets import DatasetDict, load_dataset

from dllm.core.trainers.mdlm import MDLMTrainer
from dllm.data.utils import load_sft_dataset
from dllm.utils.collators import NoAttentionMaskWrapper
from dllm.utils.configs import DataArguments as BaseDataArguments
from dllm.utils.configs import ModelArguments as BaseModelArguments
from dllm.utils.data import default_sft_map_fn, post_process_dataset
from dllm.utils.models import get_model, get_tokenizer
from dllm.utils.utils import get_default_logger, initial_training_setup, print_args_main

logger = get_default_logger(__name__)


@dataclass
class ModelArguments(BaseModelArguments):
    model_name_or_path: str = "GSAI-ML/LLaDA-8B-Instruct"
    load_in_4bit: bool = True
    lora: bool = True
    r: int = 16
    lora_alpha: int = 32


@dataclass
class DataArguments(BaseDataArguments):
    sft_backend: str = field(
        default="llada",
        metadata={"help": "Which SFT backend to use: 'llada' or 'dream'."},
    )
    # Raw dataset jsonl path
    train_jsonl_path: str = "data/train.jsonl"
    valid_jsonl_path: str = "data/valid.jsonl"
    test_jsonl_path: str = "data/test.jsonl"
    # Where to save the preprocessed dataset (HF dataset save_to_disk)
    preprocessed_dir: str = "data/my_sft_messages"
    # Whether to reuse existing preprocessed_dir
    reuse_preprocessed: bool = False
    # Whether to mask prompt loss (recommended for SFT)
    mask_prompt_loss: bool = field(
        default=True,
        metadata={"help": "Whether to mask the loss on the prompt tokens"},
    )
    # Dream SFT specific args
    perbatch_cutoff: bool = field(
        default=True,
        metadata={"help": "Dream: randomly truncate responses per batch before padding."},
    )
    resp_cutoff_ratio: float = field(
        default=0.0,
        metadata={"help": "Dream: probability of post-collation truncation."},
    )


@dataclass
class TrainingArguments(MDLMTrainer.MDLMConfig):
    output_dir: str = "models/LLaDA-8B-Instruct/my_sft"
    group_by_length: bool = True
    num_train_epochs: float = 5
    learning_rate: float = 2e-5
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 8
    gradient_checkpointing: bool = True


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


def _map_to_messages(example: Dict[str, Any]) -> Dict[str, Any]:
    prompt = _build_prompt(example)
    response = str(example.get("tgt", example.get("trg", "")))
    return {
        "messages": [
            {"role": "user", "content": prompt},
            {"role": "assistant", "content": response},
        ]
    }


def _prepare_dataset(
    train_jsonl_path: str,
    valid_jsonl_path: str,
    test_jsonl_path: str,
    preprocessed_dir: str,
    reuse_preprocessed: bool,
):
    if reuse_preprocessed and os.path.isdir(preprocessed_dir):
        logger.info("Reuse preprocessed dataset at %s", preprocessed_dir)
        return load_sft_dataset(preprocessed_dir, load_preprocessed_data=True)

    data_files = {"train": train_jsonl_path}
    if valid_jsonl_path and os.path.exists(valid_jsonl_path):
        data_files["validation"] = valid_jsonl_path
    if test_jsonl_path and os.path.exists(test_jsonl_path):
        data_files["test"] = test_jsonl_path
    logger.info("Loading raw JSONL: %s", data_files)
    raw = load_dataset("json", data_files=data_files)

    mapped = {}
    for split, ds in raw.items():
        mapped[split] = ds.map(_map_to_messages, remove_columns=ds.column_names)
    ds = DatasetDict(mapped)

    os.makedirs(preprocessed_dir, exist_ok=True)
    ds.save_to_disk(preprocessed_dir)
    logger.info("Saved preprocessed dataset to %s", preprocessed_dir)
    return ds


def train():
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments)
    )
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    data_args.sft_backend = data_args.sft_backend.lower().strip()
    if data_args.sft_backend not in {"llada", "dream"}:
        raise ValueError("sft_backend must be either 'llada' or 'dream'.")
    if (
        training_args.gradient_checkpointing
        and training_args.gradient_checkpointing_kwargs is None
    ):
        training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}
    print_args_main(model_args, data_args, training_args)
    initial_training_setup(model_args, data_args, training_args)

    model = get_model(model_args=model_args)
    tokenizer = get_tokenizer(model_args=model_args)

    with accelerate.PartialState().local_main_process_first():
        dataset = _prepare_dataset(
            data_args.train_jsonl_path,
            data_args.valid_jsonl_path,
            data_args.test_jsonl_path,
            data_args.preprocessed_dir,
            data_args.reuse_preprocessed,
        )
        map_fn = partial(
            default_sft_map_fn,
            tokenizer=tokenizer,
            mask_prompt_loss=data_args.mask_prompt_loss,
        )
        dataset = dataset.map(
            map_fn,
            num_proc=data_args.num_proc,
            desc="Mapping dataset to SFT format",
        )
        dataset = post_process_dataset(dataset, data_args)

    accelerate.PartialState().wait_for_everyone()
    logger.info("Start training...")
    eval_dataset = dataset.get("validation", dataset.get("test", None))

    if data_args.sft_backend == "dream":
        from dllm.pipelines.dream.trainer import DreamTrainer
        from dllm.pipelines.dream.utils import DreamSFTCollator

        # Dream expects extra fields and right-shift logits.
        training_args.remove_unused_columns = False
        training_args.right_shift_logits = True
        if training_args.loss_weight_type == "scheduler":
            training_args.loss_weight_type = "cart[geo_p:0.3]"

        trainer = DreamTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=eval_dataset,
            args=training_args,
            data_collator=DreamSFTCollator(
                tokenizer,
                return_tensors="pt",
                padding=True,
                perbatch_cutoff=data_args.perbatch_cutoff,
                resp_cutoff_ratio=data_args.resp_cutoff_ratio,
            ),
        )
    else:
        trainer = MDLMTrainer(
            model=model,
            tokenizer=tokenizer,
            train_dataset=dataset["train"],
            eval_dataset=eval_dataset,
            args=training_args,
            data_collator=(
                NoAttentionMaskWrapper(
                    transformers.DataCollatorForSeq2Seq(
                        tokenizer,
                        return_tensors="pt",
                        padding=True,
                        label_pad_token_id=tokenizer.pad_token_id,
                    )
                )
            ),
        )
    trainer.train()
    trainer.save_model(os.path.join(training_args.output_dir, "checkpoint-final"))
    trainer.processing_class.save_pretrained(
        os.path.join(training_args.output_dir, "checkpoint-final")
    )


if __name__ == "__main__":
    train()
