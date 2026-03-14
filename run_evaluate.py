#!/usr/bin/env python3
"""Evaluate generation results from a JSONL file.

Expected JSONL format (configurable keys):
{
  "prediction": "...",
  "reference": ["...", "..."]
}
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from transformers import AutoModel, AutoTokenizer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compute generation quality and diversity metrics for JSONL outputs."
    )
    parser.add_argument("--input", type=Path, required=True, help="Path to input JSONL.")
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional path to write a JSON metrics report.",
    )
    parser.add_argument(
        "--prediction-key",
        type=str,
        default="prediction",
        help="JSON key for prediction text.",
    )
    parser.add_argument(
        "--reference-key",
        type=str,
        default="reference",
        help="JSON key for reference text/list.",
    )
    parser.add_argument(
        "--lowercase",
        action="store_true",
        help="Lowercase predictions and references before metric computation.",
    )
    parser.add_argument(
        "--distinct-n",
        type=str,
        default="1,2",
        help="Comma-separated n values for Distinct-n (e.g. 1,2,3,4).",
    )
    parser.add_argument(
        "--ppl-model",
        type=str,
        default=None,
        help="Optional causal LM name/path for perplexity (e.g. gpt2).",
    )
    parser.add_argument(
        "--ppl-batch-size",
        type=int,
        default=8,
        help="Batch size for perplexity computation.",
    )
    parser.add_argument(
        "--ppl-max-length",
        type=int,
        default=512,
        help="Max token length for perplexity inputs.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device for perplexity model.",
    )
    return parser.parse_args()


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON at line {i} in {path}: {exc}") from exc
            if not isinstance(item, dict):
                raise ValueError(f"Line {i} is not a JSON object.")
            records.append(item)
    return records


def normalize_text(text: str, lowercase: bool) -> str:
    text = str(text).strip()
    if lowercase:
        text = text.lower()
    return re.sub(r"\s+", " ", text)


def prepare_predictions_and_references(
    data: list[dict[str, Any]],
    prediction_key: str,
    reference_key: str,
    lowercase: bool,
) -> tuple[list[str], list[list[str]]]:
    predictions: list[str] = []
    references: list[list[str]] = []
    for i, row in enumerate(data):
        if prediction_key not in row:
            raise KeyError(f"Missing '{prediction_key}' in sample index {i}.")
        if reference_key not in row:
            raise KeyError(f"Missing '{reference_key}' in sample index {i}.")

        pred = normalize_text(row[prediction_key], lowercase=lowercase)
        ref_raw = row[reference_key]
        if isinstance(ref_raw, str):
            ref_list = [ref_raw]
        elif isinstance(ref_raw, list):
            ref_list = [str(x) for x in ref_raw]
        else:
            raise TypeError(
                f"'{reference_key}' must be str or list[str] at sample index {i}."
            )

        ref_list = [
            normalize_text(r, lowercase=lowercase) for r in ref_list if str(r).strip()
        ]
        if not ref_list:
            raise ValueError(f"No valid references at sample index {i}.")

        predictions.append(pred)
        references.append(ref_list)
    return predictions, references


def _word_tokens(text: str) -> list[str]:
    return re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text.lower())


def compute_unique_word_count(predictions: list[str]) -> int:
    vocab = set()
    for pred in predictions:
        vocab.update(_word_tokens(pred))
    return len(vocab)


def compute_distinct_n(predictions: list[str], n: int) -> float:
    total = 0
    unique = set()
    for pred in predictions:
        tokens = _word_tokens(pred)
        if len(tokens) < n:
            continue
        ngrams = [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]
        total += len(ngrams)
        unique.update(ngrams)
    return float(len(unique) / total) if total > 0 else 0.0


def compute_self_bleu(predictions: list[str]) -> float:
    if len(predictions) < 2:
        return 0.0

    from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu

    smoothie = SmoothingFunction().method1
    scores = []
    for i, pred in enumerate(predictions):
        candidate = _word_tokens(pred)
        if not candidate:
            scores.append(0.0)
            continue

        refs = [_word_tokens(predictions[j]) for j in range(len(predictions)) if j != i]
        refs = [r for r in refs if r]
        if not refs:
            scores.append(0.0)
            continue
        score = sentence_bleu(
            refs,
            candidate,
            weights=(0.25, 0.25, 0.25, 0.25),
            smoothing_function=smoothie,
        )
        scores.append(float(score))
    return float(np.mean(scores))


def compute_self_similarity(predictions: list[str]) -> float:
    if len(predictions) < 2:
        return 0.0
    vec = TfidfVectorizer(ngram_range=(1, 2), lowercase=True).fit_transform(predictions)
    sim = cosine_similarity(vec)
    n = sim.shape[0]
    triu = np.triu_indices(n, k=1)
    return float(np.mean(sim[triu])) if len(triu[0]) > 0 else 0.0



def compute_self_Cossim(predictions:list[str]) -> float:
    # calculate the average cosine similarity between all pairs of generated texts. using SimCSE to get embeddings
    
    tokenizer = AutoTokenizer.from_pretrained('princeton-nlp/sup-simcse-bert-base-uncased')
    model = AutoModel.from_pretrained('princeton-nlp/sup-simcse-bert-base-uncased')
    
    inputs = tokenizer(predictions, padding=True, truncation=True, return_tensors="pt")
    with torch.no_grad():
        embeddings = model(**inputs, output_hidden_states=True, return_dict=True).pooler_output
    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
    cosine_sim_matrix = torch.matmul(embeddings, embeddings.T)
    n = cosine_sim_matrix.size(0)
    sum_cosine_sim = torch.sum(cosine_sim_matrix) - n  # exclude self-similarity
    mean_self_cossim = (sum_cosine_sim / (n * (n - 1))).item()
    
    return mean_self_cossim

def compute_coco_metrics(
    predictions: list[str], references: list[list[str]]
) -> tuple[dict[str, float | None], list[str]]:
    metrics: dict[str, float | None] = {
        "bleu_4": None,
        "rouge_l": None,
        "meteor": None,
        "cider": None,
        "spice": None,
    }
    warnings: list[str] = []

    try:
        from pycocoevalcap.bleu.bleu import Bleu
        from pycocoevalcap.cider.cider import Cider
        from pycocoevalcap.meteor.meteor import Meteor
        from pycocoevalcap.rouge.rouge import Rouge
        from pycocoevalcap.spice.spice import Spice
    except Exception as exc:
        warnings.append(f"Skipping COCO metrics imports: {exc}")
        return metrics, warnings

    gts = {i: references[i] for i in range(len(predictions))}
    res = {i: [predictions[i]] for i in range(len(predictions))}

    scorers = [
        ("bleu_4", Bleu(4)),
        ("rouge_l", Rouge()),
        ("meteor", Meteor()),
        ("cider", Cider()),
        ("spice", Spice()),
    ]

    for name, scorer in scorers:
        try:
            score, _ = scorer.compute_score(gts, res)
            if name == "bleu_4":
                metrics[name] = float(score[3])
            else:
                metrics[name] = float(score)
        except Exception as exc:
            warnings.append(f"Failed {name}: {exc}")
            metrics[name] = None

    return metrics, warnings


def compute_perplexity(
    predictions: list[str],
    model_name_or_path: str,
    batch_size: int,
    max_length: int,
    device: str,
) -> float | None:
    if not predictions:
        return None

    from transformers import AutoModelForCausalLM, AutoTokenizer

    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_device = torch.device(device)

    tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path).to(torch_device)
    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    total_nll = 0.0
    total_tokens = 0

    for i in range(0, len(predictions), batch_size):
        batch_texts = predictions[i : i + batch_size]
        enc = tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_length,
        ).to(torch_device)

        labels = enc["input_ids"].clone()
        labels[enc["attention_mask"] == 0] = -100

        with torch.no_grad():
            outputs = model(**enc, labels=labels)
            loss = outputs.loss

        valid_tokens = int((labels != -100).sum().item())
        total_nll += float(loss.item()) * valid_tokens
        total_tokens += valid_tokens

    if total_tokens == 0:
        return None
    return float(math.exp(total_nll / total_tokens))


def main() -> None:
    args = parse_args()
    data = load_jsonl(args.input)
    if not data:
        raise ValueError(f"No valid records found in {args.input}")

    predictions, references = prepare_predictions_and_references(
        data=data,
        prediction_key=args.prediction_key,
        reference_key=args.reference_key,
        lowercase=args.lowercase,
    )

    coco_metrics, warnings = compute_coco_metrics(predictions, references)

    distinct_ns = [int(x.strip()) for x in args.distinct_n.split(",") if x.strip()]
    distinct_metrics = {
        f"distinct_{n}": compute_distinct_n(predictions, n) for n in distinct_ns
    }

    ppl = None
    if args.ppl_model:
        ppl = compute_perplexity(
            predictions=predictions,
            model_name_or_path=args.ppl_model,
            batch_size=args.ppl_batch_size,
            max_length=args.ppl_max_length,
            device=args.device,
        )

    report: dict[str, Any] = {
        "input_file": str(args.input),
        "num_samples": len(predictions),
        "generation_quality": {
            **coco_metrics,
            "perplexity": ppl,
            "perplexity_model": args.ppl_model,
        },
        "diversity": {
            "unique_word_count": compute_unique_word_count(predictions),
            "self_bleu": compute_self_bleu(predictions),
            **distinct_metrics,
            "self_cosine_similarity": compute_self_Cossim(predictions),
        },
        "warnings": warnings,
    }

    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
