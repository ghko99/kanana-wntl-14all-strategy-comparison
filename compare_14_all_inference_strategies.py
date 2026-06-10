import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

import matplotlib.pyplot as plt
import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from modules.inference_module import load_inference_model
from modules.number_tokenizer import AutoNumberTokenizer
from weighted_digit_inference import (
    DEFAULT_BASE_MODEL_NAME,
    RUBRICS,
    DigitDistributionHelper,
    append_jsonl,
    build_prompt_and_label,
    clip_round_score,
    ensure_dir,
    extract_grader_scores,
    now_kst,
    parse_ground_truth_scores,
    qwk_numpy,
    save_json,
    select_rubric_steps,
)


RUBRIC_LABELS = {
    "task_1": "과제 수행의 충실성",
    "content_1": "설명의 명료성",
    "content_2": "설명의 구체성",
    "content_3": "설명의 적절성",
    "organization_1": "문장의 연결성",
    "organization_2": "글의 통일성",
    "expression_1": "어휘의 적절성",
    "expression_2": "어법의 적절성",
}


def jsonl_write(path: str, row: Dict[str, Any]) -> None:
    append_jsonl(row, path)


def metrics_without_overall(labels: np.ndarray, preds: np.ndarray, min_rating: int, max_rating: int) -> Dict[str, float]:
    result = {
        rubric: qwk_numpy(labels[:, idx], preds[:, idx], min_rating, max_rating)
        for idx, rubric in enumerate(RUBRICS)
    }
    result["average"] = float(np.mean([result[rubric] for rubric in RUBRICS]))
    return result


def hard_bank_to_numpy(bank: List[List[Optional[List[int]]]]) -> np.ndarray:
    n_items = len(bank)
    max_m = len(bank[0]) if n_items else 0
    arr = np.full((n_items, max_m, len(RUBRICS)), -1, dtype=np.int16)
    for item_idx, samples in enumerate(bank):
        if len(samples) != max_m:
            raise ValueError(f"sample bank length mismatch at idx={item_idx}: {len(samples)} != {max_m}")
        for m_idx, scores in enumerate(samples):
            if scores is not None and len(scores) == len(RUBRICS):
                arr[item_idx, m_idx, :] = np.asarray(scores, dtype=np.int16)
    return arr


def soft_bank_to_numpy(bank: List[List[Optional[List[float]]]]) -> np.ndarray:
    n_items = len(bank)
    max_m = len(bank[0]) if n_items else 0
    arr = np.full((n_items, max_m, len(RUBRICS)), np.nan, dtype=np.float32)
    for item_idx, samples in enumerate(bank):
        if len(samples) != max_m:
            raise ValueError(f"sample bank length mismatch at idx={item_idx}: {len(samples)} != {max_m}")
        for m_idx, scores in enumerate(samples):
            if scores is not None and len(scores) == len(RUBRICS):
                arr[item_idx, m_idx, :] = np.asarray(scores, dtype=np.float32)
    return arr


def hard_preds_at_m(hard_arr: np.ndarray, m: int, min_digit: int, max_digit: int, fallback_score: int) -> np.ndarray:
    subset = hard_arr[:, :m, :]
    valid = subset != -1
    sums = np.where(valid, subset, 0).sum(axis=1, dtype=np.float32)
    counts = valid.sum(axis=1)
    means = sums / np.maximum(counts, 1)
    preds = np.rint(means).astype(np.int64)
    preds[counts == 0] = fallback_score
    return np.clip(preds, min_digit, max_digit)


def soft_preds_at_m(soft_arr: np.ndarray, m: int, min_digit: int, max_digit: int, fallback_score: int) -> Tuple[np.ndarray, np.ndarray]:
    with np.errstate(invalid="ignore"):
        means = np.nanmean(soft_arr[:, :m, :], axis=1)
    means = np.where(np.isfinite(means), means, float(fallback_score))
    preds = np.rint(means).astype(np.int64)
    return np.clip(preds, min_digit, max_digit), means


def selected_scores_from_steps(selected_steps: List[Dict[str, Any]]) -> Tuple[List[int], List[float], List[int], List[float]]:
    hard_scores: List[int] = []
    soft_raw: List[float] = []
    argmax_scores: List[int] = []
    raw_digit_mass: List[float] = []

    for step in selected_steps:
        chosen_digit = step.get("chosen_digit")
        argmax_digit = int(step["argmax_digit"])
        hard_scores.append(int(chosen_digit) if chosen_digit is not None else argmax_digit)
        soft_raw.append(float(step["expected_digit"]))
        argmax_scores.append(argmax_digit)
        raw_digit_mass.append(float(step["raw_digit_mass"]))

    return hard_scores, soft_raw, argmax_scores, raw_digit_mass


@torch.inference_mode()
def run_greedy(
    model,
    tokenizer,
    dataset,
    helper: DigitDistributionHelper,
    out_dir: str,
    max_new_tokens: int,
    min_digit: int,
    max_digit: int,
    min_digit_mass_for_fallback: float,
    fallback_score: int,
    limit: Optional[int],
) -> Tuple[np.ndarray, List[Optional[List[float]]], List[Optional[List[float]]], np.ndarray, np.ndarray, np.ndarray]:
    labels: List[List[int]] = []
    grader_1_labels: List[Optional[List[float]]] = []
    grader_2_labels: List[Optional[List[float]]] = []
    hard_preds: List[List[int]] = []
    soft_preds: List[List[int]] = []
    soft_raw_values: List[List[float]] = []

    ground_truth_path = os.path.join(out_dir, "ground_truth.jsonl")
    greedy_path = os.path.join(out_dir, "greedy_predictions.jsonl")
    for path in [ground_truth_path, greedy_path]:
        if os.path.exists(path):
            os.remove(path)

    iterator = dataset if limit is None else dataset.select(range(min(limit, len(dataset))))
    for sample_idx, example in enumerate(tqdm(iterator, desc="Greedy / soft greedy inference")):
        prompt, gt_text, uses_chat_template = build_prompt_and_label(tokenizer, example)
        gt_scores = parse_ground_truth_scores(gt_text, expected_count=len(RUBRICS))
        grader_1_scores, grader_2_scores = extract_grader_scores(example)
        labels.append(gt_scores)
        grader_1_labels.append(grader_1_scores)
        grader_2_labels.append(grader_2_scores)
        jsonl_write(
            ground_truth_path,
            {
                "sample_idx": sample_idx,
                "ground_truth": gt_scores,
                "grader_1_scores": grader_1_scores,
                "grader_2_scores": grader_2_scores,
            },
        )

        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=tokenizer.model_max_length,
            padding=False,
            add_special_tokens=(not uses_chat_template),
        )
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)

        gen_out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            num_beams=1,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        generated_ids = gen_out.sequences[:, input_ids.size(1):]
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        step_infos: List[Dict[str, Any]] = []
        for step_idx, step_logits in enumerate(gen_out.scores):
            chosen_id = int(generated_ids[0, step_idx].item())
            summary = helper.summarize_logits(step_logits[0])
            step_infos.append(
                {
                    "gen_pos": step_idx + 1,
                    "chosen_id": chosen_id,
                    "chosen_text": tokenizer.decode([chosen_id], skip_special_tokens=True),
                    "chosen_digit": helper.id_to_digit.get(chosen_id),
                    "raw_digit_mass": summary["raw_digit_mass"],
                    "expected_digit": summary["expected_digit"],
                    "argmax_digit": summary["argmax_digit"],
                    "renorm_by_digit": summary["renorm_by_digit"],
                    "raw_by_digit": summary["raw_by_digit"],
                    "used_fallback": False,
                }
            )

        selected_steps = select_rubric_steps(
            step_infos=step_infos,
            expected_count=len(RUBRICS),
            min_digit_mass_for_fallback=min_digit_mass_for_fallback,
        )
        if len(selected_steps) != len(RUBRICS):
            hard_scores = [fallback_score] * len(RUBRICS)
            soft_raw = [float(fallback_score)] * len(RUBRICS)
            argmax_scores = [fallback_score] * len(RUBRICS)
            raw_digit_mass = [0.0] * len(RUBRICS)
        else:
            hard_scores, soft_raw, argmax_scores, raw_digit_mass = selected_scores_from_steps(selected_steps)

        soft_rounded = [
            clip_round_score(score, min_digit=min_digit, max_digit=max_digit)
            for score in soft_raw
        ]

        hard_preds.append(hard_scores)
        soft_preds.append(soft_rounded)
        soft_raw_values.append(soft_raw)

        jsonl_write(
            greedy_path,
            {
                "sample_idx": sample_idx,
                "ground_truth": gt_scores,
                "grader_1_scores": grader_1_scores,
                "grader_2_scores": grader_2_scores,
                "generated_text": generated_text,
                "hard_greedy_pred": hard_scores,
                "soft_greedy_raw": soft_raw,
                "soft_greedy_pred": soft_rounded,
                "selected_positions": [step["gen_pos"] for step in selected_steps],
                "selected_argmax_digits": argmax_scores,
                "selected_raw_digit_mass": raw_digit_mass,
            },
        )

    return (
        np.asarray(labels, dtype=np.int64),
        grader_1_labels,
        grader_2_labels,
        np.asarray(hard_preds, dtype=np.int64),
        np.asarray(soft_preds, dtype=np.int64),
        np.asarray(soft_raw_values, dtype=np.float32),
    )


@torch.inference_mode()
def run_self_consistency(
    model,
    tokenizer,
    dataset,
    helper: DigitDistributionHelper,
    out_dir: str,
    max_m: int,
    chunk_m: int,
    top_k: int,
    temperature: float,
    max_new_tokens: int,
    min_digit: int,
    max_digit: int,
    min_digit_mass_for_fallback: float,
    fallback_score: int,
    limit: Optional[int],
) -> Tuple[List[List[Optional[List[int]]]], List[List[Optional[List[float]]]], Dict[str, Any]]:
    sample_path = os.path.join(out_dir, "self_consistency_samples.jsonl")
    if os.path.exists(sample_path):
        os.remove(sample_path)

    hard_bank: List[List[Optional[List[int]]]] = []
    soft_bank: List[List[Optional[List[float]]]] = []
    stats = defaultdict(int)

    iterator = dataset if limit is None else dataset.select(range(min(limit, len(dataset))))
    for sample_idx, example in enumerate(tqdm(iterator, desc=f"Self-consistency sampling m={max_m}")):
        prompt, gt_text, uses_chat_template = build_prompt_and_label(tokenizer, example)
        gt_scores = parse_ground_truth_scores(gt_text, expected_count=len(RUBRICS))
        grader_1_scores, grader_2_scores = extract_grader_scores(example)

        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=tokenizer.model_max_length,
            padding=False,
            add_special_tokens=(not uses_chat_template),
        )
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)

        hard_samples: List[Optional[List[int]]] = []
        soft_samples: List[Optional[List[float]]] = []
        remaining = max_m
        sample_number = 0

        while remaining > 0:
            cur_chunk_size = min(chunk_m, remaining)
            gen_out = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                num_return_sequences=cur_chunk_size,
                num_beams=1,
                temperature=temperature,
                top_k=top_k,
                return_dict_in_generate=True,
                output_scores=True,
                pad_token_id=tokenizer.pad_token_id,
                eos_token_id=tokenizer.eos_token_id,
            )

            generated_ids = gen_out.sequences[:, input_ids.size(1):]
            generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

            for seq_idx, generated_text in enumerate(generated_texts):
                step_infos: List[Dict[str, Any]] = []
                for step_idx, step_logits in enumerate(gen_out.scores):
                    chosen_id = int(generated_ids[seq_idx, step_idx].item())
                    summary = helper.summarize_logits(step_logits[seq_idx])
                    step_infos.append(
                        {
                            "gen_pos": step_idx + 1,
                            "chosen_id": chosen_id,
                            "chosen_text": tokenizer.decode([chosen_id], skip_special_tokens=True),
                            "chosen_digit": helper.id_to_digit.get(chosen_id),
                            "raw_digit_mass": summary["raw_digit_mass"],
                            "expected_digit": summary["expected_digit"],
                            "argmax_digit": summary["argmax_digit"],
                            "renorm_by_digit": summary["renorm_by_digit"],
                            "raw_by_digit": summary["raw_by_digit"],
                            "used_fallback": False,
                        }
                    )

                selected_steps = select_rubric_steps(
                    step_infos=step_infos,
                    expected_count=len(RUBRICS),
                    min_digit_mass_for_fallback=min_digit_mass_for_fallback,
                )
                if len(selected_steps) == len(RUBRICS):
                    hard_scores, soft_raw, argmax_scores, raw_digit_mass = selected_scores_from_steps(selected_steps)
                    stats["hard_valid_samples"] += 1
                    stats["soft_valid_samples"] += 1
                else:
                    hard_scores = None
                    soft_raw = None
                    argmax_scores = []
                    raw_digit_mass = []
                    stats["invalid_samples"] += 1

                hard_samples.append(hard_scores)
                soft_samples.append(soft_raw)
                sample_number += 1

                jsonl_write(
                    sample_path,
                    {
                        "sample_idx": sample_idx,
                        "sample_number": sample_number,
                        "ground_truth": gt_scores,
                        "grader_1_scores": grader_1_scores,
                        "grader_2_scores": grader_2_scores,
                        "generated_text": generated_text,
                        "hard_digits": hard_scores,
                        "soft_expected_raw": soft_raw,
                        "selected_positions": [step["gen_pos"] for step in selected_steps],
                        "selected_argmax_digits": argmax_scores,
                        "selected_expected_round": [
                            clip_round_score(step["expected_digit"], min_digit=min_digit, max_digit=max_digit)
                            for step in selected_steps
                        ],
                        "selected_raw_digit_mass": raw_digit_mass,
                    },
                )

            remaining -= cur_chunk_size

        hard_bank.append(hard_samples)
        soft_bank.append(soft_samples)
        stats["num_examples"] += 1

    stats["max_m"] = max_m
    stats["chunk_m"] = chunk_m
    return hard_bank, soft_bank, dict(stats)


def build_curves(
    labels: np.ndarray,
    hard_arr: np.ndarray,
    soft_arr: np.ndarray,
    min_digit: int,
    max_digit: int,
    fallback_score: int,
) -> Tuple[Dict[str, Any], Dict[str, int], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    ms = list(range(1, hard_arr.shape[1] + 1))
    curves = {
        "m": ms,
        "hard_sc_average_qwk": [],
        "soft_sc_average_qwk": [],
        "hard_sc_rubric_qwk": [],
        "soft_sc_rubric_qwk": [],
    }

    for m in ms:
        hard_preds = hard_preds_at_m(hard_arr, m, min_digit, max_digit, fallback_score)
        soft_preds, _ = soft_preds_at_m(soft_arr, m, min_digit, max_digit, fallback_score)
        hard_metrics = metrics_without_overall(labels, hard_preds, min_digit, max_digit)
        soft_metrics = metrics_without_overall(labels, soft_preds, min_digit, max_digit)
        curves["hard_sc_average_qwk"].append(hard_metrics["average"])
        curves["soft_sc_average_qwk"].append(soft_metrics["average"])
        curves["hard_sc_rubric_qwk"].append({rubric: hard_metrics[rubric] for rubric in RUBRICS})
        curves["soft_sc_rubric_qwk"].append({rubric: soft_metrics[rubric] for rubric in RUBRICS})

    best_m = {
        "hard_sc": int(ms[int(np.argmax(curves["hard_sc_average_qwk"]))]),
        "soft_sc": int(ms[int(np.argmax(curves["soft_sc_average_qwk"]))]),
    }
    final_preds = {
        "hard_sc": hard_preds_at_m(hard_arr, best_m["hard_sc"], min_digit, max_digit, fallback_score),
        "soft_sc": soft_preds_at_m(soft_arr, best_m["soft_sc"], min_digit, max_digit, fallback_score)[0],
    }
    final_soft_raw = {
        "soft_sc": soft_preds_at_m(soft_arr, best_m["soft_sc"], min_digit, max_digit, fallback_score)[1],
    }
    return curves, best_m, final_preds, final_soft_raw


def save_curve_csv(curves: Dict[str, Any], path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["m", "hard_sc_average_qwk", "soft_sc_average_qwk"])
        writer.writeheader()
        for idx, m in enumerate(curves["m"]):
            writer.writerow(
                {
                    "m": m,
                    "hard_sc_average_qwk": curves["hard_sc_average_qwk"][idx],
                    "soft_sc_average_qwk": curves["soft_sc_average_qwk"][idx],
                }
            )


def save_tables(
    out_dir: str,
    strategy_metrics: Dict[str, Dict[str, Any]],
    best_m: Dict[str, int],
) -> None:
    summary_path = os.path.join(out_dir, "strategy_summary.csv")
    fieldnames = ["strategy", "m", "average"] + RUBRICS
    with open(summary_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for strategy, payload in strategy_metrics.items():
            row = {
                "strategy": strategy,
                "m": payload.get("m", ""),
                "average": payload["metrics"]["average"],
            }
            row.update({rubric: payload["metrics"][rubric] for rubric in RUBRICS})
            writer.writerow(row)

    rubric_path = os.path.join(out_dir, "rubric_comparison.csv")
    with open(rubric_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["rubric", "label", "hard_greedy", "soft_greedy", "hard_sc", "soft_sc"],
        )
        writer.writeheader()
        for rubric in RUBRICS:
            writer.writerow(
                {
                    "rubric": rubric,
                    "label": RUBRIC_LABELS[rubric],
                    "hard_greedy": strategy_metrics["hard_greedy"]["metrics"][rubric],
                    "soft_greedy": strategy_metrics["soft_greedy"]["metrics"][rubric],
                    "hard_sc": strategy_metrics["hard_sc"]["metrics"][rubric],
                    "soft_sc": strategy_metrics["soft_sc"]["metrics"][rubric],
                }
            )
        writer.writerow(
            {
                "rubric": "average",
                "label": "평균 (Average)",
                "hard_greedy": strategy_metrics["hard_greedy"]["metrics"]["average"],
                "soft_greedy": strategy_metrics["soft_greedy"]["metrics"]["average"],
                "hard_sc": strategy_metrics["hard_sc"]["metrics"]["average"],
                "soft_sc": strategy_metrics["soft_sc"]["metrics"]["average"],
            }
        )

    markdown_path = os.path.join(out_dir, "rubric_comparison.md")
    with open(markdown_path, "w", encoding="utf-8") as f:
        f.write("| 평가 항목 | hard greedy | soft greedy | hard SC | soft SC |\n")
        f.write("|---|---:|---:|---:|---:|\n")
        for rubric in RUBRICS:
            f.write(
                f"| {RUBRIC_LABELS[rubric]} "
                f"| {strategy_metrics['hard_greedy']['metrics'][rubric]:.3f} "
                f"| {strategy_metrics['soft_greedy']['metrics'][rubric]:.3f} "
                f"| {strategy_metrics['hard_sc']['metrics'][rubric]:.3f} "
                f"| {strategy_metrics['soft_sc']['metrics'][rubric]:.3f} |\n"
            )
        f.write(
            f"| 평균 (Average) "
            f"| {strategy_metrics['hard_greedy']['metrics']['average']:.3f} "
            f"| {strategy_metrics['soft_greedy']['metrics']['average']:.3f} "
            f"| {strategy_metrics['hard_sc']['metrics']['average']:.3f} "
            f"| {strategy_metrics['soft_sc']['metrics']['average']:.3f} |\n\n"
        )
        f.write(f"- hard SC best m by 8-rubric average QWK: {best_m['hard_sc']}\n")
        f.write(f"- soft SC best m by 8-rubric average QWK: {best_m['soft_sc']}\n")


def save_final_predictions(
    out_dir: str,
    labels: np.ndarray,
    grader_1_scores: List[Optional[List[float]]],
    grader_2_scores: List[Optional[List[float]]],
    hard_greedy: np.ndarray,
    soft_greedy: np.ndarray,
    soft_greedy_raw: np.ndarray,
    hard_sc: np.ndarray,
    soft_sc: np.ndarray,
    soft_sc_raw: np.ndarray,
    best_m: Dict[str, int],
) -> None:
    path = os.path.join(out_dir, "final_strategy_predictions.jsonl")
    if os.path.exists(path):
        os.remove(path)
    for idx in range(labels.shape[0]):
        jsonl_write(
            path,
            {
                "sample_idx": idx,
                "ground_truth": labels[idx].astype(int).tolist(),
                "grader_1_scores": grader_1_scores[idx],
                "grader_2_scores": grader_2_scores[idx],
                "hard_greedy_pred": hard_greedy[idx].astype(int).tolist(),
                "soft_greedy_raw": soft_greedy_raw[idx].astype(float).tolist(),
                "soft_greedy_pred": soft_greedy[idx].astype(int).tolist(),
                "hard_sc_m": best_m["hard_sc"],
                "hard_sc_pred": hard_sc[idx].astype(int).tolist(),
                "soft_sc_m": best_m["soft_sc"],
                "soft_sc_raw_mean": soft_sc_raw[idx].astype(float).tolist(),
                "soft_sc_pred": soft_sc[idx].astype(int).tolist(),
            },
        )


def plot_average_qwk(
    curves: Dict[str, Any],
    hard_greedy_average: float,
    soft_greedy_average: float,
    out_dir: str,
    xtick_step: int,
) -> None:
    ms = curves["m"]
    plt.figure(figsize=(10, 6))
    plt.plot(ms, curves["hard_sc_average_qwk"], marker="o", markersize=3, linewidth=2, label="Hard SC")
    plt.plot(ms, curves["soft_sc_average_qwk"], marker="o", markersize=3, linewidth=2, label="Soft SC")
    plt.axhline(hard_greedy_average, color="tab:gray", linestyle="--", linewidth=2, label="Hard greedy")
    plt.axhline(soft_greedy_average, color="tab:orange", linestyle="--", linewidth=2, label="Soft greedy")
    plt.xlabel("m (number of samples)")
    plt.ylabel("Average QWK (8 rubrics)")
    plt.xticks(ms[:: max(1, xtick_step)])
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "average_qwk_vs_m_with_greedy_baselines.png"), dpi=200)
    plt.close()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser("compare_14_all_inference_strategies")
    parser.add_argument("--adapter_dir", type=str, default="./kanana_wntl_20260407_002343")
    parser.add_argument("--base_model_name", type=str, default=DEFAULT_BASE_MODEL_NAME)
    parser.add_argument("--test_path", type=str, default="./aes_dataset_mtl/test_14_all.jsonl")
    parser.add_argument("--output_root", type=str, default="./strategy_comparison_results")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--tag", type=str, default="kanana_wntl_14_all_strategy_comparison")
    parser.add_argument("--device_id", type=int, default=0)
    parser.add_argument("--max_m", type=int, default=50)
    parser.add_argument("--chunk_m", type=int, default=10)
    parser.add_argument("--top_k", type=int, default=9)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--max_seq_length", type=int, default=3072)
    parser.add_argument("--min_digit", type=int, default=1)
    parser.add_argument("--max_digit", type=int, default=9)
    parser.add_argument("--min_digit_mass_for_fallback", type=float, default=0.05)
    parser.add_argument("--fallback_score", type=int, default=5)
    parser.add_argument("--xtick_step", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.device_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    timestamp = now_kst().strftime("%Y%m%d_%H%M%S_KST")
    out_dir = args.output_dir or os.path.join(args.output_root, f"{args.tag}_{timestamp}")
    ensure_dir(out_dir)

    config = vars(args).copy()
    config["time_kst"] = now_kst().isoformat()
    config["out_dir"] = out_dir
    config["selection_rule"] = "best m is selected by 8-rubric average QWK; overall QWK is not used."
    config["hard_sc_definition"] = "average hard decoded rubric scores over m samples, then round and clip."
    config["soft_sc_definition"] = "average soft digit expected scores over m samples, then round and clip."
    save_json(config, os.path.join(out_dir, "run_config.json"))

    print(f"[OUTPUT] {out_dir}")
    print("[LOAD] model/tokenizer/dataset")
    model = load_inference_model(args.adapter_dir, base_model_name=args.base_model_name)
    tokenizer = AutoNumberTokenizer.from_pretrained(args.adapter_dir, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.max_seq_length
    helper = DigitDistributionHelper(tokenizer, model.device, min_digit=args.min_digit, max_digit=args.max_digit)
    dataset = load_dataset("json", data_files=args.test_path)["train"]

    if args.dry_run:
        print("[DRY RUN] loaded successfully; no inference executed.")
        return

    labels, grader_1_scores, grader_2_scores, hard_greedy, soft_greedy, soft_greedy_raw = run_greedy(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        helper=helper,
        out_dir=out_dir,
        max_new_tokens=args.max_new_tokens,
        min_digit=args.min_digit,
        max_digit=args.max_digit,
        min_digit_mass_for_fallback=args.min_digit_mass_for_fallback,
        fallback_score=args.fallback_score,
        limit=args.limit,
    )

    hard_bank, soft_bank, sc_stats = run_self_consistency(
        model=model,
        tokenizer=tokenizer,
        dataset=dataset,
        helper=helper,
        out_dir=out_dir,
        max_m=args.max_m,
        chunk_m=args.chunk_m,
        top_k=args.top_k,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
        min_digit=args.min_digit,
        max_digit=args.max_digit,
        min_digit_mass_for_fallback=args.min_digit_mass_for_fallback,
        fallback_score=args.fallback_score,
        limit=args.limit,
    )

    hard_arr = hard_bank_to_numpy(hard_bank)
    soft_arr = soft_bank_to_numpy(soft_bank)
    curves, best_m, sc_final_preds, sc_final_raw = build_curves(
        labels=labels,
        hard_arr=hard_arr,
        soft_arr=soft_arr,
        min_digit=args.min_digit,
        max_digit=args.max_digit,
        fallback_score=args.fallback_score,
    )

    hard_greedy_metrics = metrics_without_overall(labels, hard_greedy, args.min_digit, args.max_digit)
    soft_greedy_metrics = metrics_without_overall(labels, soft_greedy, args.min_digit, args.max_digit)
    hard_sc_metrics = metrics_without_overall(labels, sc_final_preds["hard_sc"], args.min_digit, args.max_digit)
    soft_sc_metrics = metrics_without_overall(labels, sc_final_preds["soft_sc"], args.min_digit, args.max_digit)

    strategy_metrics = {
        "hard_greedy": {"m": None, "metrics": hard_greedy_metrics},
        "soft_greedy": {"m": None, "metrics": soft_greedy_metrics},
        "hard_sc": {"m": best_m["hard_sc"], "metrics": hard_sc_metrics},
        "soft_sc": {"m": best_m["soft_sc"], "metrics": soft_sc_metrics},
    }

    save_json({"stats": sc_stats, "best_m": best_m, "strategies": strategy_metrics}, os.path.join(out_dir, "results.json"))
    save_json(curves, os.path.join(out_dir, "curves.json"))
    save_curve_csv(curves, os.path.join(out_dir, "curves.csv"))
    save_tables(out_dir, strategy_metrics, best_m)
    save_final_predictions(
        out_dir=out_dir,
        labels=labels,
        grader_1_scores=grader_1_scores,
        grader_2_scores=grader_2_scores,
        hard_greedy=hard_greedy,
        soft_greedy=soft_greedy,
        soft_greedy_raw=soft_greedy_raw,
        hard_sc=sc_final_preds["hard_sc"],
        soft_sc=sc_final_preds["soft_sc"],
        soft_sc_raw=sc_final_raw["soft_sc"],
        best_m=best_m,
    )
    plot_average_qwk(
        curves=curves,
        hard_greedy_average=hard_greedy_metrics["average"],
        soft_greedy_average=soft_greedy_metrics["average"],
        out_dir=out_dir,
        xtick_step=args.xtick_step,
    )

    print("\n=== 14-all strategy comparison (8-rubric average QWK, no overall) ===")
    for name, payload in strategy_metrics.items():
        m_text = "-" if payload["m"] is None else str(payload["m"])
        print(f"{name}: m={m_text}, average={payload['metrics']['average']:.4f}")
    print(f"\nSaved outputs to: {out_dir}")


if __name__ == "__main__":
    main()
