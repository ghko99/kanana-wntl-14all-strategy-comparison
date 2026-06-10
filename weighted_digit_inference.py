import argparse
import datetime as dt
import json
import os
import unicodedata
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm

from modules.inference_module import load_inference_model
from modules.number_tokenizer import AutoNumberTokenizer


RUBRICS = [
    "task_1",
    "content_1",
    "content_2",
    "content_3",
    "organization_1",
    "organization_2",
    "expression_1",
    "expression_2",
]


def now_kst() -> dt.datetime:
    return dt.datetime.now(dt.timezone(dt.timedelta(hours=9)))


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: Any, path: str, indent: Optional[int] = 2) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)


def append_jsonl(obj: Any, path: str) -> None:
    ensure_dir(os.path.dirname(path))
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def extract_digits_0_9(text: str, limit: Optional[int] = None) -> List[int]:
    text = unicodedata.normalize("NFKC", text)
    digits = [int(ch) for ch in text if ch.isdecimal()]
    return digits[:limit] if limit is not None else digits


def normalize_example(example: Dict[str, Any]) -> Tuple[str, str]:
    instruction = example.get("instruction")
    output = example.get("output")
    if instruction and output:
        return instruction, output

    system_msg = example.get("system", "") or ""
    user_msg = example.get("user", "") or ""
    assistant_msg = example.get("assistant", "") or ""
    if (user_msg or system_msg) and assistant_msg:
        instruction_fallback = (system_msg + "\n" + user_msg).strip() if system_msg else user_msg
        return instruction_fallback, assistant_msg

    raise ValueError(
        f"[SCHEMA ERROR] cannot find (instruction,output) or (system/user,assistant) in keys={list(example.keys())}"
    )


def build_prompt_and_label(tokenizer, example: Dict[str, Any]) -> Tuple[str, str, bool]:
    has_chat_keys = ("assistant" in example) and (("user" in example) or ("system" in example))
    if has_chat_keys:
        system_msg = unicodedata.normalize("NFC", example.get("system", "") or "")
        user_msg = unicodedata.normalize("NFC", example.get("user", "") or "")
        assistant_msg = example.get("assistant", "") or ""
        messages = [
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_msg},
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return prompt, assistant_msg, True

    instruction = example.get("instruction")
    output = example.get("output")
    if instruction and output:
        return instruction, output, False

    raise ValueError(
        f"[SCHEMA ERROR] cannot find (instruction,output) or (system/user,assistant) in keys={list(example.keys())}"
    )


def parse_ground_truth_scores(text: str, expected_count: int = 8) -> List[int]:
    digits = extract_digits_0_9(text, limit=expected_count)
    if len(digits) != expected_count:
        raise ValueError(f"[GT FORMAT ERROR] expected {expected_count} digits, got {digits} from {repr(text)}")
    return digits


class DigitDistributionHelper:
    def __init__(self, tokenizer, device: torch.device, min_digit: int = 1, max_digit: int = 9):
        self.min_digit = min_digit
        self.max_digit = max_digit
        self.id_to_digit: Dict[int, int] = {}
        self.positions_by_digit: Dict[int, torch.Tensor] = {}

        grouped_ids: Dict[int, List[int]] = {digit: [] for digit in range(min_digit, max_digit + 1)}
        for token, token_id in tokenizer.get_vocab().items():
            try:
                value = tokenizer.decode_number_token(token)
            except ValueError:
                continue

            if float(value).is_integer():
                digit = int(value)
                if min_digit <= digit <= max_digit:
                    grouped_ids[digit].append(token_id)
                    self.id_to_digit[token_id] = digit

        missing = [digit for digit, ids in grouped_ids.items() if not ids]
        if missing:
            raise ValueError(f"Missing digit token ids for digits: {missing}")

        flat_ids: List[int] = []
        flat_values: List[float] = []
        for digit in range(min_digit, max_digit + 1):
            ids = sorted(set(grouped_ids[digit]))
            start = len(flat_ids)
            flat_ids.extend(ids)
            flat_values.extend([float(digit)] * len(ids))
            end = len(flat_ids)
            self.positions_by_digit[digit] = torch.arange(start, end, device=device, dtype=torch.long)

        self.flat_digit_token_ids = torch.tensor(flat_ids, device=device, dtype=torch.long)
        self.flat_digit_values = torch.tensor(flat_values, device=device, dtype=torch.float32)

    def summarize_logits(self, logits: torch.Tensor) -> Dict[str, Any]:
        probs = torch.softmax(torch.clamp(logits, min=-50, max=50), dim=-1)
        digit_probs = probs.index_select(0, self.flat_digit_token_ids)

        raw_digit_mass = float(digit_probs.sum().item())
        raw_by_digit: Dict[int, float] = {}
        renorm_by_digit: Dict[int, float] = {}
        expected_digit = 0.0
        argmax_digit = self.min_digit

        if raw_digit_mass > 0.0:
            for digit in range(self.min_digit, self.max_digit + 1):
                mass = float(digit_probs.index_select(0, self.positions_by_digit[digit]).sum().item())
                raw_by_digit[digit] = mass
                renorm_by_digit[digit] = mass / raw_digit_mass
            expected_digit = float(
                sum(float(digit) * renorm_by_digit[digit] for digit in range(self.min_digit, self.max_digit + 1))
            )
            argmax_digit = max(renorm_by_digit, key=renorm_by_digit.get)
        else:
            uniform = 1.0 / (self.max_digit - self.min_digit + 1)
            for digit in range(self.min_digit, self.max_digit + 1):
                raw_by_digit[digit] = 0.0
                renorm_by_digit[digit] = uniform
            expected_digit = float(
                sum(float(digit) * renorm_by_digit[digit] for digit in range(self.min_digit, self.max_digit + 1))
            )

        return {
            "raw_digit_mass": raw_digit_mass,
            "raw_by_digit": raw_by_digit,
            "renorm_by_digit": renorm_by_digit,
            "expected_digit": expected_digit,
            "argmax_digit": int(argmax_digit),
        }


def clip_round_score(value: float, min_digit: int = 1, max_digit: int = 9) -> int:
    return int(np.clip(np.rint(value), min_digit, max_digit))


def qwk_numpy(y_true: np.ndarray, y_pred: np.ndarray, min_rating: int, max_rating: int) -> float:
    y_true = np.asarray(y_true, dtype=np.int64)
    y_pred = np.asarray(y_pred, dtype=np.int64)

    k = max_rating - min_rating + 1
    if k <= 1:
        return 0.0

    yt = y_true - min_rating
    yp = y_pred - min_rating

    observed = np.bincount(k * yt + yp, minlength=k * k).reshape(k, k).astype(np.float64)
    total = observed.sum()
    if total == 0:
        return 0.0

    hist_true = observed.sum(axis=1)
    hist_pred = observed.sum(axis=0)
    expected = np.outer(hist_true, hist_pred) / total

    grid = np.arange(k)
    weights = (grid[:, None] - grid[None, :]) ** 2 / ((k - 1) ** 2)

    denom = (weights * expected).sum()
    if denom == 0:
        return 0.0
    return float(1.0 - (weights * observed).sum() / denom)


def compute_metrics(labels: np.ndarray, preds: np.ndarray) -> Dict[str, float]:
    min_rating = int(min(labels.min(), preds.min()))
    max_rating = int(max(labels.max(), preds.max()))

    result: Dict[str, float] = {}
    for idx, rubric in enumerate(RUBRICS):
        result[rubric] = qwk_numpy(labels[:, idx], preds[:, idx], min_rating, max_rating)
    result["overall"] = qwk_numpy(labels.flatten(), preds.flatten(), min_rating, max_rating)
    result["average"] = float(np.mean(list(result.values())))
    return result


def select_rubric_steps(
    step_infos: List[Dict[str, Any]],
    expected_count: int,
    min_digit_mass_for_fallback: float,
) -> List[Dict[str, Any]]:
    selected = [step for step in step_infos if step["chosen_digit"] is not None][:expected_count]
    selected_positions = {step["gen_pos"] for step in selected}

    if len(selected) < expected_count:
        fallback_steps = [
            dict(step, used_fallback=True)
            for step in step_infos
            if step["gen_pos"] not in selected_positions and step["raw_digit_mass"] >= min_digit_mass_for_fallback
        ]
        fallback_steps.sort(key=lambda step: step["gen_pos"])
        selected.extend(fallback_steps[: expected_count - len(selected)])

    selected.sort(key=lambda step: step["gen_pos"])
    return selected[:expected_count]


@torch.inference_mode()
def run_weighted_digit_inference(args: argparse.Namespace) -> str:
    if args.device_id is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.device_id)

    timestamp = now_kst().strftime("%Y%m%d_%H%M%S_KST")
    tag = args.tag or f"{os.path.basename(os.path.normpath(args.adapter_dir))}_{os.path.splitext(os.path.basename(args.test_path))[0]}"
    out_dir = args.output_dir or os.path.join(args.output_root, f"{tag}_{timestamp}")
    ensure_dir(out_dir)

    save_json(
        {
            "time_kst": now_kst().isoformat(),
            "adapter_dir": args.adapter_dir,
            "base_model_name": args.base_model_name,
            "test_path": args.test_path,
            "device_id": args.device_id,
            "max_new_tokens": args.max_new_tokens,
            "max_seq_length": args.max_seq_length,
            "min_digit_mass_for_fallback": args.min_digit_mass_for_fallback,
            "fallback_score": args.fallback_score,
            "digit_range": [args.min_digit, args.max_digit],
        },
        os.path.join(out_dir, "run_config.json"),
    )

    print(f"[RUN] outputs -> {out_dir}")
    print("Loading model and tokenizer...")

    model = load_inference_model(args.adapter_dir, base_model_name=args.base_model_name)
    tokenizer = AutoNumberTokenizer.from_pretrained(args.adapter_dir, use_fast=True, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.model_max_length = args.max_seq_length

    test_ds = load_dataset("json", data_files=args.test_path)["train"]

    if args.dry_run:
        print("Dry run complete. Model, tokenizer, and dataset loaded successfully.")
        return out_dir

    helper = DigitDistributionHelper(tokenizer, model.device, min_digit=args.min_digit, max_digit=args.max_digit)

    labels: List[List[int]] = []
    preds_expected: List[List[int]] = []
    preds_argmax: List[List[int]] = []
    sample_summaries: List[Dict[str, Any]] = []

    predictions_path = os.path.join(out_dir, "predictions.jsonl")
    if os.path.exists(predictions_path):
        os.remove(predictions_path)

    for sample_idx, example in enumerate(tqdm(test_ds, desc="Weighted digit inference")):
        prompt, gt_text, uses_chat_template = build_prompt_and_label(tokenizer, example)
        gt_scores = parse_ground_truth_scores(gt_text, expected_count=8)
        labels.append(gt_scores)

        enc = tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=args.max_seq_length,
            padding=False,
            add_special_tokens=(not uses_chat_template),
        )
        input_ids = enc["input_ids"].to(model.device)
        attention_mask = enc["attention_mask"].to(model.device)

        gen_out = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
            top_p=1.0,
            return_dict_in_generate=True,
            output_scores=True,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        generated_ids = gen_out.sequences[:, input_ids.size(1):]
        generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

        step_infos: List[Dict[str, Any]] = []
        for step_idx, logits in enumerate(gen_out.scores):
            step_logits = logits.squeeze(0)
            chosen_id = int(generated_ids[0, step_idx].item())
            summary = helper.summarize_logits(step_logits)
            chosen_digit = helper.id_to_digit.get(chosen_id)

            step_infos.append(
                {
                    "gen_pos": step_idx + 1,
                    "chosen_id": chosen_id,
                    "chosen_text": tokenizer.decode([chosen_id], skip_special_tokens=True),
                    "chosen_digit": chosen_digit,
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
            expected_count=8,
            min_digit_mass_for_fallback=args.min_digit_mass_for_fallback,
        )

        expected_scores = [clip_round_score(step["expected_digit"], args.min_digit, args.max_digit) for step in selected_steps]
        argmax_scores = [int(step["argmax_digit"]) for step in selected_steps]

        while len(expected_scores) < 8:
            expected_scores.append(args.fallback_score)
            argmax_scores.append(args.fallback_score)

        preds_expected.append(expected_scores[:8])
        preds_argmax.append(argmax_scores[:8])

        sample_summary = {
            "sample_idx": sample_idx,
            "ground_truth": gt_scores,
            "pred_expected": expected_scores[:8],
            "pred_argmax": argmax_scores[:8],
            "detected_digit_steps": sum(step["chosen_digit"] is not None for step in step_infos),
            "selected_positions": [step["gen_pos"] for step in selected_steps],
            "selected_raw_digit_mass": [step["raw_digit_mass"] for step in selected_steps],
            "selected_expected_raw": [step["expected_digit"] for step in selected_steps],
            "generated_text": generated_text,
            "selected_steps": selected_steps,
        }
        sample_summaries.append(sample_summary)
        append_jsonl(sample_summary, predictions_path)

    labels_arr = np.array(labels, dtype=np.int64)
    preds_expected_arr = np.array(preds_expected, dtype=np.int64)
    preds_argmax_arr = np.array(preds_argmax, dtype=np.int64)

    metrics = {
        "weighted_expected_round": compute_metrics(labels_arr, preds_expected_arr),
        "digit_argmax": compute_metrics(labels_arr, preds_argmax_arr),
        "diagnostics": {
            "num_samples": int(labels_arr.shape[0]),
            "mean_detected_digit_steps": float(np.mean([sample["detected_digit_steps"] for sample in sample_summaries])),
            "samples_with_all_8_detected": int(sum(sample["detected_digit_steps"] >= 8 for sample in sample_summaries)),
            "samples_using_fallback_selection": int(
                sum(any(step["used_fallback"] for step in sample["selected_steps"]) for sample in sample_summaries)
            ),
        },
    }

    save_json(metrics, os.path.join(out_dir, "metrics.json"))

    print("\n=== Weighted Expected Round ===")
    print(json.dumps(metrics["weighted_expected_round"], ensure_ascii=False, indent=2))
    print("\n=== Digit Argmax ===")
    print(json.dumps(metrics["digit_argmax"], ensure_ascii=False, indent=2))
    print("\n=== Diagnostics ===")
    print(json.dumps(metrics["diagnostics"], ensure_ascii=False, indent=2))

    return out_dir


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        "weighted_digit_inference",
        description="Single-pass weighted digit inference using all digit token mass with digit-only renormalization.",
    )
    parser.add_argument("--adapter_dir", type=str, default="./kanana_wntl_20260407_002343")
    parser.add_argument("--base_model_name", type=str, default="/home/khko/models/kanana")
    parser.add_argument("--test_path", type=str, default="./aes_dataset_mtl/test_14_all.jsonl")
    parser.add_argument("--device_id", type=int, default=0)

    parser.add_argument("--max_seq_length", type=int, default=1456)
    parser.add_argument("--max_new_tokens", type=int, default=32)
    parser.add_argument("--min_digit", type=int, default=1)
    parser.add_argument("--max_digit", type=int, default=9)
    parser.add_argument("--min_digit_mass_for_fallback", type=float, default=0.05)
    parser.add_argument("--fallback_score", type=int, default=5)

    parser.add_argument("--output_root", type=str, default="./weighted_results")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--tag", type=str, default=None)
    parser.add_argument("--dry_run", action="store_true")
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    run_weighted_digit_inference(args)


if __name__ == "__main__":
    main()
