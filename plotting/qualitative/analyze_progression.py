import argparse
import json
import os
import re
import textwrap
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Analyze checkpoint progression for probabilistic free-form forecasts. "
            "Computes per-sample Brier scores, finds strictly improving questions, "
            "and highlights regressions."
        )
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default="/fast/nchandak/forecasting/evals/freeform/manual/validation-retrieval_207/analyze",
        help="Directory containing checkpoint evaluation JSONL files.",
    )
    parser.add_argument(
        "--judge",
        type=str,
        default="Qwen3_4B",
        help="Judge name used for score fields (e.g., Qwen3_4B -> score_Qwen3_4B).",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=10,
        help="Number of samples to show for strict improvements and regressions.",
    )
    parser.add_argument(
        "--max-response-chars",
        type=int,
        default=50000,
        help="Maximum characters to print per response (after trimming whitespace).",
    )
    parser.add_argument(
        "--output-file",
        type=str,
        default="analysis_progression.txt",
        help="Path to write the formatted analysis summary.",
    )
    return parser.parse_args()


def load_jsonl_file(file_path: str) -> List[Dict]:
    data = []
    with open(file_path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                data.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return data


def safe_float(value: Optional[float]) -> Optional[float]:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def calculate_brier_score(probability: float, is_correct: bool) -> float:
    if is_correct:
        return -((1 - probability) ** 2)
    return -(probability ** 2)


@dataclass
class GenerationEvaluation:
    brier: float
    is_correct: bool


def evaluate_generation(
    generation_answer, generation_scores
) -> Optional[GenerationEvaluation]:
    if isinstance(generation_answer, dict) and isinstance(generation_scores, dict):
        brier_total = 0.0
        any_correct = False
        probability_final = 1
        
        for answer_option, probability in generation_answer.items():
            probability = safe_float(probability)
            if probability is None:
                return None 
            
            probability_final = probability
            is_correct = False
            if answer_option in generation_scores:
                try:
                    is_correct = int(generation_scores[answer_option]) == 1
                except (TypeError, ValueError):
                    is_correct = bool(generation_scores[answer_option])
            if is_correct:
                any_correct = True
            brier_total += calculate_brier_score(probability, is_correct)
        if not any_correct:
            brier_total -= 1
        
        # brier_total = probability_final - 1
        return GenerationEvaluation(brier=brier_total, is_correct=any_correct)

    # if isinstance(generation_answer, str) and isinstance(generation_scores, (int, float)):
    #     is_correct = int(generation_scores) == 1
    #     brier_value = 0.0 if is_correct else -2.0
    #     return GenerationEvaluation(brier=brier_value, is_correct=is_correct)

    return None


@dataclass
class CheckpointSampleStats:
    adjusted_brier: Optional[float]
    raw_briers: List[float] = field(default_factory=list)
    is_correct: Optional[bool] = None
    valid: Optional[bool] = True
    responses: List[str] = field(default_factory=list)


@dataclass
class SampleProgression:
    question_id: str
    title: str
    background_info: str
    resolution_criteria: str
    question_text: str
    answer: Optional[str]
    checkpoints: Dict[int, CheckpointSampleStats] = field(default_factory=dict)


def compute_sample_stats(item: Dict, judge_field: str) -> CheckpointSampleStats:
    extracted_answers = item.get("extracted_answer", [])
    judge_scores = item.get(judge_field, [])

    generation_results: List[GenerationEvaluation] = []
    for gen_idx in range(min(len(extracted_answers), len(judge_scores))):
        evaluation = evaluate_generation(extracted_answers[gen_idx], judge_scores[gen_idx])
        if evaluation is not None:
            generation_results.append(evaluation)

    if not generation_results:
        return CheckpointSampleStats(adjusted_brier=None, valid=False)

    raw_briers = [result.brier for result in generation_results]
    adjusted_brier = sum(score + 1 for score in raw_briers) / len(raw_briers)
    is_correct = any(result.is_correct for result in generation_results)

    return CheckpointSampleStats(
        adjusted_brier=adjusted_brier,
        raw_briers=raw_briers,
        is_correct=is_correct,
        responses=item.get("response", []),
    )


def extract_checkpoint_number(filename: str) -> Optional[int]:
    match = re.search(r"checkpoint(\d+)", filename)
    if match:
        return int(match.group(1))
    return None


def build_progressions(input_dir: str, judge_field: str) -> Tuple[Dict[str, SampleProgression], List[int]]:
    checkpoint_files = []
    for entry in os.listdir(input_dir):
        if not entry.endswith(".jsonl"):
            continue
        checkpoint = extract_checkpoint_number(entry)
        if checkpoint is None:
            continue
        checkpoint_files.append((checkpoint, os.path.join(input_dir, entry)))

    checkpoint_files.sort()
    progressions: Dict[str, SampleProgression] = {}

    for checkpoint, file_path in checkpoint_files:
        data = load_jsonl_file(file_path)
        print(f"Loaded {len(data)} samples from checkpoint {checkpoint} ({file_path})")
        for item in data:
            sample_id = str(item.get("question_id", item.get("idx")))
            if sample_id not in progressions:
                title = item.get("question_title") or item.get("full_question") or "Unknown question"
                question_text = item.get("full_question") or item.get("question_title") or ""
                background_info = item.get("background") or "N/A"
                resolution_criteria = item.get("resolution_criteria") or "N/A"
                progressions[sample_id] = SampleProgression(
                    question_id=sample_id,
                    title=title,
                    question_text=question_text,
                    background_info=background_info,
                    resolution_criteria=resolution_criteria,
                    answer=item.get("answer"),
                )

            stats = compute_sample_stats(item, judge_field)
            progressions[sample_id].checkpoints[checkpoint] = stats

    checkpoint_order = [cp for cp, _ in checkpoint_files]
    return progressions, checkpoint_order


def is_strictly_increasing(values: List[float]) -> bool:
    return all(values[i + 1] > values[i] for i in range(len(values) - 1))


def find_strictly_improving_samples(
    progressions: Dict[str, SampleProgression],
    checkpoint_order: List[int],
    top_k: int,
) -> List[Tuple[float, SampleProgression]]:
    candidates: List[Tuple[float, SampleProgression]] = []
    for sample in progressions.values():
        scores: List[float] = []
        correctness: List[bool] = []
        valid = True 
        
        for checkpoint in checkpoint_order:
            stats = sample.checkpoints.get(checkpoint)
            if stats is None or stats.adjusted_brier is None:
                scores = []
                correctness = []
                break
            scores.append(stats.adjusted_brier)
            correctness.append(stats.is_correct)
            valid = stats.valid 
            
        if len(scores) != len(checkpoint_order):
            continue
        # if is_strictly_increasing(scores):
        # only keep those samples where the correctness is the same across all checkpoints
        if not valid: 
            continue 
        
        if all(correctness[i] == correctness[i+1] for i in range(len(correctness) - 1)):
            if not is_strictly_increasing(scores):
                continue 
            
            improvement = scores[-1] - scores[0]
            candidates.append((improvement, sample))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[:top_k]


def find_accuracy_regressions(
    progressions: Dict[str, SampleProgression],
    checkpoint_order: List[int],
    top_k: int,
) -> List[Tuple[float, SampleProgression]]:
    if not checkpoint_order:
        return []
    first_cp = checkpoint_order[0]
    candidates: List[Tuple[float, SampleProgression]] = []
    for sample in progressions.values():
        first_stats = sample.checkpoints.get(first_cp)
        if not first_stats or not first_stats.is_correct:
            continue
        became_wrong = False
        last_score = first_stats.adjusted_brier if first_stats.adjusted_brier is not None else 0.0
        for checkpoint in checkpoint_order[1:]:
            stats = sample.checkpoints.get(checkpoint)
            if not stats:
                continue
            if stats.adjusted_brier is not None:
                last_score = stats.adjusted_brier
            if stats.is_correct:
                continue
            became_wrong = True
        if became_wrong:
            delta = (first_stats.adjusted_brier or 0.0) - (last_score or 0.0)
            candidates.append((delta, sample))
    candidates.sort(key=lambda x: x[0], reverse=True)
    return candidates[:top_k]


def clean_response(text: str, max_chars: int) -> str:
    text = text.strip()
    text = re.sub(r"\s+", " ", text)
    if len(text) <= max_chars:
        return text
    return textwrap.shorten(text, width=max_chars, placeholder="...")


def print_sample_history(
    sample: SampleProgression,
    checkpoint_order: List[int],
    max_response_chars: int,
) -> List[str]:
    lines: List[str] = []
    def add(line: str = "") -> None:
        lines.append(line)

    add(f"Question {sample.question_id}: {sample.title}")
    add(f"Background: {sample.background_info}")
    add(f"Resolution Criteria: {sample.resolution_criteria}")
    if sample.answer:
        add(f"  Resolution answer: {sample.answer}")
    if sample.question_text and sample.question_text != sample.title:
        text = sample.question_text
        suffix = "..." if len(text) > 200 else ""
        add(f"  Question text: {text[:200]}{suffix}")
    for checkpoint in checkpoint_order:
        stats = sample.checkpoints.get(checkpoint)
        if not stats:
            continue
        status = "correct" if stats.is_correct else "wrong"
        score_str = "N/A" if stats.adjusted_brier is None else f"{stats.adjusted_brier:.3f}"
        add(f"    Checkpoint {checkpoint}: score={score_str}, status={status}")
        for gen_idx, response in enumerate(stats.responses):
            formatted = clean_response(response, max_chars=max_response_chars)
            add(f"      Gen {gen_idx + 1}: {formatted}")
    add("-" * 80)
    return lines


def main():
    args = parse_args()
    judge_field = f"score_{args.judge}"

    if not os.path.isdir(args.input_dir):
        raise FileNotFoundError(f"Input directory not found: {args.input_dir}")

    progressions, checkpoint_order = build_progressions(args.input_dir, judge_field)
    output_lines: List[str] = []

    def write(line: str = "") -> None:
        output_lines.append(line)
        print(line)

    write(f"Total samples tracked: {len(progressions)}")
    write(f"Checkpoints: {checkpoint_order}")

    improving = find_strictly_improving_samples(progressions, checkpoint_order, args.top_k)
    write("")
    write("=== Strictly improving samples (shifted Brier, higher is better) ===")
    if not improving:
        write("No samples found with strictly improving Brier scores across checkpoints.")
    for rank, (delta, sample) in enumerate(improving, start=1):
        write("")
        write(f"#{rank}: Δ={delta:.3f}")
        for line in print_sample_history(sample, checkpoint_order, args.max_response_chars):
            write(line)

    regressions = find_accuracy_regressions(progressions, checkpoint_order, args.top_k)
    write("")
    write("=== Regression samples (initially correct, later wrong) ===")
    if not regressions:
        write("No samples found where correctness regressed.")
    for rank, (delta, sample) in enumerate(regressions, start=1):
        write("")
        write(f"#{rank}: Δ={delta:.3f} (positive means later checkpoints scored worse)")
        for line in print_sample_history(sample, checkpoint_order, args.max_response_chars):
            write(line)

    with open(args.output_file, "w") as handle:
        handle.write("\n".join(output_lines) + "\n")
    print(f"\nSaved analysis summary to {args.output_file}")


if __name__ == "__main__":
    main()

