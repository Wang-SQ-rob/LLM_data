#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert SQA3D balanced split to Qwen2.5 training format (text-only).

This script converts:
- v1_balanced_questions_{split}_scannetv2.json
- v1_balanced_sqa_annotations_{split}_scannetv2.json

into a Qwen2.5-style JSON file:
- sqa3d_qwen25_{split}.json

Note:
- This conversion does NOT require ScanNet 3D scene data.
- Only textual question, situation, and answer are used.
"""

import json
import argparse
from pathlib import Path


# =====================
# ARGPARSE
# =====================
def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert SQA3D balanced dataset to Qwen2.5 format"
    )

    parser.add_argument(
        "--base_dir",
        type=str,
        required=True,
        help="Path to SQA3D sqa_task/balanced directory"
    )

    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split to convert"
    )

    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output JSON path"
    )

    return parser.parse_args()


# =====================
# MAIN
# =====================
def main():
    args = parse_args()

    base_dir = Path(args.base_dir)

    questions_path = base_dir / f"v1_balanced_questions_{args.split}_scannetv2.json"
    annotations_path = base_dir / f"v1_balanced_sqa_annotations_{args.split}_scannetv2.json"

    if not questions_path.exists():
        raise FileNotFoundError(f"Questions file not found: {questions_path}")
    if not annotations_path.exists():
        raise FileNotFoundError(f"Annotations file not found: {annotations_path}")

    output_path = (
        Path(args.output)
        if args.output is not None
        else base_dir / f"sqa3d_qwen25_{args.split}.json"
    )

    print(f"[INFO] Loading questions from: {questions_path}")
    questions_json = json.load(open(questions_path, "r", encoding="utf-8"))

    print(f"[INFO] Loading annotations from: {annotations_path}")
    annotations_json = json.load(open(annotations_path, "r", encoding="utf-8"))

    questions = questions_json["questions"]
    annotations = annotations_json["annotations"]

    # question_id -> annotation
    ann_map = {int(a["question_id"]): a for a in annotations}

    qwen_data = []
    missing = 0

    for q in questions:
        qid = int(q["question_id"])
        ann = ann_map.get(qid)

        if ann is None:
            missing += 1
            continue

        # -------- user input --------
        situation = q.get("situation", "")
        question_text = q.get("question", "")

        user_text = (
            f"Situation: {situation}\n"
            f"Question: {question_text}"
        )

        # -------- assistant output --------
        answers = ann.get("answers", [])
        if not answers:
            missing += 1
            continue

        # Standard practice: take the first answer
        answer_text = answers[0]["answer"]

        sample = {
            "id": f"sqa3d_{qid}",
            "scene_id": q.get("scene_id"),
            "conversations": [
                {
                    "from": "user",
                    "value": user_text
                },
                {
                    "from": "assistant",
                    "value": answer_text
                }
            ]
        }

        qwen_data.append(sample)

    print("========== SUMMARY ==========")
    print(f"Split            : {args.split}")
    print(f"Total questions  : {len(questions)}")
    print(f"Converted samples: {len(qwen_data)}")
    print(f"Skipped samples  : {missing}")
    print(f"Output file      : {output_path}")
    print("=============================")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(qwen_data, f, ensure_ascii=False, indent=2)

    print("[DONE] Conversion finished successfully.")


if __name__ == "__main__":
    main()
