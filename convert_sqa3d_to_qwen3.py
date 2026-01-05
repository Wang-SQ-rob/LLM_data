#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert SQA3D to Qwen3 format (with 3D relative geometry).

REQUIRES:
- ScanNet scene data (or preprocessed object positions)

Author: you
"""

import json
import argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation as R


# =====================
# Placeholder functions
# =====================

def get_object_world_position(scene_id, answer_text):
    """
    USER MUST IMPLEMENT THIS.

    Given a scene_id and answer text (e.g. 'picture'),
    return the object's world position (x, y, z).
    """
    raise NotImplementedError(
        "You must implement object lookup using ScanNet annotations."
    )


def compute_relative_position(agent_pos, agent_quat, obj_pos):
    """
    Convert world position to agent-relative coordinates.
    """
    agent_pos = np.array(agent_pos)
    obj_pos = np.array(obj_pos)

    r = R.from_quat(agent_quat)
    rel = r.inv().apply(obj_pos - agent_pos)
    return rel.tolist()


# =====================
# Main
# =====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--base_dir", required=True)
    parser.add_argument("--split", choices=["train", "val", "test"], required=True)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    base = Path(args.base_dir)

    q_path = base / f"v1_balanced_questions_{args.split}_scannetv2.json"
    a_path = base / f"v1_balanced_sqa_annotations_{args.split}_scannetv2.json"

    out_path = (
        Path(args.output)
        if args.output
        else base / f"sqa3d_qwen3_{args.split}.json"
    )

    questions = json.load(open(q_path))["questions"]
    annotations = json.load(open(a_path))["annotations"]

    ann_map = {int(a["question_id"]): a for a in annotations}

    output = []

    for q in questions:
        qid = int(q["question_id"])
        ann = ann_map[qid]

        answer = ann["answers"][0]["answer"]

        agent_pos = [
            ann["position"]["x"],
            ann["position"]["y"],
            ann["position"]["z"],
        ]

        agent_quat = [
            ann["rotation"]["_w"],
            ann["rotation"]["_x"],
            ann["rotation"]["_y"],
            ann["rotation"]["_z"],
        ]

        # ===== Requires ScanNet =====
        obj_world_pos = get_object_world_position(q["scene_id"], answer)
        rel_pos = compute_relative_position(agent_pos, agent_quat, obj_world_pos)

        sample = {
            "id": f"sqa3d_{qid}",
            "scene_id": q["scene_id"],
            "conversations": [
                {
                    "from": "user",
                    "value": f"Situation: {q['situation']}\nQuestion: {q['question']}"
                },
                {
                    "from": "assistant",
                    "value": answer
                }
            ],
            "agent": {
                "position": [0.0, 0.0, 0.0],
                "rotation_quat": agent_quat
            },
            "object": {
                "relative_position": rel_pos
            }
        }

        output.append(sample)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(output)} samples to {out_path}")


if __name__ == "__main__":
    main()
