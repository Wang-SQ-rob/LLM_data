#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Convert RoboAfford bbox-style ShareGPT JSON (top-level list) into:
- Qwen2.5-VL: absolute bbox ints (rounded half-up)
- Qwen3-VL  : norm_1000 bbox ints in [0, 1000) (default clamp to 999)

Output format (LLaMA-Factory multimodal ShareGPT):
{
  "id": "...",
  "images": ["rel/path.jpg"],
  "conversations": [{"from":"human","value":"<image> ..."}, {"from":"gpt","value":"..."}]
}

Pretty output is DEFAULT. Use --compact to output compact JSON.

Optional:
- --abs_clamp: clamp absolute coords for qwen25 (none|wh|wh-1)
- --skip_missing_images: skip records whose image missing/unreadable (drop record)
- --skip_log: write skipped records to TSV (id\timage\treason)
"""

import argparse
import json
import os
import re
import sys
from typing import Dict, Tuple, Optional, Iterable, Any, List

try:
    import ijson  # streaming json
except Exception:
    ijson = None

try:
    from PIL import Image, ImageFile
    ImageFile.LOAD_TRUNCATED_IMAGES = True
except Exception:
    Image = None


BOX4_RE = re.compile(
    r"\(\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*,\s*([-+]?\d*\.?\d+)\s*\)"
)


def round_half_up(x: float) -> int:
    # avoid bankers rounding for .5
    return int(x + 0.5) if x >= 0 else int(x - 0.5)


def clamp(v: int, lo: int, hi: int) -> int:
    if v < lo:
        return lo
    if v > hi:
        return hi
    return v


def normalize_1000(v_int: int, denom: int, clamp_max: int = 999) -> int:
    # round(v/denom*1000), then clamp to [0, clamp_max]
    if denom <= 0:
        return 0
    n = round_half_up(v_int / denom * 1000.0)
    return clamp(n, 0, clamp_max)


def iter_items(path: str) -> Iterable[Dict[str, Any]]:
    if ijson is None:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for obj in data:
            yield obj
    else:
        with open(path, "rb") as f:
            for obj in ijson.items(f, "item"):
                yield obj


def get_rel_image(obj: Dict[str, Any]) -> Optional[str]:
    rel = obj.get("image")
    if rel is None:
        imgs = obj.get("images")
        if isinstance(imgs, list) and imgs:
            rel = imgs[0]
    return rel


def get_image_wh(image_root: str, rel_path: str, cache: Dict[str, Tuple[int, int]]) -> Tuple[int, int]:
    if rel_path in cache:
        return cache[rel_path]
    if Image is None:
        raise RuntimeError("PIL not available. Install pillow: pip install pillow")
    p = os.path.join(image_root, rel_path)
    if not os.path.exists(p):
        raise FileNotFoundError(f"image not found: {p}")
    with Image.open(p) as im:
        w, h = im.size
    cache[rel_path] = (w, h)
    return w, h


def replace_bboxes_in_text(
    text: str,
    wh: Optional[Tuple[int, int]],
    mode: str,
    clamp_norm_max: int = 999,
    abs_clamp: str = "none",  # none|wh|wh-1
) -> str:
    if "(" not in text:
        return text

    W = H = None
    if wh is not None:
        W, H = wh

    def repl(m: re.Match) -> str:
        x1f, y1f, x2f, y2f = map(float, m.groups())
        x1 = round_half_up(x1f)
        y1 = round_half_up(y1f)
        x2 = round_half_up(x2f)
        y2 = round_half_up(y2f)

        # enforce ordering
        if x2 < x1:
            x1, x2 = x2, x1
        if y2 < y1:
            y1, y2 = y2, y1

        # optional clamp for qwen25 (and also affects qwen3 since we normalize after rounding)
        if wh is not None and abs_clamp != "none":
            if abs_clamp == "wh":
                x1 = clamp(x1, 0, W)
                y1 = clamp(y1, 0, H)
                x2 = clamp(x2, 0, W)
                y2 = clamp(y2, 0, H)
            elif abs_clamp == "wh-1":
                x1 = clamp(x1, 0, max(0, W - 1))
                y1 = clamp(y1, 0, max(0, H - 1))
                x2 = clamp(x2, 0, max(0, W - 1))
                y2 = clamp(y2, 0, max(0, H - 1))

        if mode == "qwen25":
            return f"({x1}, {y1}, {x2}, {y2})"

        # qwen3: normalize to [0, 1000)
        if wh is None:
            raise RuntimeError("qwen3 requires wh; use --wh_mode image or fixed.")
        nx1 = normalize_1000(x1, W, clamp_norm_max)
        ny1 = normalize_1000(y1, H, clamp_norm_max)
        nx2 = normalize_1000(x2, W, clamp_norm_max)
        ny2 = normalize_1000(y2, H, clamp_norm_max)
        return f"({nx1}, {ny1}, {nx2}, {ny2})"

    return BOX4_RE.sub(repl, text)


def _write_array_stream(
    out_path: str,
    items: Iterable[Dict[str, Any]],
    compact: bool,
    indent: int,
):
    """
    Streaming JSON array writer.
    Default pretty:
    [
      { ... },
      { ... }
    ]
    """
    with open(out_path, "w", encoding="utf-8") as f:
        if compact:
            f.write("[")
            first = True
            for item in items:
                if not first:
                    f.write(",\n")
                first = False
                f.write(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
            f.write("]\n")
            return

        # pretty
        arr_prefix = " " * indent
        f.write("[\n")
        first = True
        for item in items:
            if not first:
                f.write(",\n")
            first = False
            s = json.dumps(item, ensure_ascii=False, indent=indent)
            s = arr_prefix + s.replace("\n", "\n" + arr_prefix)
            f.write(s)
        f.write("\n]\n")


def do_stats(input_json: str, n: int):
    scanned = 0
    max_x2 = None
    max_y2 = None
    cnt_boxes = 0

    for obj in iter_items(input_json):
        scanned += 1
        for m in obj.get("conversations", []):
            t = m.get("value", "")
            for mm in BOX4_RE.finditer(t):
                x1f, y1f, x2f, y2f = map(float, mm.groups())
                x2 = round_half_up(x2f)
                y2 = round_half_up(y2f)
                max_x2 = x2 if max_x2 is None else max(max_x2, x2)
                max_y2 = y2 if max_y2 is None else max(max_y2, y2)
                cnt_boxes += 1

        if scanned >= n:
            break

    print(f"[stats] scanned={scanned}")
    print(f"[stats] boxes_found={cnt_boxes}")
    print(f"[stats] max_x2={max_x2} max_y2={max_y2}")
    if ijson is None:
        print("[warn] ijson not installed; stats used full json.load (may be memory-heavy).")
        print("       Recommended: pip install ijson")


def do_convert(
    input_json: str,
    out_qwen25: str,
    out_qwen3: str,
    wh_mode: str,
    fixed_wh: Tuple[int, int],
    image_root: str,
    clamp_norm_max: int,
    abs_clamp: str,
    skip_missing_images: bool,
    skip_log: str,
    compact: bool,
    indent: int,
):
    if wh_mode == "image" and not image_root:
        raise ValueError("--wh_mode image requires --image_root")
    if wh_mode == "image" and Image is None:
        raise RuntimeError("PIL not available. Install pillow: pip install pillow")

    wh_cache: Dict[str, Tuple[int, int]] = {}
    skipped: List[str] = []

    def mark_skip(obj, rel_img, reason: str):
        skipped.append(f"{obj.get('id')}\t{rel_img or ''}\t{reason}")

    def gen_out(mode: str) -> Iterable[Dict[str, Any]]:
        for obj in iter_items(input_json):
            rel_img = get_rel_image(obj)
            if rel_img is None:
                if skip_missing_images:
                    mark_skip(obj, None, "missing_image_field")
                    continue
                raise ValueError(f"missing image field in obj id={obj.get('id')}")

            # wh
            if wh_mode == "fixed":
                wh = fixed_wh
            else:
                try:
                    wh = get_image_wh(image_root, rel_img, wh_cache)
                except Exception as e:
                    if skip_missing_images:
                        mark_skip(obj, rel_img, f"image_open_failed:{type(e).__name__}")
                        continue
                    raise

            # build output
            new_obj = {"id": obj.get("id"), "images": [rel_img], "conversations": []}
            for m in obj.get("conversations", []):
                val = m.get("value", "")
                new_val = replace_bboxes_in_text(
                    val,
                    wh=wh,  # bbox: qwen3 needs wh; abs_clamp also needs wh
                    mode=mode,
                    clamp_norm_max=clamp_norm_max,
                    abs_clamp=abs_clamp if mode == "qwen25" else abs_clamp,  # affects rounding/clamp before norm too
                )
                new_obj["conversations"].append({"from": m.get("from"), "value": new_val})
            yield new_obj

    _write_array_stream(out_qwen25, gen_out("qwen25"), compact=compact, indent=indent)
    _write_array_stream(out_qwen3,  gen_out("qwen3"),  compact=compact, indent=indent)

    print("[ok] wrote:")
    print("  ", out_qwen25)
    print("  ", out_qwen3)

    if skip_missing_images:
        print(f"[ok] skipped_records={len(skipped)}")
        if skip_log:
            with open(skip_log, "w", encoding="utf-8") as f:
                f.write("id\timage\treason\n")
                for line in skipped:
                    f.write(line + "\n")
            print(f"[ok] wrote skip_log: {skip_log}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["stats", "convert"])
    ap.add_argument("--input_json", required=True)
    ap.add_argument("--stats_n", type=int, default=3000)

    ap.add_argument("--out_qwen25", default="")
    ap.add_argument("--out_qwen3", default="")

    ap.add_argument("--wh_mode", choices=["fixed", "image"], default="fixed")
    ap.add_argument("--fixed_wh", nargs=2, type=int, default=[640, 480])
    ap.add_argument("--image_root", default="")

    ap.add_argument("--norm_clamp_max", type=int, default=999)

    ap.add_argument("--abs_clamp", choices=["none", "wh", "wh-1"], default="none")

    ap.add_argument("--skip_missing_images", action="store_true")
    ap.add_argument("--skip_log", default="")

    # pretty by default
    ap.add_argument("--compact", action="store_true", help="output compact JSON (default: pretty)")
    ap.add_argument("--indent", type=int, default=2, help="pretty indent (default: 2)")

    args = ap.parse_args()

    if args.mode == "stats":
        do_stats(args.input_json, args.stats_n)
        return

    if not args.out_qwen25 or not args.out_qwen3:
        print("Error: --out_qwen25 and --out_qwen3 are required for --mode convert", file=sys.stderr)
        sys.exit(2)

    do_convert(
        input_json=args.input_json,
        out_qwen25=args.out_qwen25,
        out_qwen3=args.out_qwen3,
        wh_mode=args.wh_mode,
        fixed_wh=(args.fixed_wh[0], args.fixed_wh[1]),
        image_root=args.image_root,
        clamp_norm_max=args.norm_clamp_max,
        abs_clamp=args.abs_clamp,
        skip_missing_images=args.skip_missing_images,
        skip_log=args.skip_log,
        compact=args.compact,
        indent=args.indent,
    )


if __name__ == "__main__":
    main()
