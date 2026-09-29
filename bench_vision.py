#!/usr/bin/env python3
"""
bench_vision.py - compare textsnap's vision-encoder variants on your images.

The q4 / q8 encoders produce slightly different image features, so
switching variants changes OCR output. This script measures, per variant:

  * vision-encoder time (the part the variant affects directly),
  * prefill / decode time and token count,
  * character error rate (CER) against a reference variant (default q4,
    the current default) and, optionally, against ground-truth text --
    the better measure, since neither variant is full precision.

Usage:
    python bench_vision.py page1.png page2.jpg ...
    python bench_vision.py scans/*.png --truth truth/   # truth/<stem>.txt

Downloads the q8 encoder (443 MB) if not yet cached. Use a
handful of images that look like your real inputs; CER on a few pages is
noisy, so treat small differences as ties.
"""
import argparse
import statistics
import sys
import time
from pathlib import Path

import textsnap as ts


def cer(hyp, ref):
    """Character error rate: Levenshtein(hyp, ref) / len(ref)."""
    if not ref:
        return 0.0 if not hyp else 1.0
    prev = list(range(len(hyp) + 1))
    for i, rc in enumerate(ref, 1):
        cur = [i] + [0] * len(hyp)
        for j, hc in enumerate(hyp, 1):
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1,
                         prev[j - 1] + (rc != hc))
        prev = cur
    return prev[-1] / len(ref)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("images", nargs="+")
    ap.add_argument("--variants", default="q4,q8",
                    help="comma-separated variants to compare")
    ap.add_argument("--reference", default="q4",
                    help="variant whose output the others are scored against")
    ap.add_argument("--truth", default=None,
                    help="directory of ground-truth <image stem>.txt files")
    ap.add_argument("--model-dir", default=None)
    ap.add_argument("--max-tokens", type=int, default=2048)
    ap.add_argument("--max-pixels", type=int, default=ts.MAX_PIXELS)
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()
    ts.VERBOSE = args.verbose

    variants = [v.strip() for v in args.variants.split(",") if v.strip()]
    for v in variants + [args.reference]:
        if v not in ts.VISION_VARIANTS:
            ap.error(f"unknown variant '{v}'")
    if args.reference not in variants:
        variants.append(args.reference)

    images = [(Path(p).stem, ts.load_from_file(Path(p))[0])
              for p in args.images]
    truth = {}
    if args.truth:
        for stem, _ in images:
            f = Path(args.truth) / f"{stem}.txt"
            if f.is_file():
                truth[stem] = f.read_text(encoding="utf-8").strip()

    results = {}   # variant -> list of (stem, text, stats)
    for v in variants:
        md = ts.get_model_dir(args.model_dir, vision=v)
        engine = ts.OCREngine(md, vision=v)
        rows = []
        for stem, img in images:
            text = engine.recognize(img, max_tokens=args.max_tokens,
                                    max_pixels=args.max_pixels)
            rows.append((stem, text, dict(engine.last_stats)))
            s = engine.last_stats
            print(f"[{v:>4}] {stem}: encoder {s['vision_s']:.2f}s, "
                  f"prefill {s['prefill_s']:.2f}s, decode {s['decode_s']:.2f}s, "
                  f"{s['tokens']} tokens ({s['stop']})", file=sys.stderr)
        results[v] = rows
        del engine

    ref = {stem: text for stem, text, _ in results[args.reference]}
    base = statistics.mean(s["vision_s"] for _, _, s in results[variants[0]])
    head = (f"{'variant':>8} {'encoder s':>10} {'vs ' + variants[0]:>9} "
            f"{'total s':>8} {'CER vs ' + args.reference:>13} {'identical':>10}")
    if truth:
        head += f" {'CER vs truth':>13}"
    print(head)
    for v in variants:
        rows = results[v]
        enc = statistics.mean(s["vision_s"] for _, _, s in rows)
        tot = statistics.mean(s["vision_s"] + s["prefill_s"] + s["decode_s"]
                              for _, _, s in rows)
        c_ref = statistics.mean(cer(t, ref[stem]) for stem, t, _ in rows)
        same = sum(t == ref[stem] for stem, t, _ in rows)
        line = (f"{v:>8} {enc:>10.2f} {base / enc:>8.2f}x {tot:>8.2f} "
                f"{c_ref:>12.2%} {same:>5}/{len(rows):<4}")
        if truth:
            scored = [cer(t, truth[stem]) for stem, t, _ in rows
                      if stem in truth]
            line += f" {statistics.mean(scored):>12.2%}"
        print(line)


if __name__ == "__main__":
    main()
