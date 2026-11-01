"""Blind A/B test of two rankers on a real shoot.

The decision this exists to settle: every number validating the learned ranker
measures ORDERING (64.0% agreement against 51.8%, leave-one-shoot-out). The only
real-world evidence measures SELECTION, where the legacy ranker returned 35
frames of which the photographer would have kept 34. Those are different claims,
and the second is what a user receives, so the learned ranker ships disabled
until selection is measured too.

Measuring it needs human eyes, so this makes that as cheap as possible:

  * ONLY the disagreements are exported. If both rankers keep a frame, it tells
    you nothing about which is better. On one shoot that cuts 70 frames to
    judge down to the 27 that differ.
  * The two sets are labelled A and B, assigned by a coin flip recorded in a
    file you are told not to open. Knowing which folder is the new ranker is
    enough to bias the count, in either direction.
  * Equal sizes. Both rankers return the same top_n, so the disagreement sets
    are the same size and the comparison is "how many of these 27 would you
    keep" against "how many of these 27 would you keep".

    python tools/ab_cull.py ~/Desktop/JUDGE-06-26
    python tools/ab_cull.py <folder> --top-n 35 --out ~/Desktop
    python tools/ab_cull.py --reveal          # after you have counted
"""
from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import engine  # noqa: E402

EXTS = {".jpg", ".jpeg", ".png", ".webp"}
KEYFILE = "DO-NOT-OPEN-until-counted.json"


def keepers(metrics, mode: str, top_n: int, blur: float, dupes: int, preset: str):
    saved = engine.RANKER
    try:
        engine.RANKER = mode
        res = engine.rank_metrics(metrics, blur_threshold=blur, top_n=top_n,
                                  duplicate_threshold=dupes, preset=preset)
    finally:
        engine.RANKER = saved
    return [k["filename"] for k in res["keepers"]], res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("folder", type=Path, nargs="?")
    ap.add_argument("--out", type=Path, default=Path.home() / "Desktop")
    ap.add_argument("--top-n", type=int, default=35)
    ap.add_argument("--blur", type=float, default=40.0)
    ap.add_argument("--dupes", type=int, default=2)
    ap.add_argument("--preset", default="Sports Action")
    ap.add_argument("--reveal", action="store_true",
                    help="print which folder was which, after counting")
    a = ap.parse_args()

    out_a = a.out / "CULL-TEST-A"
    out_b = a.out / "CULL-TEST-B"
    key_path = a.out / KEYFILE

    if a.reveal:
        if not key_path.exists():
            sys.exit(f"no key at {key_path}; run the test first")
        k = json.loads(key_path.read_text())
        print(f"\nfolder A = {k['A']}")
        print(f"folder B = {k['B']}")
        print(f"\n{k['n']} frames each, from {k['folder']}")
        print("\nMore keepers in the 'v2' folder means the learned ranker")
        print("selects better as well as orders better, and is safe to enable")
        print("with CLUTCHCULL_RANKER=v2. Roughly equal means ordering is the")
        print("only gain and it is a judgement call. Fewer means leave it off.")
        return

    if not a.folder or not a.folder.is_dir():
        sys.exit("pass a shoot folder, or --reveal to see the answer")

    paths = sorted(p for p in a.folder.iterdir() if p.suffix.lower() in EXTS)
    if not paths:
        sys.exit(f"no images in {a.folder}")
    print(f"scoring {len(paths)} frames with both rankers ...")
    metrics = engine.compute_metrics_batch(paths)

    lg, lg_res = keepers(metrics, "legacy", a.top_n, a.blur, a.dupes, a.preset)
    v2, v2_res = keepers(metrics, "v2", a.top_n, a.blur, a.dupes, a.preset)
    # Sanity: the gate must behave identically, otherwise this compares two
    # different filters rather than two rankings and the result means nothing.
    if lg_res["blurry_removed"] != v2_res["blurry_removed"]:
        print(f"  WARNING gate differs: legacy rejected {lg_res['blurry_removed']}, "
              f"v2 rejected {v2_res['blurry_removed']}. Not a clean ranker test.")

    only_lg = [n for n in lg if n not in set(v2)]
    only_v2 = [n for n in v2 if n not in set(lg)]
    shared = len(set(lg) & set(v2))
    print(f"\nboth rankers kept {shared} of {a.top_n} frames; those are skipped.")
    print(f"they disagree on {len(only_lg)} vs {len(only_v2)} frames.")
    if not only_lg or not only_v2:
        print("\nNo disagreement to judge. Nothing to test.")
        return

    by_name = {p.name: p for p in paths}
    flip = random.Random().random() < 0.5
    sets = {"A": only_v2, "B": only_lg} if flip else {"A": only_lg, "B": only_v2}
    labels = {"A": "v2 (learned)", "B": "legacy (current)"} if flip else \
             {"A": "legacy (current)", "B": "v2 (learned)"}

    for tag, dest in (("A", out_a), ("B", out_b)):
        if dest.exists():
            shutil.rmtree(dest)
        dest.mkdir(parents=True)
        # Numbered so they open in a stable order, but the NUMBER is just the
        # position in that ranker's list and carries no quality claim across
        # the two folders.
        for i, name in enumerate(sets[tag], 1):
            src = by_name.get(name)
            if src:
                shutil.copy2(src, dest / f"{i:02d}_{name}")
        print(f"  {dest.name}: {len(sets[tag])} frames")

    key_path.write_text(json.dumps({
        "A": labels["A"], "B": labels["B"],
        "n": len(only_lg), "folder": str(a.folder),
    }, indent=2))

    print(f"""
HOW TO JUDGE THIS

  1. Open {out_a.name} and count how many you would actually keep.
  2. Open {out_b.name} and do the same.
  3. Tell me the two counts, then run:
         python tools/ab_cull.py --reveal

Do not open {KEYFILE} until you have both counts. Knowing which folder is
the new ranker is enough to shift the count without you noticing.

Both folders contain frames that passed the sharpness gate, so none are
blurry. The question is not "is this acceptable" but "would this make my
post", which is the decision the ranker is trying to predict.
""")


if __name__ == "__main__":
    main()
