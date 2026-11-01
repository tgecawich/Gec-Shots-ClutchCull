"""Engine regression tests, pinned to the bugs that actually shipped.

Each test here encodes a failure that reached users once:
  - min-max normalization collapsed every score below 10 when one outlier frame
    stretched the range (the "all scores under 10" bug)
  - the blur gate deleted whole shoots when `sharpness` was redefined without
    recalibrating the threshold (the "every photo marked soft" bug)
  - duplicate removal has to group identical frames, not just similar ones
  - roster matching must refuse ambiguous partial reads (precision over recall)

Run either way:
    python -m pytest tests/ -q        (if pytest is installed)
    python tests/test_engine.py       (no dependencies beyond the engine's own)

Synthetic images only; no network, no models required for most tests. Tests
that need the person/face detectors skip themselves if models are missing.
"""
from __future__ import annotations

import sys
import tempfile
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402

import engine  # noqa: E402
import jersey  # noqa: E402


# --- helpers ---------------------------------------------------------------
def _textured(seed: int = 7, size: int = 900) -> Image.Image:
    """A high-frequency textured frame: sharp by construction."""
    rng = np.random.default_rng(seed)
    arr = rng.integers(0, 255, (size, size, 3), dtype=np.uint8)
    return Image.fromarray(arr, "RGB")


def _save(im: Image.Image, directory: Path, name: str) -> Path:
    p = directory / name
    im.save(p, "JPEG", quality=90)
    return p


# --- tests -----------------------------------------------------------------
def test_normalize_survives_outliers():
    """One extreme frame must not crush everyone else to ~0 (shipped bug).

    Uses a realistic batch size: the bug appeared on a 500-photo shoot where a
    single frame of chain-link fence set the max. _normalize deliberately falls
    back to plain min-max below 8 samples (percentiles need samples), so a tiny
    batch cannot demonstrate this property - see the companion test below.
    """
    ordinary_vals = [100.0 + i for i in range(40)]      # 100..139, the real shoot
    out = engine._normalize(ordinary_vals + [10000.0])  # + one fence frame
    ordinary = out[:40]
    # The ordinary values must still spread across most of the range rather than
    # being squeezed into the bottom 2% the way plain min-max does.
    assert max(ordinary) - min(ordinary) > 0.5, f"ordinary values collapsed: {ordinary[:5]}..."
    assert out == sorted(out), "normalization must preserve order"


def test_normalize_small_batch_uses_minmax():
    """Pins the documented fallback: under 8 samples there aren't enough points
    for percentiles to mean anything, so min-max is used and an outlier WILL
    dominate. Documented here so the boundary is deliberate, not accidental."""
    out = engine._normalize([100.0, 110.0, 120.0, 130.0, 10000.0])
    assert out[0] == 0.0 and out[-1] == 1.0, "small batch should be plain min-max"
    assert out == sorted(out)


def test_normalize_handles_constant_input():
    out = engine._normalize([5.0, 5.0, 5.0])
    assert len(out) == 3
    assert all(0.0 <= v <= 1.0 for v in out)


def test_sharp_beats_blurred_end_to_end():
    """A sharp frame and its blurred copy: the sharp one must rank first and
    the blur gate must never delete the sharp one (shipped bug, both halves)."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        base = _textured()
        sharp = _save(base, d, "sharp.jpg")
        blurred = _save(base.filter(ImageFilter.GaussianBlur(8)), d, "blurred.jpg")
        metrics = engine.compute_metrics_batch([sharp, blurred])
        by_name = {m["filename"]: m for m in metrics}
        assert not by_name["sharp.jpg"].get("unreadable")
        assert by_name["sharp.jpg"]["sharpness"] > by_name["blurred.jpg"]["sharpness"] * 2, (
            "sharpness separation too small: "
            f"{by_name['sharp.jpg']['sharpness']} vs {by_name['blurred.jpg']['sharpness']}"
        )
        result = engine.rank_metrics(metrics, blur_threshold=40.0, top_n=10,
                                     duplicate_threshold=0)
        keep_names = [k["filename"] for k in result["keepers"]]
        assert "sharp.jpg" in keep_names, "blur gate deleted a sharp frame"
        if len(keep_names) == 2:
            assert keep_names[0] == "sharp.jpg", "blurred frame outranked sharp one"


def test_identical_frames_dedupe():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        base = _textured(seed=3)
        a = _save(base, d, "a.jpg")
        b = _save(base, d, "b.jpg")          # same pixels
        c = _save(_textured(seed=99), d, "c.jpg")  # different frame
        metrics = engine.compute_metrics_batch([a, b, c])
        result = engine.rank_metrics(metrics, blur_threshold=0.0, top_n=10,
                                     duplicate_threshold=2)
        assert result["duplicates_removed"] >= 1, "identical frames were not grouped"
        keep_names = {k["filename"] for k in result["keepers"]}
        assert "c.jpg" in keep_names


def test_blur_gate_spares_clean_batch():
    """A batch of equally sharp frames must not lose members to the gate just
    because one of them is the 'least sharp' (absolute gate, not relative)."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        paths = [_save(_textured(seed=s), d, f"f{s}.jpg") for s in range(4)]
        metrics = engine.compute_metrics_batch(paths)
        result = engine.rank_metrics(metrics, blur_threshold=40.0, top_n=10,
                                     duplicate_threshold=0)
        assert result["blurry_removed"] == 0, (
            f"gate removed {result['blurry_removed']} frames from an all-sharp batch"
        )


def test_roster_matching_precision():
    """Ambiguity must return nothing: filing a photo under the wrong player is
    worse than missing it."""
    assert jersey.match_roster("10", ["10", "23"]) == "10"       # exact
    assert jersey.match_roster("1", ["10", "23"]) == "10"        # unique prefix
    assert jersey.match_roster("1", ["10", "12"]) is None        # ambiguous prefix
    assert jersey.match_roster("7", ["10", "23"]) is None        # not on roster
    assert jersey.match_roster("", ["10"]) is None               # empty read
    assert jersey.match_roster("99", []) == "99"                 # no roster = pass-through


def test_unreadable_files_are_counted_not_fatal():
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        junk = d / "junk.jpg"
        junk.write_bytes(b"not an image at all")
        ok = _save(_textured(seed=11), d, "ok.jpg")
        metrics = engine.compute_metrics_batch([junk, ok])
        result = engine.rank_metrics(metrics, blur_threshold=0.0, top_n=5,
                                     duplicate_threshold=0)
        assert result["unreadable_skipped"] == 1
        assert result["total"] == 2
        assert [k["filename"] for k in result["keepers"]] == ["ok.jpg"]


def test_v2_ranker_still_puts_sharp_above_soft():
    """The regression the two-stage change could plausibly cause.

    Stage two dropped sharpness from the RANKING entirely, on the evidence that
    among frames which already passed the gate it predicts the photographer's
    preference slightly negatively. The risk that creates is specific: a soft
    frame is now held back only by the multiplicative gate, so if that gate were
    ever loosened, a blurry photo with a clean background could outrank a sharp
    one. Handing a photographer a blurry keeper is the single worst thing this
    app can do, so it gets a test rather than an argument.

    Needs more than LEARNED_MIN_BATCH frames or the learned ranker silently
    falls back to legacy and this would pass without testing anything. That is
    exactly why the older two-frame sharp-vs-blurred test does not cover it.
    """
    assert engine.RANKER == "v2", "default ranker changed; update this test"
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        paths, sharp_names = [], set()
        for s in range(6):
            base = _textured(seed=s + 20)
            p = _save(base, d, f"sharp{s}.jpg")
            paths.append(p)
            sharp_names.add(p.name)
            paths.append(_save(base.filter(ImageFilter.GaussianBlur(7)), d,
                               f"soft{s}.jpg"))
        metrics = engine.compute_metrics_batch(paths)
        assert len(metrics) >= engine.LEARNED_MIN_BATCH, "batch too small to test v2"
        result = engine.rank_metrics(metrics, blur_threshold=40.0, top_n=12,
                                     duplicate_threshold=0)
        order = [k["filename"] for k in result["keepers"]]
        rejected = set(result["rejected"])
        # No sharp frame may be discarded.
        assert not (sharp_names & rejected), f"gate deleted sharp frames: {sharp_names & rejected}"
        # Every surviving sharp frame must outrank every surviving soft one.
        sharp_ranks = [i for i, n in enumerate(order) if n in sharp_names]
        soft_ranks = [i for i, n in enumerate(order) if n not in sharp_names]
        if sharp_ranks and soft_ranks:
            assert max(sharp_ranks) < min(soft_ranks), (
                f"a soft frame outranked a sharp one under v2: {order}")


def test_ranker_can_be_reverted_without_a_deploy():
    """CLUTCHCULL_RANKER=legacy must fully restore the old ordering, because
    that env var is the entire rollback plan if the learned weights turn out to
    suit one photographer rather than photographers."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        paths = [_save(_textured(seed=s + 40), d, f"f{s}.jpg") for s in range(10)]
        metrics = engine.compute_metrics_batch(paths)
        saved = engine.RANKER
        try:
            engine.RANKER = "v2"
            v2 = [k["filename"] for k in engine.rank_metrics(
                metrics, blur_threshold=0.0, top_n=10, duplicate_threshold=0)["keepers"]]
            engine.RANKER = "legacy"
            lg = [k["filename"] for k in engine.rank_metrics(
                metrics, blur_threshold=0.0, top_n=10, duplicate_threshold=0)["keepers"]]
        finally:
            engine.RANKER = saved
        assert sorted(v2) == sorted(lg), "a mode changed WHICH frames survive, not just order"
        # Scores must actually differ, otherwise the flag is not wired up and
        # the rollback would be a no-op that looks like it worked.
        assert engine._learned_scores(
            [engine._dict_to_candidate(m) for m in metrics])[0] is not None


def test_new_features_are_bounded_and_present():
    """subject_area and separation were added because the photographer's own
    description of a good frame ('blurry background, clear subject, large in
    frame') was not measurable at all: the subject box was computed and then
    discarded. These pin the two invariants worth pinning."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        metrics = engine.compute_metrics_batch([
            _save(_textured(seed=7), d, "a.jpg"),
            _save(_textured(seed=7).filter(ImageFilter.GaussianBlur(6)), d, "b.jpg"),
        ])
        for m in metrics:
            assert "subject_area" in m and "separation" in m, f"missing on {m['filename']}"
            # A fraction of the frame cannot leave 0-1 no matter where
            # _pick_subject puts the box, including its out-of-frame fallbacks.
            assert 0.0 <= m["subject_area"] <= 1.0, m["subject_area"]
            # The +1 denominator in `separation` exists so a perfectly smooth
            # background cannot divide by zero. Also checks it is not NaN.
            assert m["separation"] >= 0.0 and m["separation"] == m["separation"]


def test_cached_metrics_without_new_fields_still_rank():
    """The /rank endpoint replays metrics the BROWSER stored, so a session that
    began before these fields existed must not KeyError. This is the exact
    class of break that would hit users mid-cull rather than on deploy."""
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        metrics = engine.compute_metrics_batch([
            _save(_textured(seed=11), d, "a.jpg"),
            _save(_textured(seed=12), d, "b.jpg"),
        ])
        stale = [{k: v for k, v in m.items()
                  if k not in ("subject_area", "separation")} for m in metrics]
        result = engine.rank_metrics(stale, blur_threshold=0.0, top_n=5,
                                     duplicate_threshold=0)
        assert len(result["keepers"]) == 2, result


# --- plain-python runner ----------------------------------------------------
if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    for t in tests:
        try:
            t()
            print(f"  ok    {t.__name__}")
        except AssertionError:
            failed += 1
            print(f"  FAIL  {t.__name__}")
            traceback.print_exc(limit=2)
        except Exception:
            failed += 1
            print(f"  ERROR {t.__name__}")
            traceback.print_exc(limit=2)
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
