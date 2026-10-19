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
