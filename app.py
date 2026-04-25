import json
import shutil
import zipfile
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import imagehash
import numpy as np
import streamlit as st
from PIL import Image

INPUT_DIR = Path("input_photos")
OUTPUT_DIR = Path("output_photos")
CANVAS_DIR = Path("canvas_photos")
USAGE_STATS_PATH = Path("usage_stats.json")
VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}
RESAMPLING = getattr(Image, "Resampling", Image)
SCORING_PRESETS = {
    "Sports Action": {
        "sharpness": 0.6,
        "detail": 0.25,
        "contrast": 0.1,
        "exposure": 0.05,
    },
    "Portraits": {
        "sharpness": 0.25,
        "detail": 0.15,
        "contrast": 0.3,
        "exposure": 0.3,
    },
    "Events": {
        "sharpness": 0.35,
        "detail": 0.2,
        "contrast": 0.2,
        "exposure": 0.25,
    },
    "Balanced": {
        "sharpness": 0.55,
        "detail": 0.2,
        "contrast": 0.15,
        "exposure": 0.1,
    },
}


def default_user_stats() -> dict:
    return {
        "sessions": 0,
        "photos_processed": 0,
        "exports": 0,
        "minutes_saved": 0.0,
    }


def default_usage_stats() -> dict:
    return {
        "total_users": 0,
        "total_sessions": 0,
        "total_photos_processed": 0,
        "total_exports": 0,
        "total_minutes_saved": 0.0,
        "users": {},
    }


def safe_number(value, default=0):
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return value
    return default


def normalize_email(email: str) -> str:
    return email.strip().lower()


def clean_usage_stats(stats: dict | None) -> dict:
    clean_stats = default_usage_stats()

    if not isinstance(stats, dict):
        return clean_stats

    for key in clean_stats:
        if key == "users":
            continue
        clean_stats[key] = safe_number(stats.get(key), clean_stats[key])

    users = stats.get("users", {})
    if isinstance(users, dict):
        cleaned_users = {}

        for email, user_stats in users.items():
            if not isinstance(email, str) or not isinstance(user_stats, dict):
                continue

            normalized_email = normalize_email(email)
            cleaned_user_stats = default_user_stats()
            for key in cleaned_user_stats:
                cleaned_user_stats[key] = safe_number(
                    user_stats.get(key),
                    cleaned_user_stats[key],
                )

            if normalized_email:
                cleaned_users[normalized_email] = cleaned_user_stats

        clean_stats["users"] = cleaned_users

    clean_stats["total_users"] = len(clean_stats["users"])
    return clean_stats


def load_usage_stats() -> dict:
    if not USAGE_STATS_PATH.exists():
        stats = default_usage_stats()
        save_usage_stats(stats)
        return stats

    try:
        with open(USAGE_STATS_PATH, "r", encoding="utf-8") as stats_file:
            return clean_usage_stats(json.load(stats_file))
    except (OSError, json.JSONDecodeError):
        return default_usage_stats()


def save_usage_stats(stats: dict) -> None:
    try:
        with open(USAGE_STATS_PATH, "w", encoding="utf-8") as stats_file:
            json.dump(clean_usage_stats(stats), stats_file, indent=2)
    except (OSError, TypeError):
        pass


def get_or_create_user_stats(stats: dict, email: str) -> dict:
    users = stats.setdefault("users", {})
    if email not in users:
        users[email] = default_user_stats()
        stats["total_users"] = len(users)
    return users[email]


def update_session_stats(email: str) -> dict:
    stats = load_usage_stats()

    if not st.session_state.get("usage_global_session_counted", False):
        stats["total_sessions"] += 1
        st.session_state.usage_global_session_counted = True

    if email:
        counted_emails = list(st.session_state.get("usage_session_emails_counted", []))
        if email not in counted_emails:
            user_stats = get_or_create_user_stats(stats, email)
            user_stats["sessions"] += 1
            counted_emails.append(email)
            st.session_state.usage_session_emails_counted = counted_emails

    save_usage_stats(stats)
    return stats


def update_photos_processed_stats(email: str, photo_count: int) -> dict:
    stats = load_usage_stats()
    stats["total_photos_processed"] += photo_count

    if email:
        user_stats = get_or_create_user_stats(stats, email)
        user_stats["photos_processed"] += photo_count

    save_usage_stats(stats)
    return stats


def calculate_minutes_saved(photos_processed: int, seconds_per_photo: int) -> float:
    manual_minutes = photos_processed * seconds_per_photo / 60
    ai_minutes = manual_minutes * 0.25
    return manual_minutes - ai_minutes


def update_export_stats(
    email: str,
    photos_processed: int,
    seconds_per_photo: int,
) -> dict:
    stats = load_usage_stats()
    minutes_saved = calculate_minutes_saved(photos_processed, seconds_per_photo)

    stats["total_exports"] += 1
    stats["total_minutes_saved"] += minutes_saved

    if email:
        user_stats = get_or_create_user_stats(stats, email)
        user_stats["exports"] += 1
        user_stats["minutes_saved"] += minutes_saved

    save_usage_stats(stats)
    return stats


def render_impact_dashboard(email: str) -> None:
    stats = load_usage_stats()
    estimated_total_hours_saved = stats["total_minutes_saved"] / 60

    st.write("### Impact Dashboard")
    impact_columns = st.columns(5)
    impact_columns[0].metric("Total users (emails entered)", stats["total_users"])
    impact_columns[1].metric("Total sessions", stats["total_sessions"])
    impact_columns[2].metric("Total photos processed", stats["total_photos_processed"])
    impact_columns[3].metric("Total exports", stats["total_exports"])
    impact_columns[4].metric("Estimated total hours saved", round(estimated_total_hours_saved, 1))

    if email:
        user_stats = stats.get("users", {}).get(email, default_user_stats())
        st.write("### Your Usage")
        user_columns = st.columns(4)
        user_columns[0].metric("Your sessions", user_stats["sessions"])
        user_columns[1].metric("Your photos processed", user_stats["photos_processed"])
        user_columns[2].metric("Your exports", user_stats["exports"])
        user_columns[3].metric("Your estimated hours saved", round(user_stats["minutes_saved"] / 60, 1))


@dataclass
class CanvasSettings:
    width: int
    height: int
    padding: int
    create_exports: bool


@dataclass
class PhotoCandidate:
    path: Path
    sharpness: float
    detail_ratio: float
    contrast: float
    brightness_mean: float
    exposure_balance: float
    perceptual_hash: imagehash.ImageHash
    score: float = 0.0
    score_breakdown: dict[str, float] = field(default_factory=dict)
    selection_reason: str = ""


@dataclass
class SimilarPhotoGroup:
    keeper: PhotoCandidate
    rejected: list[PhotoCandidate] = field(default_factory=list)


def ensure_directories() -> None:
    for folder in (INPUT_DIR, OUTPUT_DIR, CANVAS_DIR):
        folder.mkdir(exist_ok=True)


def clear_output_folder(folder: Path) -> None:
    folder.mkdir(exist_ok=True)
    for item in folder.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def get_image_files(folder: Path) -> list[Path]:
    return sorted(
        file
        for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() in VALID_EXTENSIONS
    )


def load_image_metrics(image_path: Path) -> PhotoCandidate | None:
    image = cv2.imread(str(image_path))
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sharpness = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    contrast = float(gray.std())
    brightness_mean = float(gray.mean())

    edges = cv2.Canny(gray, 100, 200)
    detail_ratio = float(np.count_nonzero(edges) / edges.size)

    exposure_balance = max(0.0, 1.0 - abs(brightness_mean - 127.5) / 127.5)

    try:
        with Image.open(image_path) as pil_image:
            perceptual_hash = imagehash.phash(pil_image)
    except Exception:
        return None

    return PhotoCandidate(
        path=image_path,
        sharpness=sharpness,
        detail_ratio=detail_ratio,
        contrast=contrast,
        brightness_mean=brightness_mean,
        exposure_balance=exposure_balance,
        perceptual_hash=perceptual_hash,
    )


def normalize_metric(values: list[float]) -> list[float]:
    if not values:
        return []

    min_value = min(values)
    max_value = max(values)

    if max_value - min_value < 1e-9:
        return [0.5 for _ in values]

    return [(value - min_value) / (max_value - min_value) for value in values]


def add_quality_scores(
    candidates: list[PhotoCandidate],
    weights: dict[str, float],
) -> list[PhotoCandidate]:
    if not candidates:
        return []

    total_weight = sum(weights.values())
    normalized_weights = {
        metric: weight / total_weight
        for metric, weight in weights.items()
    }

    sharpness_scores = normalize_metric([candidate.sharpness for candidate in candidates])
    detail_scores = normalize_metric([candidate.detail_ratio for candidate in candidates])
    contrast_scores = normalize_metric([candidate.contrast for candidate in candidates])
    exposure_scores = [candidate.exposure_balance for candidate in candidates]

    for index, candidate in enumerate(candidates):
        candidate.score_breakdown = {
            "sharpness": sharpness_scores[index],
            "detail": detail_scores[index],
            "contrast": contrast_scores[index],
            "exposure": exposure_scores[index],
        }
        candidate.score = 100 * (
            normalized_weights["sharpness"] * sharpness_scores[index]
            + normalized_weights["detail"] * detail_scores[index]
            + normalized_weights["contrast"] * contrast_scores[index]
            + normalized_weights["exposure"] * exposure_scores[index]
        )
        candidate.selection_reason = build_selection_reason(candidate)

    return candidates


def build_selection_reason(candidate: PhotoCandidate) -> str:
    component_labels = {
        "sharpness": "strong sharpness",
        "detail": "good subject detail",
        "contrast": "clean contrast",
        "exposure": "balanced exposure",
    }

    top_components = sorted(
        candidate.score_breakdown.items(),
        key=lambda item: item[1],
        reverse=True,
    )[:2]

    highlights = [
        component_labels[name]
        for name, value in top_components
        if value >= 0.55
    ]

    if not highlights:
        highlights = ["solid overall image quality"]

    summary = " and ".join(highlights)
    return (
        f"Selected for {summary}. "
        f"Sharpness {candidate.sharpness:.1f}, "
        f"detail {candidate.detail_ratio:.1%}, "
        f"brightness {candidate.brightness_mean:.0f}."
    )


def filter_blurry_images(
    image_files: list[Path],
    blur_threshold: float,
    progress_bar=None,
    progress_text=None,
) -> tuple[list[PhotoCandidate], int, int]:
    candidates: list[PhotoCandidate] = []
    blurry_count = 0
    unreadable_count = 0
    total_images = len(image_files)

    for index, image_path in enumerate(image_files, start=1):
        if progress_text is not None:
            progress_text.text(f"Analyzing image {index} of {total_images}...")
        if progress_bar is not None and total_images > 0:
            progress_bar.progress(index / total_images)

        candidate = load_image_metrics(image_path)
        if candidate is None:
            unreadable_count += 1
            continue

        if candidate.sharpness < blur_threshold:
            blurry_count += 1
            continue

        candidates.append(candidate)

    return candidates, blurry_count, unreadable_count


def remove_near_duplicates(
    candidates: list[PhotoCandidate],
    duplicate_threshold: int,
) -> tuple[list[PhotoCandidate], int, list[SimilarPhotoGroup]]:
    kept_candidates: list[PhotoCandidate] = []
    duplicate_groups_by_keeper: dict[str, SimilarPhotoGroup] = {}
    duplicate_count = 0

    for candidate in sorted(candidates, key=lambda item: item.score, reverse=True):
        duplicate_keeper = None

        for kept in kept_candidates:
            if abs(candidate.perceptual_hash - kept.perceptual_hash) <= duplicate_threshold:
                duplicate_keeper = kept
                break

        if duplicate_keeper is not None:
            duplicate_count += 1
            keeper_name = duplicate_keeper.path.name
            duplicate_groups_by_keeper.setdefault(
                keeper_name,
                SimilarPhotoGroup(keeper=duplicate_keeper),
            ).rejected.append(candidate)
            continue

        kept_candidates.append(candidate)

    duplicate_groups = list(duplicate_groups_by_keeper.values())
    return kept_candidates, duplicate_count, duplicate_groups


def create_white_canvas(
    image_path: Path,
    output_path: Path,
    canvas_width: int,
    canvas_height: int,
    padding: int,
) -> None:
    with Image.open(image_path) as img:
        img = img.convert("RGB")

        max_width = canvas_width - (2 * padding)
        max_height = canvas_height - (2 * padding)
        img.thumbnail((max_width, max_height), RESAMPLING.LANCZOS)

        canvas = Image.new("RGB", (canvas_width, canvas_height), "white")
        x = (canvas_width - img.width) // 2
        y = (canvas_height - img.height) // 2

        canvas.paste(img, (x, y))
        canvas.save(output_path, quality=95)


def export_selected_images(
    selected_candidates: list[PhotoCandidate],
    canvas_settings: CanvasSettings,
) -> tuple[list[Path], list[Path]]:
    clear_output_folder(OUTPUT_DIR)
    clear_output_folder(CANVAS_DIR)

    saved_files: list[Path] = []
    canvas_files: list[Path] = []

    for rank, candidate in enumerate(selected_candidates, start=1):
        destination = OUTPUT_DIR / f"{rank:02d}_{candidate.path.name}"
        shutil.copy2(candidate.path, destination)
        saved_files.append(destination)

        if canvas_settings.create_exports:
            canvas_destination = CANVAS_DIR / f"{rank:02d}_canvas.jpg"
            create_white_canvas(
                destination,
                canvas_destination,
                canvas_settings.width,
                canvas_settings.height,
                canvas_settings.padding,
            )
            canvas_files.append(canvas_destination)

    return saved_files, canvas_files


def process_images(
    blur_threshold: float,
    duplicate_threshold: int,
    top_n: int,
    scoring_weights: dict[str, float],
    scoring_preset: str,
    progress_bar=None,
    progress_text=None,
) -> dict:
    ensure_directories()

    image_files = get_image_files(INPUT_DIR)
    candidates, blurry_count, unreadable_count = filter_blurry_images(
        image_files,
        blur_threshold,
        progress_bar=progress_bar,
        progress_text=progress_text,
    )

    if progress_text is not None:
        progress_text.text("Ranking photos and removing near-duplicates...")

    scored_candidates = add_quality_scores(candidates, scoring_weights)
    unique_candidates, duplicate_count, duplicate_groups = remove_near_duplicates(
        scored_candidates,
        duplicate_threshold,
    )
    selected_candidates = unique_candidates[:top_n]

    if progress_bar is not None:
        progress_bar.progress(1.0)
    if progress_text is not None:
        progress_text.text("Processing complete.")

    return {
        "total": len(image_files),
        "blurry_removed": blurry_count,
        "duplicates_removed": duplicate_count,
        "unreadable_skipped": unreadable_count,
        "selected": len(selected_candidates),
        "selected_candidates": selected_candidates,
        "duplicate_groups": duplicate_groups,
        "scoring_preset": scoring_preset,
    }


def save_uploaded_files(uploaded_files: list) -> None:
    ensure_directories()
    clear_output_folder(INPUT_DIR)

    for uploaded_file in uploaded_files:
        file_name = Path(uploaded_file.name).name
        file_path = INPUT_DIR / file_name
        with open(file_path, "wb") as file_handle:
            file_handle.write(uploaded_file.getbuffer())


def make_zip(folder: Path, zip_name: str) -> Path:
    zip_path = Path(zip_name)
    with zipfile.ZipFile(zip_path, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for file in sorted(folder.iterdir()):
            if file.is_file():
                zip_file.write(file, arcname=file.name)
    return zip_path


def render_summary(results: dict) -> None:
    metric_columns = st.columns(5)
    metric_columns[0].metric("Uploaded", results["total"])
    metric_columns[1].metric("Blurred Removed", results["blurry_removed"])
    metric_columns[2].metric("Duplicates Removed", results["duplicates_removed"])
    metric_columns[3].metric("Unreadable Skipped", results["unreadable_skipped"])
    metric_columns[4].metric("Auto Shortlist", results["selected"])


def render_selected_table(candidates: list[PhotoCandidate]) -> None:
    table_rows = []
    for rank, candidate in enumerate(candidates, start=1):
        table_rows.append(
            {
                "Rank": rank,
                "File": candidate.path.name,
                "Score": round(candidate.score, 1),
                "Sharpness": round(candidate.sharpness, 1),
                "Detail %": round(candidate.detail_ratio * 100, 2),
                "Contrast": round(candidate.contrast, 1),
                "Brightness": round(candidate.brightness_mean, 1),
                "Why Selected": candidate.selection_reason,
            }
        )

    st.dataframe(table_rows, use_container_width=True, hide_index=True)


def render_image_grid(
    candidates: list[PhotoCandidate],
) -> None:
    if not candidates:
        return

    columns = st.columns(3)

    for index, candidate in enumerate(candidates):
        with columns[index % 3]:
            st.image(str(candidate.path), use_container_width=True)
            st.caption(f"#{index + 1} {candidate.path.name}")
            st.caption(candidate.selection_reason)


def get_duplicate_candidate_lookup(
    duplicate_groups: list[SimilarPhotoGroup],
) -> dict[str, PhotoCandidate]:
    lookup = {}

    for group in duplicate_groups:
        lookup[group.keeper.path.name] = group.keeper
        for candidate in group.rejected:
            lookup[candidate.path.name] = candidate

    return lookup


def apply_similar_photo_swaps(
    candidates: list[PhotoCandidate],
    duplicate_groups: list[SimilarPhotoGroup],
) -> list[PhotoCandidate]:
    swaps = st.session_state.get("similar_photo_swaps", {})
    duplicate_lookup = get_duplicate_candidate_lookup(duplicate_groups)
    effective_candidates = []

    for candidate in candidates:
        replacement_name = swaps.get(candidate.path.name)
        replacement = duplicate_lookup.get(replacement_name)
        effective_candidates.append(replacement or candidate)

    return effective_candidates


def swap_similar_photo(original_name: str, current_name: str, replacement_name: str) -> None:
    swaps = dict(st.session_state.get("similar_photo_swaps", {}))
    selected_filenames = set(st.session_state.get("selected_filenames", set()))

    swaps[original_name] = replacement_name

    if current_name in selected_filenames:
        selected_filenames.remove(current_name)
        selected_filenames.add(replacement_name)

    st.session_state.similar_photo_swaps = swaps
    st.session_state.selected_filenames = selected_filenames
    st.session_state.export_results = None
    st.session_state.export_signature = None


def render_compare_similar_photos(
    candidates: list[PhotoCandidate],
    duplicate_groups: list[SimilarPhotoGroup],
) -> list[PhotoCandidate]:
    relevant_groups = [
        group
        for group in duplicate_groups
        if group.keeper.path.name in {candidate.path.name for candidate in candidates}
    ]

    if not relevant_groups:
        return candidates

    st.write("### Compare Similar Photos")

    swaps = st.session_state.get("similar_photo_swaps", {})

    for group_index, group in enumerate(relevant_groups):
        current_name = swaps.get(group.keeper.path.name, group.keeper.path.name)
        all_options = [group.keeper] + group.rejected
        current_candidate = next(
            (
                candidate
                for candidate in all_options
                if candidate.path.name == current_name
            ),
            group.keeper,
        )
        current_name = current_candidate.path.name

        st.write(f"Similar set {group_index + 1}")
        columns = st.columns(len(all_options))

        for option_index, candidate in enumerate(all_options):
            with columns[option_index]:
                st.image(str(candidate.path), use_container_width=True)
                label = "Current pick" if candidate.path.name == current_name else "Use this instead"
                st.caption(f"{candidate.path.name} | Score {candidate.score:.1f}")

                if st.button(
                    label,
                    key=f"swap_{group.keeper.path.name}_{candidate.path.name}",
                    disabled=candidate.path.name == current_name,
                ):
                    swap_similar_photo(
                        original_name=group.keeper.path.name,
                        current_name=current_candidate.path.name,
                        replacement_name=candidate.path.name,
                    )
                    st.rerun()

    return apply_similar_photo_swaps(candidates, duplicate_groups)


def reset_manual_selection(candidates: list[PhotoCandidate]) -> None:
    for key in list(st.session_state.keys()):
        if key.startswith("select_photo__"):
            del st.session_state[key]

    st.session_state.selected_filenames = {
        candidate.path.name
        for candidate in candidates
    }
    st.session_state.export_results = None
    st.session_state.export_signature = None
    st.session_state.similar_photo_swaps = {}


def get_selected_candidates(candidates: list[PhotoCandidate]) -> list[PhotoCandidate]:
    selected_filenames = set(st.session_state.get("selected_filenames", set()))
    return [
        candidate
        for candidate in candidates
        if candidate.path.name in selected_filenames
    ]


def get_export_signature(
    candidates: list[PhotoCandidate],
    canvas_settings: CanvasSettings,
) -> tuple:
    selected_signature = tuple(
        candidate.path.name
        for candidate in get_selected_candidates(candidates)
    )
    canvas_signature = (
        canvas_settings.width,
        canvas_settings.height,
        canvas_settings.padding,
        canvas_settings.create_exports,
    )
    return selected_signature + canvas_signature


def render_manual_selection_grid(candidates: list[PhotoCandidate]) -> list[PhotoCandidate]:
    selected_filenames = set(st.session_state.get("selected_filenames", set()))
    columns = st.columns(3)

    for index, candidate in enumerate(candidates):
        checkbox_key = f"select_photo__{candidate.path.name}"

        with columns[index % 3]:
            st.image(str(candidate.path), use_container_width=True)
            st.caption(f"#{index + 1} {candidate.path.name}")
            is_checked = st.checkbox(
                "Keep in final export",
                value=candidate.path.name in selected_filenames,
                key=checkbox_key,
            )

        if is_checked:
            selected_filenames.add(candidate.path.name)
        else:
            selected_filenames.discard(candidate.path.name)

    st.session_state.selected_filenames = selected_filenames
    return get_selected_candidates(candidates)


def render_downloads(results: dict, create_canvas_exports: bool) -> None:
    output_zip = make_zip(OUTPUT_DIR, "selected_photos.zip")
    with open(output_zip, "rb") as output_handle:
        st.download_button(
            "Download Selected Photos ZIP",
            data=output_handle.read(),
            file_name="selected_photos.zip",
            mime="application/zip",
        )

    if create_canvas_exports and results["canvas_files"]:
        canvas_zip = make_zip(CANVAS_DIR, "canvas_photos.zip")
        with open(canvas_zip, "rb") as canvas_handle:
            st.download_button(
                "Download Canvas Photos ZIP",
                data=canvas_handle.read(),
                file_name="canvas_photos.zip",
                mime="application/zip",
            )


def main() -> None:
    ensure_directories()

    st.set_page_config(page_title="Gec Shots ClutchCull", layout="wide")

    st.title("Gec Shots ClutchCull")
    st.subheader("AI culling for sports photographers")
    st.write("Upload a batch, filter weak frames, remove duplicates, and keep the best shots.")

    st.sidebar.header("Culling Settings")
    scoring_preset = st.sidebar.selectbox(
        "Scoring Preset",
        list(SCORING_PRESETS.keys()),
        index=list(SCORING_PRESETS.keys()).index("Balanced"),
    )
    blur_threshold = st.sidebar.slider("Blur Threshold", 0.0, 100.0, 12.0, 1.0)
    duplicate_threshold = st.sidebar.slider("Duplicate Threshold", 0, 10, 2, 1)
    top_n = st.sidebar.slider("Number of Final Photos", 1, 100, 35, 1)
    seconds_per_photo = st.sidebar.slider("Manual review seconds per photo", 2, 15, 6, 1)

    st.sidebar.caption(
        "Higher blur thresholds are stricter. Higher duplicate thresholds remove more lookalike frames."
    )

    st.sidebar.header("Canvas Settings")
    create_canvas_exports = st.sidebar.checkbox(
        "Create Instagram-ready white canvas exports",
        value=True,
    )
    canvas_width = 1080
    canvas_height = 1350
    padding = 80

    if create_canvas_exports:
        canvas_width = st.sidebar.selectbox("Canvas Width", [1080, 1200], index=0)
        canvas_height = st.sidebar.selectbox("Canvas Height", [1080, 1350], index=1)
        padding = st.sidebar.slider("Canvas Padding", 0, 200, 80, 5)

    uploaded_files = st.file_uploader(
        "Upload your photos",
        type=["jpg", "jpeg", "png", "webp"],
        accept_multiple_files=True,
    )
    email = normalize_email(
        st.text_input("Email (optional, used only to track usage impact)")
    )
    st.caption("Used only to track app usage and improve the tool. No spam.")

    update_session_stats(email)

    if not uploaded_files:
        st.info("Add a batch of photos to start culling.")
        return

    st.write(f"{len(uploaded_files)} files uploaded.")

    if st.button("Process Photos", type="primary"):
        progress_text = st.empty()
        progress_bar = st.progress(0)

        with st.spinner("Analyzing sharpness, duplicates, and overall image quality..."):
            save_uploaded_files(uploaded_files)
            results = process_images(
                blur_threshold=blur_threshold,
                duplicate_threshold=duplicate_threshold,
                top_n=top_n,
                scoring_weights=SCORING_PRESETS[scoring_preset],
                scoring_preset=scoring_preset,
                progress_bar=progress_bar,
                progress_text=progress_text,
            )

        st.session_state.cull_results = results
        st.session_state.last_photos_processed = results["total"]
        st.session_state.current_batch_id = st.session_state.get("current_batch_id", 0) + 1
        update_photos_processed_stats(email, results["total"])
        reset_manual_selection(results["selected_candidates"])

    results = st.session_state.get("cull_results")

    if results is None:
        return

    canvas_settings = CanvasSettings(
        width=canvas_width,
        height=canvas_height,
        padding=padding,
        create_exports=create_canvas_exports,
    )

    st.success("Processing complete.")
    render_summary(results)

    duplicate_groups = results.get("duplicate_groups", [])
    effective_candidates = apply_similar_photo_swaps(
        results["selected_candidates"],
        duplicate_groups,
    )

    if not effective_candidates:
        st.warning(
            "No photos passed the current filters. Try lowering the blur threshold "
            "or increasing the duplicate threshold."
        )
        render_impact_dashboard(email)
        return

    st.write("### Selection Notes")
    st.write(
        f"Final ranking uses the {results['scoring_preset']} preset after blur "
        "filtering and duplicate removal."
    )
    render_selected_table(effective_candidates)

    st.write("### Selected Photos")
    render_image_grid(effective_candidates)

    effective_candidates = render_compare_similar_photos(
        results["selected_candidates"],
        duplicate_groups,
    )

    st.write("### Manual Final Selection")
    selected_candidates = render_manual_selection_grid(effective_candidates)
    st.write(f"{len(selected_candidates)} of {len(effective_candidates)} photos selected for export.")

    current_signature = get_export_signature(effective_candidates, canvas_settings)
    export_results = st.session_state.get("export_results")
    export_signature = st.session_state.get("export_signature")

    if st.button("Export Checked Photos", type="primary", disabled=not selected_candidates):
        with st.spinner("Exporting checked photos and building ZIP files..."):
            saved_files, canvas_files = export_selected_images(
                selected_candidates,
                canvas_settings,
            )

        export_results = {
            "saved_files": saved_files,
            "canvas_files": canvas_files,
        }

        current_batch_id = st.session_state.get("current_batch_id")
        counted_export_batch_ids = list(st.session_state.get("counted_export_batch_ids", []))
        if current_batch_id not in counted_export_batch_ids:
            update_export_stats(
                email=email,
                photos_processed=st.session_state.get("last_photos_processed", results["total"]),
                seconds_per_photo=seconds_per_photo,
            )
            counted_export_batch_ids.append(current_batch_id)
            st.session_state.counted_export_batch_ids = counted_export_batch_ids

        st.session_state.export_results = export_results
        st.session_state.export_signature = current_signature
        export_signature = current_signature
        st.success("Checked photos exported.")

    if export_results and export_signature == current_signature:
        if canvas_settings.create_exports and export_results["canvas_files"]:
            st.write("### Canvas Versions")
            canvas_columns = st.columns(3)
            for index, image_path in enumerate(export_results["canvas_files"]):
                with canvas_columns[index % 3]:
                    st.image(str(image_path), caption=image_path.name, use_container_width=True)

        render_downloads(export_results, canvas_settings.create_exports)

    elif export_results:
        st.info("Your manual selection changed. Export checked photos again to refresh the ZIP files.")

    render_impact_dashboard(email)


if __name__ == "__main__":
    main()
