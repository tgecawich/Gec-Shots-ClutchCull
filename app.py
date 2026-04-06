import shutil
import zipfile
from pathlib import Path

import cv2
import imagehash
import streamlit as st
from PIL import Image

INPUT_DIR = Path("input_photos")
OUTPUT_DIR = Path("output_photos")
CANVAS_DIR = Path("canvas_photos")

VALID_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp"}


def clear_output_folder(folder: Path) -> None:
    folder.mkdir(exist_ok=True)
    for item in folder.iterdir():
        if item.is_file():
            item.unlink()
        elif item.is_dir():
            shutil.rmtree(item)


def get_image_files(folder: Path) -> list[Path]:
    return [
        file for file in folder.iterdir()
        if file.is_file() and file.suffix.lower() in VALID_EXTENSIONS
    ]


def blur_score(image_path: Path) -> float:
    image = cv2.imread(str(image_path))
    if image is None:
        return 0.0
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def get_hash(image_path: Path):
    try:
        with Image.open(image_path) as img:
            return imagehash.phash(img)
    except Exception:
        return None


def create_white_canvas(
    image_path: Path,
    output_path: Path,
    canvas_width: int,
    canvas_height: int,
    padding: int
) -> None:
    with Image.open(image_path) as img:
        img = img.convert("RGB")

        max_width = canvas_width - 2 * padding
        max_height = canvas_height - 2 * padding
        img.thumbnail((max_width, max_height))

        canvas = Image.new("RGB", (canvas_width, canvas_height), "white")

        x = (canvas_width - img.width) // 2
        y = (canvas_height - img.height) // 2

        canvas.paste(img, (x, y))
        canvas.save(output_path, quality=95)


def save_uploaded_files(uploaded_files):
    clear_output_folder(INPUT_DIR)
    for uploaded_file in uploaded_files:
        file_path = INPUT_DIR / uploaded_file.name
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())


def process_images(
    blur_threshold: float,
    duplicate_threshold: int,
    top_n: int,
    canvas_width: int,
    canvas_height: int,
    padding: int
):
    OUTPUT_DIR.mkdir(exist_ok=True)
    CANVAS_DIR.mkdir(exist_ok=True)

    clear_output_folder(OUTPUT_DIR)
    clear_output_folder(CANVAS_DIR)

    image_files = get_image_files(INPUT_DIR)

    kept_images = []
    kept_hashes = []

    blurry_count = 0
    duplicate_count = 0

    for image_path in image_files:
        score = blur_score(image_path)

        if score < blur_threshold:
            blurry_count += 1
            continue

        img_hash = get_hash(image_path)
        if img_hash is None:
            continue

        is_duplicate = False
        for existing_hash in kept_hashes:
            if abs(img_hash - existing_hash) <= duplicate_threshold:
                is_duplicate = True
                break

        if is_duplicate:
            duplicate_count += 1
            continue

        kept_images.append((image_path, score))
        kept_hashes.append(img_hash)

    kept_images.sort(key=lambda x: x[1], reverse=True)
    top_images = kept_images[:top_n]

    saved_files = []
    canvas_files = []

    for rank, (image_path, score) in enumerate(top_images, start=1):
        new_name = f"{rank:02d}_{image_path.name}"
        destination = OUTPUT_DIR / new_name
        shutil.copy2(image_path, destination)
        saved_files.append(destination)

        canvas_name = f"{rank:02d}_canvas.jpg"
        canvas_destination = CANVAS_DIR / canvas_name
        create_white_canvas(
            destination,
            canvas_destination,
            canvas_width,
            canvas_height,
            padding
        )
        canvas_files.append(canvas_destination)

    return {
        "total": len(image_files),
        "blurry_removed": blurry_count,
        "duplicates_removed": duplicate_count,
        "selected": len(saved_files),
        "saved_files": saved_files,
        "canvas_files": canvas_files,
    }


def make_zip(folder: Path, zip_name: str) -> Path:
    zip_path = Path(zip_name)
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for file in folder.iterdir():
            if file.is_file():
                zipf.write(file, arcname=file.name)
    return zip_path


st.set_page_config(page_title="Gec Shots ClutchCull", layout="wide")

st.title("Gec Shots ClutchCull")
st.subheader("AI culling for sports photographers")
st.write("Upload a batch, get your best shots faster.")

# SIDEBAR SETTINGS
st.sidebar.header("Culling Settings")
blur_threshold = st.sidebar.slider("Blur Threshold", 0.0, 100.0, 10.0, 1.0)
duplicate_threshold = st.sidebar.slider("Duplicate Threshold", 0, 10, 0, 1)
top_n = st.sidebar.slider("Number of Final Photos", 1, 100, 35, 1)

st.sidebar.header("Canvas Settings")
canvas_width = st.sidebar.selectbox("Canvas Width", [1080, 1200], index=0)
canvas_height = st.sidebar.selectbox("Canvas Height", [1080, 1350], index=1)
padding = st.sidebar.slider("Canvas Padding", 0, 200, 80, 5)

uploaded_files = st.file_uploader(
    "Upload your photos",
    type=["jpg", "jpeg", "png", "webp"],
    accept_multiple_files=True
)

if uploaded_files:
    st.write(f"{len(uploaded_files)} files uploaded.")

    if st.button("Process Photos"):
        save_uploaded_files(uploaded_files)
        results = process_images(
            blur_threshold,
            duplicate_threshold,
            top_n,
            canvas_width,
            canvas_height,
            padding
        )

        st.success("Processing complete.")

        st.write("### Results")
        st.write(f"Total uploaded: {results['total']}")
        st.write(f"Blurred removed: {results['blurry_removed']}")
        st.write(f"Duplicates removed: {results['duplicates_removed']}")
        st.write(f"Final selected: {results['selected']}")

        st.write("### Selected Photos")
        selected_columns = st.columns(3)
        for i, image_path in enumerate(results["saved_files"]):
            with selected_columns[i % 3]:
                st.image(str(image_path), caption=image_path.name, use_container_width=True)

        st.write("### Canvas Versions")
        canvas_columns = st.columns(3)
        for i, image_path in enumerate(results["canvas_files"]):
            with canvas_columns[i % 3]:
                st.image(str(image_path), caption=image_path.name, use_container_width=True)

        output_zip = make_zip(OUTPUT_DIR, "selected_photos.zip")
        canvas_zip = make_zip(CANVAS_DIR, "canvas_photos.zip")

        with open(output_zip, "rb") as f:
            st.download_button(
                "Download Selected Photos ZIP",
                data=f,
                file_name="selected_photos.zip",
                mime="application/zip"
            )

        with open(canvas_zip, "rb") as f:
            st.download_button(
                "Download Canvas Photos ZIP",
                data=f,
                file_name="canvas_photos.zip",
                mime="application/zip"
            )