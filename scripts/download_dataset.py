"""
MediScan -- HAM10000 Dataset Download Script

Downloads the HAM10000 dataset from Kaggle and extracts it into the
project's data/raw/ directory.

Usage:
    python scripts/download_dataset.py

Requirements:
    - Kaggle account
    - Kaggle API credentials configured (see instructions below)

Kaggle API Setup:
    1. Go to https://www.kaggle.com -> Account -> API -> Create New API Token
    2. This downloads a kaggle.json file
    3. Place it in:
       - Windows: C:\\Users\\<username>\\.kaggle\\kaggle.json
       - Linux/Mac: ~/.kaggle/kaggle.json
    4. Ensure the file has restricted permissions (Linux/Mac: chmod 600)

The script will NOT expose or require credentials in source code.
"""

import os
import sys
import shutil
import zipfile
from pathlib import Path


def get_project_root() -> Path:
    """Get the project root directory (parent of scripts/)."""
    return Path(__file__).resolve().parent.parent


def check_dataset_exists(raw_dir: Path) -> bool:
    """
    Check if the dataset appears to already be downloaded.

    Returns True if the raw directory contains image files and metadata.
    """
    if not raw_dir.exists():
        return False

    # Look for image files
    image_files = list(raw_dir.rglob("*.jpg")) + list(raw_dir.rglob("*.jpeg"))
    if len(image_files) < 100:
        return False

    # Look for metadata CSV
    csv_files = list(raw_dir.rglob("*metadata*.csv")) + list(raw_dir.rglob("*metadata*"))
    csv_found = any(f.suffix in ('.csv', '') and 'metadata' in f.name.lower() for f in raw_dir.rglob("*"))

    return len(image_files) > 100 and csv_found


def check_kaggle_credentials() -> bool:
    """Check if Kaggle API credentials are configured."""
    # Check environment variable
    if os.environ.get("KAGGLE_USERNAME") and os.environ.get("KAGGLE_KEY"):
        return True

    # Check kaggle.json file
    kaggle_json_paths = [
        Path.home() / ".kaggle" / "kaggle.json",
        Path(os.environ.get("KAGGLE_CONFIG_DIR", "")) / "kaggle.json" if os.environ.get("KAGGLE_CONFIG_DIR") else None,
    ]

    for path in kaggle_json_paths:
        if path and path.exists():
            return True

    return False


def download_with_kaggle_api(raw_dir: Path) -> bool:
    """Download HAM10000 using the Kaggle Python API."""
    try:
        from kaggle.api.kaggle_api_extended import KaggleApi

        api = KaggleApi()
        api.authenticate()

        print("Downloading HAM10000 from Kaggle...")
        print("Dataset: kmader/skin-cancer-mnist-ham10000")
        print(f"Destination: {raw_dir}")
        print("This may take several minutes (~2.6 GB)...\n")

        api.dataset_download_files(
            "kmader/skin-cancer-mnist-ham10000",
            path=str(raw_dir),
            unzip=True,
        )

        print("\nDownload complete.")
        return True

    except ImportError:
        print("ERROR: kaggle package not installed.")
        print("Install it with: pip install kaggle")
        return False
    except Exception as e:
        print(f"ERROR during download: {e}")
        return False


def download_with_opendatasets(raw_dir: Path) -> bool:
    """Fallback download method using opendatasets."""
    try:
        import opendatasets as od

        print("Downloading HAM10000 using opendatasets...")
        print("You may be prompted for your Kaggle username and API key.\n")

        url = "https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000"
        od.download(url, data_dir=str(raw_dir.parent))

        # opendatasets creates a subdirectory; move contents if needed
        sub_dir = raw_dir.parent / "skin-cancer-mnist-ham10000"
        if sub_dir.exists() and sub_dir != raw_dir:
            # Move contents to raw_dir
            raw_dir.mkdir(parents=True, exist_ok=True)
            for item in sub_dir.iterdir():
                dest = raw_dir / item.name
                if not dest.exists():
                    shutil.move(str(item), str(dest))
            # Clean up empty directory
            if not list(sub_dir.iterdir()):
                sub_dir.rmdir()

        return True

    except ImportError:
        print("ERROR: opendatasets package not installed.")
        print("Install it with: pip install opendatasets")
        return False
    except Exception as e:
        print(f"ERROR during download: {e}")
        return False


def extract_zip_files(raw_dir: Path) -> None:
    """Extract any remaining ZIP files in the raw directory."""
    for zip_file in raw_dir.glob("*.zip"):
        print(f"Extracting: {zip_file.name}")
        try:
            with zipfile.ZipFile(zip_file, 'r') as zf:
                zf.extractall(raw_dir)
            print(f"  Extracted successfully.")
        except zipfile.BadZipFile:
            print(f"  WARNING: {zip_file.name} is not a valid ZIP file, skipping.")


def copy_metadata(raw_dir: Path, metadata_dir: Path) -> None:
    """Copy metadata CSV to the metadata directory."""
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Search for metadata files
    for pattern in ["*metadata*.csv", "*metadata*"]:
        for f in raw_dir.rglob(pattern):
            if f.is_file():
                dest = metadata_dir / f.name
                if not dest.exists():
                    shutil.copy2(str(f), str(dest))
                    print(f"Copied metadata: {f.name} -> {metadata_dir}")


def verify_download(raw_dir: Path) -> dict:
    """Verify the downloaded dataset structure."""
    result = {
        "raw_dir_exists": raw_dir.exists(),
        "image_count": 0,
        "csv_files": [],
        "subdirectories": [],
        "total_size_mb": 0,
    }

    if not raw_dir.exists():
        return result

    # Count images
    image_extensions = (".jpg", ".jpeg", ".png")
    image_count = 0
    total_size = 0

    for root, dirs, files in os.walk(raw_dir):
        result["subdirectories"].extend(
            [os.path.relpath(os.path.join(root, d), raw_dir) for d in dirs]
        )
        for f in files:
            filepath = Path(root) / f
            total_size += filepath.stat().st_size
            if f.lower().endswith(image_extensions):
                image_count += 1
            if f.endswith(".csv"):
                result["csv_files"].append(f)

    result["image_count"] = image_count
    result["total_size_mb"] = round(total_size / (1024 * 1024), 1)

    return result


def main():
    project_root = get_project_root()
    raw_dir = project_root / "data" / "raw"
    metadata_dir = project_root / "data" / "metadata"

    print("=" * 60)
    print("MediScan -- HAM10000 Dataset Download")
    print("=" * 60)
    print(f"Project root: {project_root}")
    print(f"Raw data dir: {raw_dir}")
    print()

    # Step 1: Check if dataset already exists
    if check_dataset_exists(raw_dir):
        print("Dataset appears to already be downloaded.")
        print("Verifying existing data...\n")
        result = verify_download(raw_dir)
        print(f"  Images found: {result['image_count']}")
        print(f"  CSV files: {result['csv_files']}")
        print(f"  Total size: {result['total_size_mb']} MB")

        # Copy metadata if needed
        copy_metadata(raw_dir, metadata_dir)
        print("\nDataset is ready. Run 'python scripts/eda.py' for exploration.")
        return

    # Step 2: Create directories
    raw_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    # Step 3: Check Kaggle credentials
    if not check_kaggle_credentials():
        print("Kaggle API credentials not found.\n")
        print("To download the dataset, you have two options:\n")
        print("OPTION A -- Kaggle API (recommended):")
        print("  1. Go to https://www.kaggle.com")
        print("  2. Account -> Settings -> API -> Create New API Token")
        print("  3. Save kaggle.json to:")
        if sys.platform == "win32":
            print(f"     C:\\Users\\{os.getlogin()}\\.kaggle\\kaggle.json")
        else:
            print("     ~/.kaggle/kaggle.json")
        print("  4. Re-run this script\n")
        print("OPTION B -- Manual download:")
        print("  1. Go to: https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000")
        print("  2. Click 'Download' (requires Kaggle account)")
        print("  3. Extract the ZIP contents into:")
        print(f"     {raw_dir}")
        print("  4. Run 'python scripts/eda.py' to verify\n")
        return

    # Step 4: Download
    success = download_with_kaggle_api(raw_dir)
    if not success:
        print("\nFalling back to opendatasets...")
        success = download_with_opendatasets(raw_dir)

    if not success:
        print("\nDownload failed. Please download manually:")
        print(f"  https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000")
        print(f"  Extract to: {raw_dir}")
        return

    # Step 5: Extract any remaining ZIP files
    extract_zip_files(raw_dir)

    # Step 6: Copy metadata
    copy_metadata(raw_dir, metadata_dir)

    # Step 7: Verify
    print("\nVerifying download...")
    result = verify_download(raw_dir)
    print(f"  Images found: {result['image_count']}")
    print(f"  CSV files: {result['csv_files']}")
    print(f"  Total size: {result['total_size_mb']} MB")
    print(f"  Subdirectories: {result['subdirectories']}")

    if result["image_count"] > 0:
        print("\nDataset downloaded successfully!")
        print("Run 'python scripts/eda.py' for exploration.")
    else:
        print("\nWARNING: No images found after download.")
        print("The download may have failed or the archive structure may differ.")
        print(f"Please check: {raw_dir}")


if __name__ == "__main__":
    main()
