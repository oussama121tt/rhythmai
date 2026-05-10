#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Download and setup PTB-XL datasets.
Run once before training:
    python setup_data.py
"""

import os
import sys
import zipfile
import shutil
import urllib.request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')

# Where the pipelines expect the data (must match config.yaml)
SIGNALS_DIR = os.path.join(BASE_DIR, 'PTB-XL ECG dataset',
                           'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.1')
IMAGES_DIR = os.path.join(BASE_DIR, 'PTB-XL ECG image (GMC2024)')

PHYSIONET_URL = (
    'https://physionet.org/static/published-projects/ptb-xl/'
    'ptb-xl-a-large-publicly-available-electrocardiography-dataset-1.0.3.zip'
)


def download_file(url, dest_path):
    """Download with progress bar."""
    print(f"  Downloading: {url}")
    print(f"  To: {dest_path}")

    def progress(count, block_size, total_size):
        if total_size > 0:
            percent = min(100, int(count * block_size * 100 / total_size))
            mb = count * block_size / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            bar = '#' * (percent // 2) + '-' * (50 - percent // 2)
            print(f'\r  [{bar}] {percent}% ({mb:.0f}/{total_mb:.0f} MB)',
                  end='', flush=True)

    urllib.request.urlretrieve(url, dest_path, reporthook=progress)
    print()


def setup_signals():
    """Download and extract PTB-XL signals from PhysioNet."""
    csv_path = os.path.join(SIGNALS_DIR, 'ptbxl_database.csv')

    if os.path.exists(csv_path):
        n_hea = len([f for f in os.listdir(
            os.path.join(SIGNALS_DIR, 'records500', '00000'))
            if f.endswith('.hea')]) if os.path.exists(
                os.path.join(SIGNALS_DIR, 'records500', '00000')) else 0
        print(f"  PTB-XL signals already present ({csv_path})")
        print(f"  Sample check: {n_hea} .hea files in records500/00000/")
        return True

    print("\n  PTB-XL Signals")
    print("  " + "-" * 40)

    os.makedirs(DATA_DIR, exist_ok=True)
    zip_path = os.path.join(DATA_DIR, 'ptbxl.zip')

    # Download
    if not os.path.exists(zip_path):
        try:
            download_file(PHYSIONET_URL, zip_path)
        except Exception as e:
            print(f"\n  Download failed: {e}")
            print(f"  Manual download (pick one):")
            print(f"    Kaggle:    https://www.kaggle.com/datasets/khyeh0719/ptb-xl-dataset")
            print(f"    PhysioNet: https://physionet.org/content/ptb-xl/1.0.3/")
            print(f"  Then extract to: {SIGNALS_DIR}")
            return False
    else:
        print(f"  Zip already downloaded: {zip_path}")

    # Extract
    print("  Extracting (this may take a few minutes)...")
    parent_dir = os.path.join(BASE_DIR, 'PTB-XL ECG dataset')
    os.makedirs(parent_dir, exist_ok=True)

    with zipfile.ZipFile(zip_path, 'r') as z:
        z.extractall(parent_dir)

    # The zip may create a nested folder — find it
    if not os.path.exists(csv_path):
        # Look for ptbxl_database.csv recursively
        for root, dirs, files in os.walk(parent_dir):
            if 'ptbxl_database.csv' in files:
                if root != SIGNALS_DIR:
                    # Move contents to expected location
                    os.makedirs(SIGNALS_DIR, exist_ok=True)
                    for item in os.listdir(root):
                        src = os.path.join(root, item)
                        dst = os.path.join(SIGNALS_DIR, item)
                        if not os.path.exists(dst):
                            shutil.move(src, dst)
                break

    # Verify
    if os.path.exists(csv_path):
        print(f"  PTB-XL signals ready: {SIGNALS_DIR}")
        if os.path.exists(zip_path):
            os.remove(zip_path)
            print("  Cleaned up zip file.")
        return True
    else:
        print(f"  Extraction may have failed.")
        print(f"  Expected: {csv_path}")
        print(f"  Check {parent_dir} and move files manually.")
        return False


def setup_images():
    """Check for PTB-XL images (manual download required)."""
    if os.path.exists(IMAGES_DIR):
        subdirs = [d for d in os.listdir(IMAGES_DIR)
                    if os.path.isdir(os.path.join(IMAGES_DIR, d))]
        if len(subdirs) > 10:
            print(f"  PTB-XL images already present: {IMAGES_DIR}")
            print(f"  Found {len(subdirs)} subfolders")
            return True

    print("\n  PTB-XL Images (manual download)")
    print("  " + "-" * 40)
    print("  Automatic download not available for ECG images.")
    print()
    print("  Steps:")
    print("  1. Go to: https://www.kaggle.com/datasets/bjoernjostein/ptb-xl-ecg-image-gmc2024")
    print(f"  2. Download and extract to:")
    print(f"     {IMAGES_DIR}")
    print(f"  3. Expected structure:")
    print(f"     PTB-XL ECG image (GMC2024)/")
    print(f"       00000/")
    print(f"         00001_lr-0.png")
    print(f"         00002_lr-0.png")
    print(f"         ...")
    print(f"       01000/")
    print(f"         ...")
    print(f"  4. Re-run: python setup_data.py")
    print()
    print("  NOTE: Images are only needed for pipeline_images and")
    print("        pipeline_fusion. pipeline_signals works without them.")

    os.makedirs(IMAGES_DIR, exist_ok=True)
    return False


def verify():
    """Final verification."""
    print()
    print("=" * 55)
    print("  VERIFICATION")
    print("=" * 55)

    sig_ok = os.path.exists(os.path.join(SIGNALS_DIR, 'ptbxl_database.csv'))
    img_ok = (os.path.exists(IMAGES_DIR) and
              len([d for d in os.listdir(IMAGES_DIR)
                   if os.path.isdir(os.path.join(IMAGES_DIR, d))]) > 10)

    print(f"  Signals:  {'OK' if sig_ok else 'MISSING'}")
    print(f"  Images:   {'OK' if img_ok else 'MISSING (see above)'}")
    print()

    if sig_ok:
        print("  Available commands:")
        print("    python -m pipeline_signals.train --config config.yaml")
        if img_ok:
            print("    python -m pipeline_images.train --config config.yaml")
            print("    python -m pipeline_fusion.train --config config.yaml")
        else:
            print()
            print("  For image/fusion pipelines, download images first.")
    else:
        print("  Download signals first (see errors above).")

    print()
    return sig_ok


if __name__ == '__main__':
    print("=" * 55)
    print("  ECG Classification - Data Setup")
    print("=" * 55)

    ok_sig = setup_signals()
    ok_img = setup_images()
    verify()

    if not ok_sig:
        sys.exit(1)
