"""
make_test_split.py
-------------------
Run this ONCE if your downloaded dataset has no separate test/ folder
(only train/ and val/). It moves every 3rd image (~33%) out of each
class folder in data/val into the matching class folder in data/test,
so you end up with a real, untouched held-out test set.
"""

import os
import shutil

DATA_DIR = "data"
VAL_DIR = os.path.join(DATA_DIR, "val")
TEST_DIR = os.path.join(DATA_DIR, "test")

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
MOVE_EVERY_NTH = 3  # move 1 out of every 3 images -> ~33% to test


def main():
    if not os.path.isdir(VAL_DIR):
        raise FileNotFoundError(f"{VAL_DIR} not found. Copy your dataset's val/ folder into data/ first.")

    class_folders = [
        d for d in os.listdir(VAL_DIR)
        if os.path.isdir(os.path.join(VAL_DIR, d))
    ]
    if not class_folders:
        raise ValueError(f"No class subfolders found inside {VAL_DIR}.")

    total_moved = 0
    for cls in class_folders:
        src_dir = os.path.join(VAL_DIR, cls)
        dst_dir = os.path.join(TEST_DIR, cls)
        os.makedirs(dst_dir, exist_ok=True)

        images = sorted(
            f for f in os.listdir(src_dir)
            if f.lower().endswith(IMAGE_EXTENSIONS)
        )
        to_move = images[::MOVE_EVERY_NTH]

        for fname in to_move:
            shutil.move(os.path.join(src_dir, fname), os.path.join(dst_dir, fname))

        print(f"{cls}: moved {len(to_move)} of {len(images)} images from val -> test")
        total_moved += len(to_move)

    print(f"\nDone. Moved {total_moved} images total into {TEST_DIR}.")


if __name__ == "__main__":
    main()