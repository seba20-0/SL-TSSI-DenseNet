import argparse
import os
import json
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np

try:
    import mediapipe as mp
except ImportError as e:
    raise SystemExit("mediapipe is required. Install with: pip install mediapipe opencv-python")


def find_videos(src_root: Path) -> List[Path]:
    return sorted(src_root.rglob("*.mp4"))


def parse_ids(path: Path) -> Tuple[str, str, str]:
    # Expect path like .../user01/G01/R01.mp4
    user = path.parent.parent.name  # userXX
    gesture = path.parent.name      # GXX
    repetition = path.stem          # RXX or RXX_something
    if repetition.startswith("R"):
        repetition = repetition[:3]
    return user, gesture, repetition


def build_label_map(gestures: List[str]) -> dict:
    uniq = sorted(set(gestures))
    return {g: i for i, g in enumerate(uniq)}


def extract_landmarks_from_video(video_path: Path,
                                 max_frames: int = None) -> np.ndarray:
    # Returns array of shape [frames, joints, 2] with joints = 33 (pose) + 21 (LH) + 21 (RH)
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    holistic = mp.solutions.holistic.Holistic(
        static_image_mode=False,
        model_complexity=1,
        smooth_landmarks=True,
        enable_segmentation=False,
        refine_face_landmarks=False,
    )

    frames = []
    try:
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            count += 1
            if max_frames is not None and count > max_frames:
                break
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = holistic.process(rgb)

            # Pose: 33 keypoints
            pose_xy = np.full((33, 2), np.nan, dtype=np.float32)
            if results.pose_landmarks is not None:
                for i, lm in enumerate(results.pose_landmarks.landmark):
                    pose_xy[i] = [lm.x, lm.y]

            # Hands: 21 + 21
            lh_xy = np.full((21, 2), np.nan, dtype=np.float32)
            rh_xy = np.full((21, 2), np.nan, dtype=np.float32)
            if results.left_hand_landmarks is not None:
                for i, lm in enumerate(results.left_hand_landmarks.landmark):
                    lh_xy[i] = [lm.x, lm.y]
            if results.right_hand_landmarks is not None:
                for i, lm in enumerate(results.right_hand_landmarks.landmark):
                    rh_xy[i] = [lm.x, lm.y]

            # Combine: [pose, left_hand, right_hand] -> shape [75, 2]
            keypoints = np.concatenate([pose_xy, lh_xy, rh_xy], axis=0)
            # Replace NaNs with 0 to avoid NaN propagating through training
            keypoints = np.nan_to_num(keypoints, nan=0.0)

            frames.append(keypoints)
    finally:
        cap.release()
        holistic.close()

    if len(frames) == 0:
        raise RuntimeError(f"No frames extracted from: {video_path}")
    arr = np.stack(frames, axis=0)  # [T, 75, 2]
    return arr


def main():
    parser = argparse.ArgumentParser(description="Prepare MLR511 dataset to NPZ format for SL-TSSI-DenseNet")
    parser.add_argument("--src", required=True, help="Path to original dataset root (e.g., datasets/MLR511-ArabicSignLanguage-Dataset-MP4)")
    parser.add_argument("--dst", required=True, help="Output directory (e.g., datasets/mlr511_npz)")
    parser.add_argument("--val_users", default="user12", help="Comma-separated user IDs for validation split (e.g., user12)")
    parser.add_argument("--test_users", default="user01,user08,user11", help="Comma-separated user IDs for test split (e.g., user01,user08,user11)")
    parser.add_argument("--max_frames", type=int, default=None, help="Optionally cap frames per video (for quick tests)")
    args = parser.parse_args()

    src_root = Path(args.src)
    dst_root = Path(args.dst)
    samples_dir = dst_root / "samples"
    splits_dir = dst_root / "splits"

    samples_dir.mkdir(parents=True, exist_ok=True)
    splits_dir.mkdir(parents=True, exist_ok=True)

    videos = find_videos(src_root)
    if not videos:
        raise SystemExit(f"No .mp4 videos found under {src_root}")

    print(f"[INFO] Found {len(videos)} videos. Extracting landmarks...")

    gestures = []
    index = []  # (npz_rel_path, user, gesture)
    for i, vp in enumerate(videos, 1):
        user, gesture, repetition = parse_ids(vp)
        gestures.append(gesture)

        out_name = f"{user}_{gesture}_{repetition}.npz"
        out_path = samples_dir / out_name
        if out_path.exists():
            print(f"[SKIP] Exists: {out_path}")
            index.append((out_path, user, gesture))
            continue

        try:
            pose = extract_landmarks_from_video(vp, max_frames=args.max_frames)
        except Exception as e:
            print(f"[WARN] Failed {vp}: {e}")
            continue

        # Label map will be computed later. Temporarily set label=-1
        np.savez_compressed(out_path, pose=pose, label=-1)
        index.append((out_path, user, gesture))
        print(f"[OK] {i}/{len(videos)} -> {out_path}")

    # Build label map
    label_map = build_label_map(gestures)
    with open(dst_root / "label_map.json", "w", encoding="utf-8") as f:
        json.dump(label_map, f, indent=2)
    print(f"[INFO] label_map: {label_map}")

    # Write splits
    val_users = set([u.strip() for u in args.val_users.split(",") if u.strip()])
    test_users = set([u.strip() for u in args.test_users.split(",") if u.strip()])
    train_lines, val_lines, test_lines = [], [], []
    for npz_path, user, gesture in index:
        # fill label if missing
        try:
            data = np.load(npz_path, allow_pickle=True)
            if int(data['label']) == -1:
                lbl = int(label_map[gesture])
                pose = data['pose']
                np.savez_compressed(npz_path, pose=pose, label=lbl)
        except Exception as e:
            print(f"[WARN] Update label failed for {npz_path}: {e}")

        rel = os.path.relpath(npz_path, dst_root)
        if user in test_users:
            test_lines.append(rel)
        elif user in val_users:
            val_lines.append(rel)
        else:
            train_lines.append(rel)

    (splits_dir / "train.txt").write_text("\n".join(train_lines), encoding="utf-8")
    (splits_dir / "validation.txt").write_text("\n".join(val_lines), encoding="utf-8")
    (splits_dir / "test.txt").write_text("\n".join(test_lines), encoding="utf-8")
    print(f"[INFO] Wrote {len(train_lines)} train, {len(val_lines)} val, {len(test_lines)} test samples.")


if __name__ == "__main__":
    main()
