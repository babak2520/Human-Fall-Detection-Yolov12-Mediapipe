#!/usr/bin/env python3
"""
Training pipeline for the ST-GCN fall detection model.

This script trains (or fine-tunes) the TwoStreamSpatialTemporalGraph model
on skeleton data extracted from fall detection datasets.

Supported input formats:
  1. Pre-extracted .npy files  (fast — no video processing during training)
  2. Video folders             (slow — extracts skeletons with MediaPipe on the fly)

For pre-extracted data, the expected directory layout is:

    data/
      train/
        features.npy   # (N, T, V, C)  — N samples, T=30 frames, V=14 joints, C=3
        labels.npy     # (N,)          — integer class labels  (0–6)
      val/
        features.npy
        labels.npy

For video-folder mode, the expected layout is:

    data/
      train/
        Standing/
          video1.mp4
          ...
        Walking/
          ...
        Fall Down/
          ...
      val/
        ...

Usage:
    python train_stgcn.py --data-dir data --epochs 100 --batch-size 32
    python train_stgcn.py --data-dir data --from-videos --epochs 50
"""

import argparse
import os
import sys
import time
import glob

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.stgcn.stgcn_model import TwoStreamSpatialTemporalGraph
from models.stgcn.action_estimator import DEFAULT_ACTION_LABELS


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SkeletonDataset(Dataset):
    """Dataset for pre-extracted skeleton sequences.

    Each sample is a tuple of (points, motions, label).
    """

    def __init__(self, features, labels, seq_length=30):
        """
        Args:
            features: np.ndarray of shape (N, T, V, C)
            labels: np.ndarray of shape (N,)
            seq_length: Expected temporal length
        """
        assert len(features) == len(labels)
        self.features = features.astype(np.float32)
        self.labels = labels.astype(np.int64)
        self.seq_length = seq_length

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        # data: (T, V, C)
        data = self.features[idx]

        # Normalize: center on mid-hip (node 13) and scale by torso
        center = data[:, 13:14, :2]
        data[:, :, :2] -= center
        torso = np.linalg.norm(data[:, 0, :2] - data[:, 13, :2], axis=-1)
        scale = max(torso.mean(), 1e-4)
        data[:, :, :2] /= scale

        # (T, V, C) -> (C, T, V)
        points = data.transpose(2, 0, 1)  # (3, T, V)

        # Motions: frame-to-frame diffs on x, y
        motions = np.zeros((2, self.seq_length, points.shape[2]), dtype=np.float32)
        motions[:, 1:, :] = points[:2, 1:, :] - points[:2, :-1, :]

        label = self.labels[idx]
        return (
            torch.from_numpy(points),
            torch.from_numpy(motions),
            torch.tensor(label, dtype=torch.long),
        )


# ---------------------------------------------------------------------------
# Video extraction helpers
# ---------------------------------------------------------------------------

def extract_skeletons_from_videos(video_dir, action_labels, seq_length=30):
    """Extract skeleton sequences from video files organized by action label.

    Args:
        video_dir: Path containing sub-folders named after action labels.
        action_labels: List of action label strings.
        seq_length: Number of frames per sequence.

    Returns:
        features: np.ndarray (N, T, V, 3)
        labels: np.ndarray (N,)
    """
    import cv2
    import mediapipe as mp
    from models.stgcn.graph import MEDIAPIPE_TO_GRAPH

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(static_image_mode=False, model_complexity=1,
                        min_detection_confidence=0.5, min_tracking_confidence=0.5)

    all_features = []
    all_labels = []

    for label_idx, label_name in enumerate(action_labels):
        label_dir = os.path.join(video_dir, label_name)
        if not os.path.isdir(label_dir):
            print(f"  [skip] No folder for label '{label_name}' at {label_dir}")
            continue

        video_files = sorted(
            glob.glob(os.path.join(label_dir, '*.mp4')) +
            glob.glob(os.path.join(label_dir, '*.avi')) +
            glob.glob(os.path.join(label_dir, '*.mov'))
        )
        print(f"  Label '{label_name}': {len(video_files)} videos")

        for vpath in video_files:
            cap = cv2.VideoCapture(vpath)
            frame_skeletons = []

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                results = pose.process(rgb)

                if results.pose_landmarks:
                    skeleton = np.zeros((14, 3), dtype=np.float32)
                    lms = results.pose_landmarks.landmark
                    for g_idx, mp_idx in MEDIAPIPE_TO_GRAPH.items():
                        if mp_idx == -1:
                            lh, rh = lms[23], lms[24]
                            skeleton[g_idx] = [
                                (lh.x + rh.x) / 2, (lh.y + rh.y) / 2,
                                min(lh.visibility, rh.visibility)]
                        else:
                            lm = lms[mp_idx]
                            skeleton[g_idx] = [lm.x, lm.y, lm.visibility]
                    frame_skeletons.append(skeleton)

            cap.release()

            # Sliding window to create samples
            if len(frame_skeletons) >= seq_length:
                stride = max(seq_length // 2, 1)
                for start in range(0, len(frame_skeletons) - seq_length + 1, stride):
                    seq = np.stack(frame_skeletons[start:start + seq_length])
                    all_features.append(seq)
                    all_labels.append(label_idx)

    pose.close()

    if not all_features:
        raise RuntimeError(f"No skeleton data extracted from {video_dir}")

    return np.array(all_features), np.array(all_labels)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for points, motions, labels in loader:
        points = points.to(device)
        motions = motions.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(points, motions)  # (B, num_class) — sigmoid outputs

        # Use BCE loss (model outputs are sigmoid probabilities)
        targets_onehot = torch.zeros_like(outputs)
        targets_onehot.scatter_(1, labels.unsqueeze(1), 1.0)
        loss = criterion(outputs, targets_onehot)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    for points, motions, labels in loader:
        points = points.to(device)
        motions = motions.to(device)
        labels = labels.to(device)

        outputs = model(points, motions)

        targets_onehot = torch.zeros_like(outputs)
        targets_onehot.scatter_(1, labels.unsqueeze(1), 1.0)
        loss = criterion(outputs, targets_onehot)

        total_loss += loss.item() * labels.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description='Train ST-GCN for fall detection')
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Root data directory (should contain train/ and val/)')
    parser.add_argument('--from-videos', action='store_true',
                        help='Extract skeletons from video folders instead of .npy')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Initial learning rate')
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--seq-length', type=int, default=30,
                        help='Number of frames per input sequence')
    parser.add_argument('--num-class', type=int, default=7)
    parser.add_argument('--save-dir', type=str, default='weights',
                        help='Directory to save model weights')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume training')
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device: {device}")

    os.makedirs(args.save_dir, exist_ok=True)

    # ---- Load data ----
    action_labels = DEFAULT_ACTION_LABELS[:args.num_class]

    if args.from_videos:
        print("Extracting skeletons from videos...")
        train_feats, train_labels = extract_skeletons_from_videos(
            os.path.join(args.data_dir, 'train'), action_labels, args.seq_length)
        val_feats, val_labels = extract_skeletons_from_videos(
            os.path.join(args.data_dir, 'val'), action_labels, args.seq_length)
    else:
        print("Loading pre-extracted .npy data...")
        train_feats = np.load(os.path.join(args.data_dir, 'train', 'features.npy'))
        train_labels = np.load(os.path.join(args.data_dir, 'train', 'labels.npy'))
        val_feats = np.load(os.path.join(args.data_dir, 'val', 'features.npy'))
        val_labels = np.load(os.path.join(args.data_dir, 'val', 'labels.npy'))

    print(f"Training samples: {len(train_labels)}")
    print(f"Validation samples: {len(val_labels)}")
    print(f"Label distribution (train): {np.bincount(train_labels, minlength=args.num_class)}")
    print(f"Label distribution (val):   {np.bincount(val_labels, minlength=args.num_class)}")

    train_dataset = SkeletonDataset(train_feats, train_labels, args.seq_length)
    val_dataset = SkeletonDataset(val_feats, val_labels, args.seq_length)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size,
                              shuffle=True, num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size,
                            shuffle=False, num_workers=2, pin_memory=True)

    # ---- Build model ----
    graph_cfg = {'layout': 'mediapipe', 'strategy': 'spatial'}
    model = TwoStreamSpatialTemporalGraph(
        graph_cfg=graph_cfg,
        num_class=args.num_class,
    ).to(device)

    if args.resume and os.path.isfile(args.resume):
        model.load_state_dict(torch.load(args.resume, map_location=device))
        print(f"Resumed from {args.resume}")

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model parameters: {total_params:,} total, {trainable_params:,} trainable")

    criterion = nn.BCELoss()
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr,
                               weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # ---- Training loop ----
    best_val_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion,
                                                 optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step()
        elapsed = time.time() - t0

        print(f"Epoch {epoch:3d}/{args.epochs} | "
              f"Train Loss: {train_loss:.4f}  Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}  Acc: {val_acc:.4f} | "
              f"LR: {scheduler.get_last_lr()[0]:.6f} | "
              f"Time: {elapsed:.1f}s")

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(args.save_dir, 'stgcn_best.pth')
            torch.save(model.state_dict(), save_path)
            print(f"  → Saved best model (val_acc={val_acc:.4f}) to {save_path}")

        # Periodic checkpoint
        if epoch % 10 == 0:
            ckpt_path = os.path.join(args.save_dir, f'stgcn_epoch{epoch}.pth')
            torch.save(model.state_dict(), ckpt_path)

    # Save final model
    final_path = os.path.join(args.save_dir, 'stgcn_final.pth')
    torch.save(model.state_dict(), final_path)
    print(f"\nTraining complete. Best val accuracy: {best_val_acc:.4f}")
    print(f"Final model saved to: {final_path}")


if __name__ == '__main__':
    main()
