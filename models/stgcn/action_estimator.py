"""
Action Estimator — wraps TwoStreamSpatialTemporalGraph for inference.

Handles:
  - Converting MediaPipe landmarks into the (C, T, V) tensor format
  - Normalizing and scaling the skeleton
  - Computing motion (frame-to-frame diffs) as the second stream
  - Running the ST-GCN model and returning action probabilities

Usage:
    estimator = ActionEstimator(weights_path='path/to/stgcn.pth', device='cuda')
    # For each frame, call estimator.add_frame(landmarks) where
    # landmarks is the raw MediaPipe landmark list of 33 points.
    # When you have enough frames (30), call estimator.predict().
"""

import os
import numpy as np
import torch

from .graph import MEDIAPIPE_KEY_INDICES, MEDIAPIPE_TO_GRAPH
from .stgcn_model import TwoStreamSpatialTemporalGraph


# Default action labels (matching the reference training set)
DEFAULT_ACTION_LABELS = [
    'Standing',
    'Walking',
    'Sitting',
    'Lying Down',
    'Stand up',
    'Sit down',
    'Fall Down',
]

# Index of the "Fall Down" class
FALL_CLASS_INDEX = 6


class ActionEstimator:
    """Skeleton-based action recognition using ST-GCN.

    Maintains a rolling buffer of skeleton frames for each tracked person
    and performs inference with TwoStreamSpatialTemporalGraph.

    Args:
        weights_path (str): Path to the trained ST-GCN weights (.pth file).
        device (str): Device to use for inference ('cuda' or 'cpu').
        num_class (int): Number of action classes.
        seq_length (int): Number of frames in one input sequence.
        graph_cfg (dict): Graph configuration passed to the model.
        action_labels (list): Human-readable labels for each class.
    """

    def __init__(
        self,
        weights_path=None,
        device='cpu',
        num_class=7,
        seq_length=30,
        graph_cfg=None,
        action_labels=None,
    ):
        self.device = device
        self.num_class = num_class
        self.seq_length = seq_length
        self.num_joints = 14  # 13 MediaPipe joints + 1 computed mid-hip
        self.action_labels = action_labels or DEFAULT_ACTION_LABELS

        if graph_cfg is None:
            graph_cfg = {'layout': 'mediapipe', 'strategy': 'spatial'}

        # Build model
        self.model = TwoStreamSpatialTemporalGraph(
            graph_cfg=graph_cfg,
            num_class=num_class,
        )

        # Load weights if provided
        if weights_path and os.path.isfile(weights_path):
            state_dict = torch.load(weights_path, map_location=self.device)
            self.model.load_state_dict(state_dict)
            print(f"[ActionEstimator] Loaded weights from {weights_path}")
        else:
            print("[ActionEstimator] No weights loaded — model is untrained. "
                  "Predictions will be random until you train or provide weights.")

        self.model.to(self.device)
        self.model.eval()

        # Per-person skeleton frame buffers: {person_id: list of (14, 3) arrays}
        self._buffers = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add_frame(self, person_id, landmarks):
        """Add one frame of MediaPipe landmarks for a tracked person.

        Args:
            person_id: Unique ID of the tracked person.
            landmarks: List of 33 (x, y, z, visibility) tuples from MediaPipe,
                       or ``None`` if no pose was detected this frame.
        """
        if landmarks is None:
            return

        skeleton = self._extract_skeleton(landmarks)  # (14, 3)
        if skeleton is None:
            return

        if person_id not in self._buffers:
            self._buffers[person_id] = []

        self._buffers[person_id].append(skeleton)

        # Keep at most seq_length frames
        if len(self._buffers[person_id]) > self.seq_length:
            self._buffers[person_id].pop(0)

    def is_ready(self, person_id):
        """Check whether we have enough frames to run inference."""
        return (person_id in self._buffers and
                len(self._buffers[person_id]) >= self.seq_length)

    def predict(self, person_id):
        """Run ST-GCN inference for a person.

        Returns:
            dict with keys:
                'action': str — predicted action label
                'action_index': int — class index
                'probabilities': np.ndarray — per-class probabilities
                'is_fall': bool — whether the predicted action is "Fall Down"
            or ``None`` if there are not enough frames.
        """
        if not self.is_ready(person_id):
            return None

        frames = self._buffers[person_id][-self.seq_length:]  # list of (14, 3)

        # Build tensors
        points, motions = self._build_input_tensors(frames)

        with torch.no_grad():
            probs = self.model(points, motions)  # (1, num_class)

        probs_np = probs.cpu().numpy().flatten()
        action_idx = int(np.argmax(probs_np))

        return {
            'action': self.action_labels[action_idx],
            'action_index': action_idx,
            'probabilities': probs_np,
            'is_fall': action_idx == FALL_CLASS_INDEX,
        }

    def clear_person(self, person_id):
        """Remove the frame buffer for a person (e.g. when they leave the scene)."""
        self._buffers.pop(person_id, None)

    def clear_all(self):
        """Remove all frame buffers."""
        self._buffers.clear()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_skeleton(self, landmarks):
        """Convert 33 MediaPipe landmarks to our 14-joint skeleton.

        Returns:
            np.ndarray of shape (14, 3) — (x, y, score/visibility)
            or ``None`` if key landmarks are missing.
        """
        if len(landmarks) < 29:  # Need at least up to ankle (index 28)
            return None

        skeleton = np.zeros((self.num_joints, 3), dtype=np.float32)

        for graph_idx, mp_idx in MEDIAPIPE_TO_GRAPH.items():
            if mp_idx == -1:
                # Mid-hip: average of left hip (23) and right hip (24)
                lh = landmarks[23]
                rh = landmarks[24]
                skeleton[graph_idx] = [
                    (lh[0] + rh[0]) / 2.0,
                    (lh[1] + rh[1]) / 2.0,
                    min(lh[3] if len(lh) > 3 else 1.0,
                        rh[3] if len(rh) > 3 else 1.0),
                ]
            else:
                lm = landmarks[mp_idx]
                skeleton[graph_idx] = [
                    lm[0],
                    lm[1],
                    lm[3] if len(lm) > 3 else 1.0,  # visibility as score
                ]

        return skeleton

    def _build_input_tensors(self, frames):
        """Build the two-stream input tensors from a list of skeleton frames.

        The reference implementation normalizes the skeleton relative to the
        mid-hip center and scales it by the torso length.

        Args:
            frames: list of T np.ndarrays, each of shape (14, 3)

        Returns:
            points: torch.Tensor of shape (1, 3, T, 14)
            motions: torch.Tensor of shape (1, 2, T, 14)
        """
        T = len(frames)
        V = self.num_joints

        # Stack into (T, V, C=3)
        data = np.stack(frames, axis=0).astype(np.float32)  # (T, 14, 3)

        # ---- Normalize: center on mid-hip and scale by torso length ----
        # Mid-hip is node 13
        center = data[:, 13:14, :2]  # (T, 1, 2)
        data[:, :, :2] = data[:, :, :2] - center  # center on hip

        # Scale by torso length (mid-hip to nose distance)
        # nose=0, mid-hip=13
        torso = np.linalg.norm(data[:, 0, :2] - data[:, 13, :2], axis=-1)  # (T,)
        torso_mean = max(torso.mean(), 1e-4)
        data[:, :, :2] /= torso_mean

        # ---- Permute to (C, T, V) ----
        points = data.transpose(2, 0, 1)  # (3, T, V)

        # ---- Compute motions (frame diffs on x,y) ----
        # motions[:, t, v] = points[:2, t, v] - points[:2, t-1, v]
        motions = np.zeros((2, T, V), dtype=np.float32)
        motions[:, 1:, :] = points[:2, 1:, :] - points[:2, :-1, :]

        # Add batch dimension and convert to tensors
        points_t = torch.from_numpy(points).unsqueeze(0).to(self.device)   # (1, 3, T, V)
        motions_t = torch.from_numpy(motions).unsqueeze(0).to(self.device)  # (1, 2, T, V)

        return points_t, motions_t
