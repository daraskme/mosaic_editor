"""検出器 + SAM2.1 Video による動画トラッキング.

SAM3 を使わない動画自動モザイクの本命パス:

1. チャンク先頭フレームで検出器 (AnimeCensor / LocateAnything) が対象を検出
2. SAM2.1 Video が各 bbox をオブジェクトとして全フレームに伝播 (追跡)
3. チャンク境界ごとに再検出するので、途中から映り込む対象も拾える

facebook/sam2.1-hiera-* は非gated・Apache-2.0。
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np
from PIL import Image

from ..core.categories import Category
from .base import Detection, ProgressCB, pick_device, pick_dtype

# 1チャンクのフレーム数。境界ごとに検出器で再検出する。
CHUNK_FRAMES = 150

# 検出関数: (キーフレーム画像, カテゴリ) -> 検出リスト
DetectFn = Callable[[Image.Image, List[Category]], List[Detection]]


class LaSam2VideoTracker:
    """キーフレーム検出 + SAM2 Video 伝播."""

    SAM2_MODEL_ID = "facebook/sam2.1-hiera-large"

    def __init__(self, detect_fn: DetectFn):
        self.detect_fn = detect_fn
        self._loaded = False
        self.device: Optional[str] = None
        self.dtype = None
        self.model = None
        self.processor = None

    def load(self, progress_cb: ProgressCB = None):
        if self._loaded:
            return
        import torch
        from transformers import Sam2VideoModel, Sam2VideoProcessor

        self.device = pick_device()
        # bf16 だと長尺の伝播でメモリアテンションが数値的に不安定になり
        # 途中で対象をロストするため fp32 固定 (モデルは ~900MB と小さい)
        self.dtype = torch.float32
        if progress_cb:
            progress_cb(f"SAM2.1 Video をロード中 (device={self.device})...")
        self.model = Sam2VideoModel.from_pretrained(
            self.SAM2_MODEL_ID, torch_dtype=self.dtype,
        ).to(self.device).eval()
        self.processor = Sam2VideoProcessor.from_pretrained(self.SAM2_MODEL_ID)
        self._loaded = True

    def track_video(
        self,
        video_path: str,
        categories: List[Category],
        progress_cb: ProgressCB = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Dict[int, np.ndarray]:
        """動画全体を処理し {frame_idx: 結合マスク uint8} を返す."""
        import cv2
        import torch

        self.load(progress_cb=progress_cb)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"動画を開けません: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        masks_by_frame: Dict[int, np.ndarray] = {}
        frame_base = 0

        while frame_base < total:
            if cancel_check and cancel_check():
                break
            # ---- チャンク分のフレームを読み込み (RGB) ----
            n = min(CHUNK_FRAMES, total - frame_base)
            if progress_cb:
                progress_cb(f"フレーム {frame_base + 1}〜{frame_base + n}/{total} を読み込み中...")
            frames = []
            for _ in range(n):
                ret, bgr = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            if not frames:
                break

            # ---- チャンク先頭フレームで検出 ----
            if progress_cb:
                progress_cb(f"フレーム {frame_base + 1}: 対象を検出中...")
            first_img = Image.fromarray(frames[0])
            detections = self.detect_fn(first_img, categories)

            if not detections:
                # このチャンクに対象なし → 次のチャンクへ
                frame_base += len(frames)
                continue

            # ---- SAM2 Video セッションに bbox を投入して伝播 ----
            h, w = frames[0].shape[:2]
            session = self.processor.init_video_session(
                video=frames,
                inference_device=self.device,
                video_storage_device="cpu",
                dtype=self.dtype,
            )
            for oi, det in enumerate(detections):
                x1, y1, x2, y2 = det.bbox
                self.processor.add_inputs_to_inference_session(
                    inference_session=session,
                    frame_idx=0,
                    obj_ids=oi + 1,
                    input_boxes=[[[float(x1), float(y1), float(x2), float(y2)]]],
                )

            if progress_cb:
                progress_cb(f"SAM2 で {len(detections)} 個の対象を追跡中 "
                            f"({frame_base + 1}〜{frame_base + len(frames)}/{total})...")
            with torch.no_grad():
                # プロンプトを与えたフレームを先に推論してから全体へ伝播する
                self.model(inference_session=session, frame_idx=0)
                for out in self.model.propagate_in_video_iterator(
                    inference_session=session,
                    start_frame_idx=0,
                    max_frame_num_to_track=len(frames),
                ):
                    if cancel_check and cancel_check():
                        break
                    video_res_masks = self.processor.post_process_masks(
                        [out.pred_masks], original_sizes=[[h, w]], binarize=True
                    )[0]
                    combined = (video_res_masks.any(dim=0).squeeze(0)
                                .cpu().numpy().astype(np.uint8) * 255)
                    if combined.any():
                        masks_by_frame[frame_base + out.frame_idx] = combined
                    if progress_cb and out.frame_idx % 20 == 0:
                        progress_cb(
                            f"追跡中... フレーム {frame_base + out.frame_idx + 1}/{total} "
                            f"(マスク {len(masks_by_frame)} フレーム)")

            del session
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            frame_base += len(frames)

        cap.release()
        return masks_by_frame

    def unload(self):
        import gc
        self.model = None
        self.processor = None
        self._loaded = False
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
