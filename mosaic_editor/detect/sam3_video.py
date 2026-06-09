"""SAM3 Video — テキストプロンプトで動画全体を検出 + トラッキング.

旧版の「全フレームに毎回検出をかける」方式と違い、SAM3 の動画モデルは
一度見つけた対象に ID を割り当てて追跡するため、高速かつ時間方向に
一貫したマスクが得られる。

メモリ対策: フレームは CPU 側に保持し (video_storage_device="cpu")、
長い動画はチャンク単位で処理する。
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional

import numpy as np

from ..core.categories import Category
from .base import ProgressCB, pick_device

# 1チャンクで処理する最大フレーム数 (メモリと精度のバランス)
CHUNK_FRAMES = 300


class Sam3VideoTracker:
    MODEL_ID = "facebook/sam3"

    def __init__(self):
        self._loaded = False
        self.device: Optional[str] = None
        self.model = None
        self.processor = None

    def load(self, progress_cb: ProgressCB = None):
        if self._loaded:
            return
        from transformers import Sam3VideoModel, Sam3VideoProcessor

        self.device = pick_device()
        if progress_cb:
            progress_cb(f"SAM3 Video をロード中 (device={self.device})...")
        self.model = Sam3VideoModel.from_pretrained(self.MODEL_ID)
        self.model = self.model.to(self.device).eval()
        self.processor = Sam3VideoProcessor.from_pretrained(self.MODEL_ID)
        self._loaded = True

    def track_video(
        self,
        video_path: str,
        categories: List[Category],
        progress_cb: ProgressCB = None,
        cancel_check: Optional[Callable[[], bool]] = None,
    ) -> Dict[int, np.ndarray]:
        """動画ファイル全体を処理し {frame_idx: 結合マスク uint8} を返す.

        カテゴリの全プロンプトを同時にトラッキングし、フレームごとに
        全オブジェクトのマスクを 1 枚に統合する。
        """
        import cv2
        import torch

        self.load(progress_cb=progress_cb)

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise RuntimeError(f"動画を開けません: {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        prompts: List[str] = []
        for cat in categories:
            prompts.extend(cat.prompts)

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
            video = np.stack(frames)

            # ---- セッション初期化 + テキストプロンプト ----
            session = self.processor.init_video_session(
                video=video,
                inference_device=self.device,
                processing_device="cpu",
                video_storage_device="cpu",
                dtype=torch.bfloat16 if self.device == "cuda" else torch.float32,
            )
            self.processor.add_text_prompt(inference_session=session, text=prompts)

            # ---- 伝播 (トラッキング) ----
            for model_outputs in self.model.propagate_in_video_iterator(
                inference_session=session, max_frame_num_to_track=len(frames)
            ):
                if cancel_check and cancel_check():
                    break
                out = self.processor.postprocess_outputs(session, model_outputs)
                fi = frame_base + model_outputs.frame_idx
                obj_masks = out.get("masks")
                if obj_masks is not None and len(obj_masks) > 0:
                    if hasattr(obj_masks, "cpu"):
                        obj_masks = obj_masks.cpu().numpy()
                    combined = (np.any(np.asarray(obj_masks) > 0, axis=0)
                                .astype(np.uint8) * 255)
                    if combined.any():
                        masks_by_frame[fi] = combined
                if progress_cb and model_outputs.frame_idx % 10 == 0:
                    progress_cb(f"トラッキング中... フレーム {fi + 1}/{total} "
                                f"(検出 {len(masks_by_frame)} フレーム)")

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
