"""自動モザイク検出パイプライン: LocateAnything-3B (NVIDIA) + SAM (Meta).

- LocateAnything-3B: 画像 + テキスト → bounding box (vision-language grounding).
  ライセンスは非商用研究目的のみ. 約 6-8GB のモデルをHFキャッシュに DL する.
- facebook/sam-vit-base: bbox → 輪郭マスクへ再セグメント. Apache 2.0.

CUDA + NVIDIA GPU (Ampere 以降, VRAM 16GB+) を推奨. CPU / Apple MPS でも
動作はするが推論が遅い (1枚あたり数十秒〜数分).
"""
from __future__ import annotations

import re
from typing import List, Tuple, Optional, Dict
import numpy as np
from PIL import Image


DEFAULT_CATEGORIES: List[str] = [
    "penis",
    "pussy",
    "vagina",
    "anus",
    "nipples",
    "testicles",
]


GENERATION_MODES: Tuple[str, ...] = ("fast", "hybrid", "slow")


def _pick_device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _pick_dtype(device: str):
    import torch
    if device == "cuda":
        return torch.bfloat16
    if device == "mps":
        return torch.float16
    return torch.float32


class LocateAnythingDetector:
    """LocateAnything-3B のラッパー. 初回 detect 時にモデルをロードする."""

    MODEL_ID = "nvidia/LocateAnything-3B"

    def __init__(self):
        self._loaded = False
        self.device: Optional[str] = None
        self.dtype = None
        self.tokenizer = None
        self.processor = None
        self.model = None

    def load(self, progress_cb=None):
        if self._loaded:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer, AutoProcessor

        self.device = _pick_device()
        self.dtype = _pick_dtype(self.device)

        if progress_cb:
            progress_cb(f"tokenizer をロード中 (device={self.device}, dtype={self.dtype})...")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_ID, trust_remote_code=True
        )
        if progress_cb:
            progress_cb("processor をロード中...")
        self.processor = AutoProcessor.from_pretrained(
            self.MODEL_ID, trust_remote_code=True
        )
        if progress_cb:
            progress_cb("LocateAnything-3B 本体をロード中 (~6GB, 初回DLは時間がかかります)...")
        self.model = AutoModel.from_pretrained(
            self.MODEL_ID,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device).eval()
        self._loaded = True

    def detect(
        self,
        image: Image.Image,
        categories: List[str],
        generation_mode: str = "hybrid",
        max_new_tokens: int = 2048,
        progress_cb=None,
    ) -> List[Dict]:
        """単一の推論で複数カテゴリを検出して bbox 集合を返す.

        LocateAnything-3B は `cat1</c>cat2</c>...` 形式の連結プロンプトを
        受け付けるので、1 回の generate で全候補を得ることでスループットを稼ぐ.

        戻り値: [{"label": str, "bbox": (x1,y1,x2,y2), "score": float}]
        """
        self.load(progress_cb=progress_cb)
        if image.mode != "RGB":
            image = image.convert("RGB")
        w, h = image.size

        joined = "</c>".join(categories)
        label = "/".join(categories)
        if progress_cb:
            progress_cb(f"LocateAnything-3B 推論中 ({joined}) ...")

        try:
            text_out = self._predict(image, joined, generation_mode, max_new_tokens)
        except Exception as e:
            print(f"[LocateAnything] inference failed: {e}")
            return []

        return [
            {"label": label, "bbox": b, "score": 1.0}
            for b in self._parse_boxes(text_out, w, h)
        ]

    def _predict(
        self,
        image: Image.Image,
        category: str,
        generation_mode: str,
        max_new_tokens: int,
    ) -> str:
        import torch

        prompt = (
            "Locate all the instances that matches the following description: "
            f"{category}."
        )
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]}
        ]

        with torch.no_grad():
            text = self.processor.py_apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            images, videos = self.processor.process_vision_info(messages)
            inputs = self.processor(
                text=[text], images=images, videos=videos, return_tensors="pt"
            ).to(self.device)

            pixel_values = inputs["pixel_values"].to(self.dtype)
            response = self.model.generate(
                pixel_values=pixel_values,
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                image_grid_hws=inputs.get("image_grid_hws", None),
                tokenizer=self.tokenizer,
                max_new_tokens=max_new_tokens,
                use_cache=True,
                generation_mode=generation_mode,
                temperature=0.7,
                do_sample=True,
                top_p=0.9,
                repetition_penalty=1.1,
            )
        return response[0] if isinstance(response, tuple) else response

    @staticmethod
    def _parse_boxes(answer: str, w: int, h: int) -> List[Tuple[int, int, int, int]]:
        """`<box><x1><y1><x2><y2></box>` (0-1000 正規化) をピクセル座標に変換."""
        boxes: List[Tuple[int, int, int, int]] = []
        for m in re.finditer(r"<box><(\d+)><(\d+)><(\d+)><(\d+)></box>", answer):
            x1, y1, x2, y2 = (int(g) for g in m.groups())
            px1 = max(0, min(w - 1, int(round(x1 / 1000 * w))))
            py1 = max(0, min(h - 1, int(round(y1 / 1000 * h))))
            px2 = max(0, min(w, int(round(x2 / 1000 * w))))
            py2 = max(0, min(h, int(round(y2 / 1000 * h))))
            if px2 > px1 and py2 > py1:
                boxes.append((px1, py1, px2, py2))
        return boxes


class SamSegmenter:
    """facebook/sam-vit-base (transformers) を使って bbox → 輪郭マスク."""

    MODEL_ID = "facebook/sam-vit-base"

    def __init__(self):
        self._loaded = False
        self.device: Optional[str] = None
        self.processor = None
        self.model = None

    def load(self, progress_cb=None):
        if self._loaded:
            return
        from transformers import SamModel, SamProcessor

        self.device = _pick_device()
        if progress_cb:
            progress_cb(f"SAM (sam-vit-base) をロード中 (device={self.device})...")
        self.processor = SamProcessor.from_pretrained(self.MODEL_ID)
        self.model = SamModel.from_pretrained(self.MODEL_ID).to(self.device).eval()
        self._loaded = True

    def segment_box(
        self,
        image: Image.Image,
        box: Tuple[int, int, int, int],
    ) -> Optional[np.ndarray]:
        """bbox を SAM に渡して輪郭マスク (np.uint8 H×W, 0/255) を返す."""
        import torch

        self.load()
        if image.mode != "RGB":
            image = image.convert("RGB")

        x1, y1, x2, y2 = box
        inputs = self.processor(
            image,
            input_boxes=[[[float(x1), float(y1), float(x2), float(y2)]]],
            return_tensors="pt",
        ).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)

        masks = self.processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )
        # masks[0]: tensor (1, 3, H, W); pick highest predicted IoU
        scores = outputs.iou_scores.cpu()[0, 0]  # (3,)
        best = int(scores.argmax())
        mask = masks[0][0, best].numpy().astype(bool)
        return (mask.astype(np.uint8) * 255)


class AutoMosaicPipeline:
    """LocateAnything-3B 検出 → SAM 再セグメントを束ねるパイプライン."""

    def __init__(self):
        self.detector = LocateAnythingDetector()
        self.segmenter = SamSegmenter()

    def detect_and_segment(
        self,
        image: Image.Image,
        categories: List[str],
        generation_mode: str = "hybrid",
        use_segmenter: bool = True,
        progress_cb=None,
    ) -> List[Dict]:
        """戻り値: [{"label", "bbox", "score", "mask"}]. mask は np.uint8 か None."""
        detections = self.detector.detect(
            image, categories, generation_mode=generation_mode, progress_cb=progress_cb
        )
        if not use_segmenter:
            for d in detections:
                d["mask"] = None
            return detections
        for di, d in enumerate(detections):
            if progress_cb:
                progress_cb(
                    f"SAM 再セグメント中 [{di + 1}/{len(detections)}] ({d['label']})..."
                )
            try:
                d["mask"] = self.segmenter.segment_box(image, d["bbox"])
            except Exception as e:
                print(f"[SAM] segmentation failed for {d['bbox']}: {e}")
                d["mask"] = None
        return detections

    @staticmethod
    def combine_masks(
        detections: List[Dict],
        image_size: Tuple[int, int],
    ) -> np.ndarray:
        """全ての検出結果を 1 枚の uint8 マスクにマージ.

        SAM マスクが利用可能ならそれを、無ければ bbox を矩形塗りで使用.
        """
        w, h = image_size
        out = np.zeros((h, w), dtype=np.uint8)
        for d in detections:
            mask = d.get("mask")
            if mask is not None and mask.shape == (h, w):
                out[mask > 127] = 255
            else:
                x1, y1, x2, y2 = d["bbox"]
                out[y1:y2, x1:x2] = 255
        return out
