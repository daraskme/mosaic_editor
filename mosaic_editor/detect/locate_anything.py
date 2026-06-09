"""NVIDIA LocateAnything-3B バックエンド (画像 + テキスト → bbox).

- HF: nvidia/LocateAnything-3B (trust_remote_code, 非商用研究ライセンス)
- 出力: `<ref>label</ref><box><x1><y1><x2><y2></box>` (0-1000 正規化)
- generation_mode: fast (MTP) / hybrid (推奨) / slow (純AR)
"""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Tuple

from PIL import Image

from ..core.categories import Category
from .base import Detection, ProgressCB, pick_device, pick_dtype

GENERATION_MODES: Tuple[str, ...] = ("fast", "hybrid", "slow")

# <ref>label</ref> が直前にあれば取得しつつ <box> をパース
def _patch_remote_code_for_transformers_v5() -> None:
    """LocateAnything の remote code を transformers v5 で動かす互換パッチ.

    remote code (4.57.1 時代) は `_check_and_adjust_attn_implementation` を
    旧シグネチャで override しており、v5 が渡す追加キーワード引数
    (allow_all_kernels 等) で TypeError になる。動的モジュール内の該当
    override を可変長引数版に差し替える。
    """
    import sys

    from transformers.modeling_utils import PreTrainedModel

    # パッチ前の v5 本来の実装を捕まえておく (再帰防止)
    base_impl = PreTrainedModel._check_and_adjust_attn_implementation

    def _patched(self, attn_implementation, *args, **kwargs):
        if attn_implementation == "magi":
            return "magi"
        return base_impl(self, attn_implementation, *args, **kwargs)

    for name, mod in list(sys.modules.items()):
        if not name.startswith("transformers_modules"):
            continue
        for attr in dir(mod):
            obj = getattr(mod, attr, None)
            # remote code 内で定義されたクラスの override のみ差し替える
            # (re-export された transformers 本体のクラスは触らない)
            if (isinstance(obj, type)
                    and getattr(obj, "__module__", "").startswith("transformers_modules")
                    and "_check_and_adjust_attn_implementation" in vars(obj)):
                obj._check_and_adjust_attn_implementation = _patched

    # v5 では rope_theta が config.rope_parameters に移動したが、
    # remote code は config.rope_theta を直接読むため property で補完する。
    # 注意: v5 の config 構築中にもレガシー属性として probe されるため、
    # 値が未設定の間は AttributeError を投げて probe を素通りさせる。
    from transformers.models.qwen2.configuration_qwen2 import Qwen2Config

    def _rope_theta(self):
        rp = self.__dict__.get("rope_parameters")
        if isinstance(rp, dict) and "rope_theta" in rp:
            return rp["rope_theta"]
        raise AttributeError("rope_theta")

    if not isinstance(getattr(Qwen2Config, "rope_theta", None), property):
        Qwen2Config.rope_theta = property(_rope_theta)


_BOX_RE = re.compile(
    r"(?:<ref>(?P<ref>[^<]*)</ref>\s*)?"
    r"<box><(?P<x1>\d+)><(?P<y1>\d+)><(?P<x2>\d+)><(?P<y2>\d+)></box>"
)


class LocateAnythingDetector:
    """LocateAnything-3B のラッパー。初回 detect 時にモデルをロードする."""

    MODEL_ID = "nvidia/LocateAnything-3B"

    def __init__(self):
        self._loaded = False
        self.device: Optional[str] = None
        self.dtype = None
        self.tokenizer = None
        self.processor = None
        self.model = None

    def load(self, progress_cb: ProgressCB = None):
        if self._loaded:
            return
        from transformers import AutoProcessor, AutoTokenizer
        from transformers.dynamic_module_utils import get_class_from_dynamic_module

        self.device = pick_device()
        self.dtype = pick_dtype(self.device)

        if progress_cb:
            progress_cb(f"LocateAnything-3B をロード中 (device={self.device})...\n"
                        "初回は ~8GB のダウンロードが発生します")
        # remote code は transformers==4.57.1 を前提としているため、
        # クラスを先に解決して v5 互換パッチを当ててからロードする
        model_cls = get_class_from_dynamic_module(
            "modeling_locateanything.LocateAnythingForConditionalGeneration",
            self.MODEL_ID)
        _patch_remote_code_for_transformers_v5()

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_ID, trust_remote_code=True)
        self.processor = AutoProcessor.from_pretrained(
            self.MODEL_ID, trust_remote_code=True)
        self.model = model_cls.from_pretrained(
            self.MODEL_ID,
            torch_dtype=self.dtype,
            trust_remote_code=True,
        ).to(self.device).eval()
        self._loaded = True

    def detect(
        self,
        image: Image.Image,
        categories: List[Category],
        generation_mode: str = "hybrid",
        max_new_tokens: int = 8192,
        progress_cb: ProgressCB = None,
    ) -> List[Detection]:
        """1回の generate で全カテゴリの bbox を検出する.

        プロンプトには各カテゴリの先頭プロンプトを `</c>` 連結で渡し、
        出力の <ref> ラベルからカテゴリを逆引きする。
        """
        self.load(progress_cb=progress_cb)
        if image.mode != "RGB":
            image = image.convert("RGB")
        w, h = image.size

        # プロンプト文字列 → カテゴリの逆引きテーブル
        phrase_to_cat: Dict[str, Category] = {}
        phrases: List[str] = []
        for cat in categories:
            for p in cat.prompts:
                phrase_to_cat[p.lower()] = cat
                phrases.append(p)

        joined = "</c>".join(phrases)
        if progress_cb:
            progress_cb("LocateAnything-3B 推論中...")

        text_out = self._predict(image, joined, generation_mode, max_new_tokens)

        detections: List[Detection] = []
        for m in _BOX_RE.finditer(text_out):
            x1, y1, x2, y2 = (int(m.group(k)) for k in ("x1", "y1", "x2", "y2"))
            px1 = max(0, min(w - 1, round(x1 / 1000 * w)))
            py1 = max(0, min(h - 1, round(y1 / 1000 * h)))
            px2 = max(0, min(w, round(x2 / 1000 * w)))
            py2 = max(0, min(h, round(y2 / 1000 * h)))
            if px2 <= px1 or py2 <= py1:
                continue
            ref = (m.group("ref") or "").strip().lower()
            cat = phrase_to_cat.get(ref)
            detections.append(Detection(
                label=cat.label if cat else (ref or "検出"),
                category_key=cat.key if cat else "unknown",
                bbox=(px1, py1, px2, py2),
                score=1.0,  # LocateAnything はスコアを返さない
            ))
        return detections

    def _predict(self, image: Image.Image, category: str,
                 generation_mode: str, max_new_tokens: int) -> str:
        import torch

        prompt = ("Locate all the instances that matches the following "
                  f"description: {category}.")
        messages = [
            {"role": "user", "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": prompt},
            ]}
        ]

        with torch.no_grad():
            text = self.processor.py_apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True)
            images, videos = self.processor.process_vision_info(messages)
            inputs = self.processor(
                text=[text], images=images, videos=videos, return_tensors="pt"
            ).to(self.device)

            response = self.model.generate(
                pixel_values=inputs["pixel_values"].to(self.dtype),
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
        return response[0] if isinstance(response, (tuple, list)) else response

    def unload(self):
        """VRAM 解放."""
        import gc
        self.model = None
        self.processor = None
        self.tokenizer = None
        self._loaded = False
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass
