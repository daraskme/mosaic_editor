"""NVIDIA LocateAnything-3B バックエンド (画像 + テキスト → bbox).

- HF: nvidia/LocateAnything-3B (trust_remote_code, 非商用研究ライセンス)
- 出力: `<ref>label</ref><box><x1><y1><x2><y2></box>` (0-1000 正規化)
- generation_mode: fast (MTP) / hybrid (推奨) / slow (純AR)

LocateAnything の remote code は transformers==4.57.1 を前提としており、
SAM3 が必要とする v5 系とは同一プロセスで共存できない (フォワード経路が
壊れて出力が崩壊する)。そのため transformers 4.57.1 だけを
`vendor/transformers_la/` に隔離インストールし、推論は常駐サブプロセス
(`la_worker.py`) で行う。モデルのロードはワーカー起動時の1回のみ。
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

from PIL import Image

from ..core.categories import Category
from .base import Detection, ProgressCB

GENERATION_MODES: Tuple[str, ...] = ("fast", "hybrid", "slow")

_PKG_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PROJECT_DIR = os.path.dirname(_PKG_DIR)
VENDOR_DIR = os.path.join(_PROJECT_DIR, "vendor", "transformers_la")
WORKER_PATH = os.path.join(_PKG_DIR, "detect", "la_worker.py")

# <ref>label</ref> が直前にあれば取得しつつ <box> をパース
_BOX_RE = re.compile(
    r"(?:<ref>(?P<ref>[^<]*)</ref>\s*)?"
    r"<box><(?P<x1>\d+)><(?P<y1>\d+)><(?P<x2>\d+)><(?P<y2>\d+)></box>"
)


def _ensure_vendor_transformers(progress_cb: ProgressCB = None) -> None:
    """vendor/transformers_la に transformers==4.57.1 を隔離インストールする.

    メイン環境の新しい kernels / huggingface_hub が紛れ込むと 4.57.1 と
    非互換なので、互換版 kernels も同じ場所に入れて遮蔽する。
    """
    if not os.path.isdir(os.path.join(VENDOR_DIR, "transformers")):
        if progress_cb:
            progress_cb("LocateAnything 用に transformers 4.57.1 を準備中...")
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "--target", VENDOR_DIR,
            "transformers==4.57.1", "huggingface_hub<1.0",
        ])
    if not os.path.isdir(os.path.join(VENDOR_DIR, "kernels")):
        subprocess.check_call([
            sys.executable, "-m", "pip", "install",
            "--target", VENDOR_DIR, "--no-deps", "--upgrade",
            "kernels==0.10.3",
        ])


class LocateAnythingDetector:
    """LocateAnything-3B のラッパー。初回 detect 時にワーカーを起動する."""

    MODEL_ID = "nvidia/LocateAnything-3B"

    def __init__(self):
        self._proc: Optional[subprocess.Popen] = None
        self._log_path: Optional[str] = None
        self._log_file = None

    def _read_log_tail(self, n: int = 15) -> str:
        """ワーカーログの末尾を返す (エラー表示用)."""
        try:
            with open(self._log_path, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()
            return "".join(lines[-n:]).strip()
        except Exception:
            return "(ログを読み取れませんでした)"

    @property
    def _loaded(self) -> bool:
        return self._proc is not None and self._proc.poll() is None

    def load(self, progress_cb: ProgressCB = None):
        if self._loaded:
            return
        _ensure_vendor_transformers(progress_cb)
        if progress_cb:
            progress_cb("LocateAnything-3B ワーカーを起動中...\n"
                        "初回は ~8GB のダウンロードが発生します")
        env = os.environ.copy()
        env["PYTHONPATH"] = VENDOR_DIR + os.pathsep + env.get("PYTHONPATH", "")
        self._log_path = os.path.join(
            tempfile.gettempdir(), "mosaic_la_worker.log")
        self._log_file = open(self._log_path, "w", encoding="utf-8")
        self._proc = subprocess.Popen(
            [sys.executable, WORKER_PATH],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE,
            stderr=self._log_file,
            text=True, encoding="utf-8", env=env,
        )
        # ready 通知を待つ
        line = self._proc.stdout.readline()
        if not line:
            rc = self._proc.poll()
            self._proc = None
            raise RuntimeError(
                f"LocateAnything ワーカーの起動に失敗しました (exit={rc})\n"
                f"{self._read_log_tail()}")
        msg = json.loads(line)
        if not msg.get("ready"):
            self.unload()
            raise RuntimeError(f"LocateAnything ワーカー起動エラー: {msg}")

    def _request(self, payload: dict) -> dict:
        assert self._proc is not None
        self._proc.stdin.write(json.dumps(payload) + "\n")
        self._proc.stdin.flush()
        line = self._proc.stdout.readline()
        if not line:
            rc = self._proc.poll()
            self._proc = None
            raise RuntimeError(
                f"LocateAnything ワーカーが終了しました (exit={rc})\n"
                f"{self._read_log_tail()}")
        return json.loads(line)

    def detect(
        self,
        image: Image.Image,
        categories: List[Category],
        generation_mode: str = "hybrid",
        max_new_tokens: int = 8192,
        progress_cb: ProgressCB = None,
    ) -> List[Detection]:
        """1回の推論で全カテゴリの bbox を検出する.

        プロンプトには各カテゴリの全プロンプトを `</c>` 連結で渡し、
        出力の <ref> ラベルからカテゴリを逆引きする。
        """
        self.load(progress_cb=progress_cb)
        if image.mode != "RGB":
            image = image.convert("RGB")
        w, h = image.size

        phrase_to_cat: Dict[str, Category] = {}
        phrases: List[str] = []
        for cat in categories:
            for p in cat.prompts:
                phrase_to_cat[p.lower()] = cat
                phrases.append(p)

        if progress_cb:
            progress_cb("LocateAnything-3B 推論中...")

        fd, img_path = tempfile.mkstemp(suffix=".png")
        os.close(fd)
        try:
            image.save(img_path, "PNG")
            resp = self._request({
                "image": img_path,
                "prompt": "</c>".join(phrases),
                "generation_mode": generation_mode,
                "max_new_tokens": max_new_tokens,
            })
        finally:
            try:
                os.remove(img_path)
            except OSError:
                pass

        if not resp.get("ok"):
            raise RuntimeError(f"LocateAnything 推論エラー: {resp.get('error')}")

        detections: List[Detection] = []
        for m in _BOX_RE.finditer(resp["raw"]):
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

    def unload(self):
        """ワーカーを終了して VRAM を解放する."""
        if self._proc is not None:
            try:
                if self._proc.poll() is None:
                    self._proc.stdin.write(json.dumps({"exit": True}) + "\n")
                    self._proc.stdin.flush()
                    self._proc.wait(timeout=10)
            except Exception:
                self._proc.kill()
            self._proc = None
        if self._log_file is not None:
            try:
                self._log_file.close()
            except Exception:
                pass
            self._log_file = None

    def __del__(self):
        try:
            self.unload()
        except Exception:
            pass
