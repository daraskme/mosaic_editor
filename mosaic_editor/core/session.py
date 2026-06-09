"""ファイルリスト・出力フォルダ・マスク永続化の管理 (UI 非依存)."""
from __future__ import annotations

import os
from typing import List, Optional

import numpy as np

SUPPORTED_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
SUPPORTED_VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm")
ALL_SUPPORTED_EXT = SUPPORTED_EXT + SUPPORTED_VIDEO_EXT


def collect_files(paths: List[str]) -> List[str]:
    """ファイル/フォルダのパス群から対応ファイル一覧を集める."""
    files: List[str] = []
    for p in paths:
        p = p.strip()
        if os.path.isdir(p):
            for root_dir, _, filenames in os.walk(p):
                for f in filenames:
                    if f.lower().endswith(ALL_SUPPORTED_EXT):
                        files.append(os.path.join(root_dir, f))
        elif os.path.isfile(p) and p.lower().endswith(ALL_SUPPORTED_EXT):
            files.append(p)
    files.sort()
    return files


def default_output_folder(base_path: str) -> str:
    """入力パス (ファイル or フォルダ) から `_mc` 出力フォルダを決める."""
    if os.path.isdir(base_path):
        out = base_path.rstrip("/\\") + "_mc"
    else:
        out = os.path.dirname(base_path) + "_mc"
    os.makedirs(out, exist_ok=True)
    return out


class Session:
    """開いているファイル一覧と出力先・マスクファイルの管理."""

    def __init__(self):
        self.files: List[str] = []
        self.index: int = 0
        self.output_folder: Optional[str] = None
        self.skip_set: set = set()

    @property
    def current(self) -> Optional[str]:
        if not self.files:
            return None
        return self.files[self.index]

    def open_paths(self, paths: List[str]) -> bool:
        files = collect_files(paths)
        if not files:
            return False
        self.files = files
        self.index = 0
        self.output_folder = default_output_folder(paths[0])
        return True

    # ---- マスク永続化 ----

    def _masks_dir(self) -> Optional[str]:
        if self.output_folder is None:
            return None
        d = os.path.join(self.output_folder, "masks")
        os.makedirs(d, exist_ok=True)
        return d

    def mask_path(self, img_path: str) -> Optional[str]:
        d = self._masks_dir()
        if d is None:
            return None
        base = os.path.splitext(os.path.basename(img_path))[0]
        return os.path.join(d, base + ".mask.npz")

    def load_mask(self, img_path: str, w: int, h: int) -> Optional[np.ndarray]:
        """保存済みマスクを読み込む。無ければ None."""
        import cv2
        p = self.mask_path(img_path)
        if not p or not os.path.exists(p):
            return None
        try:
            data = np.load(p)
            loaded = data["mask"]
            if loaded.shape != (h, w):
                loaded = cv2.resize(loaded, (w, h), interpolation=cv2.INTER_NEAREST)
            return loaded.copy()
        except Exception:
            return None

    def save_mask(self, img_path: str, mask: Optional[np.ndarray]) -> None:
        p = self.mask_path(img_path)
        if not p:
            return
        if mask is not None and np.any(mask):
            np.savez_compressed(p, mask=mask)
        elif os.path.exists(p):
            try:
                os.remove(p)
            except Exception:
                pass

    # ---- スキップマーカー ----

    def skip_marker_path(self, img_path: str) -> Optional[str]:
        d = self._masks_dir()
        if d is None:
            return None
        base = os.path.splitext(os.path.basename(img_path))[0]
        return os.path.join(d, base + ".skip")

    def is_skipped(self, img_path: str) -> bool:
        return img_path in self.skip_set

    def load_skip_state(self, img_path: str) -> bool:
        marker = self.skip_marker_path(img_path)
        if marker and os.path.exists(marker):
            self.skip_set.add(img_path)
            return True
        self.skip_set.discard(img_path)
        return False

    def set_skipped(self, img_path: str, skipped: bool) -> None:
        marker = self.skip_marker_path(img_path)
        if skipped:
            self.skip_set.add(img_path)
            if marker:
                try:
                    open(marker, "w").close()
                except Exception:
                    pass
        else:
            self.skip_set.discard(img_path)
            if marker and os.path.exists(marker):
                try:
                    os.remove(marker)
                except Exception:
                    pass

    def output_jpg_path(self, img_path: str) -> Optional[str]:
        if self.output_folder is None:
            return None
        base = os.path.splitext(os.path.basename(img_path))[0]
        return os.path.join(self.output_folder, base + ".jpg")
