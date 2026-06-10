"""自動検出の設定ダイアログと検出結果レビューダイアログ."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Optional

from ..core.categories import Category
from ..detect.base import Detection
from .progress import safe_grab


class DetectConfig:
    """ユーザーが選んだ検出設定 (ダイアログの戻り値)."""

    def __init__(self):
        self.categories: List[Category] = []
        self.threshold: float = 0.3
        self.margin_px: int = 4
        self.process_all_frames: bool = False
        self.overwrite: bool = False


class DetectConfigDialog:
    """検出設定ダイアログ。状態 (チェック等) は呼び出し側が保持する."""

    def __init__(self, parent,
                 all_categories: List[Category],
                 enabled_keys: Dict[str, bool],
                 threshold: float = 0.3,
                 margin_px: int = 4,
                 is_video_mode: bool = False,
                 folder_mode: bool = False,
                 folder_image_count: int = 0):
        self.parent = parent
        self.all_categories = list(all_categories)
        self.enabled_keys = enabled_keys
        self.threshold = threshold
        self.margin_px = margin_px
        self.is_video_mode = is_video_mode
        self.folder_mode = folder_mode
        self.folder_image_count = folder_image_count
        self.result: Optional[DetectConfig] = None

    def show(self) -> Optional[DetectConfig]:
        win = tk.Toplevel(self.parent)
        title = "自動モザイク (フォルダ一括)" if self.folder_mode else "自動検出"
        win.title(title)
        win.geometry("560x440")
        win.minsize(480, 360)
        win.resizable(True, True)
        safe_grab(win)

        if self.folder_mode:
            tk.Label(win,
                     text=f"フォルダ内の画像 {self.folder_image_count} 枚に自動モザイクを適用します",
                     font=("", 10), pady=6).pack()

        # ---- カテゴリ ----
        cat_frame = tk.LabelFrame(win, text="検出対象")
        cat_frame.pack(fill="x", padx=10, pady=(8, 4))
        cls_vars: Dict[str, tk.BooleanVar] = {}
        for cat in self.all_categories:
            var = tk.BooleanVar(
                value=self.enabled_keys.get(cat.key, cat.enabled_default))
            cls_vars[cat.key] = var
            text = cat.label
            if cat.note:
                text += f"  ({cat.note})"
            tk.Checkbutton(cat_frame, text=text, variable=var, anchor="w",
                           wraplength=500, justify="left"
                           ).pack(fill="x", padx=8, pady=1)

        # ---- 詳細設定 ----
        opt_frame = tk.LabelFrame(win, text="詳細設定")
        opt_frame.pack(fill="x", padx=10, pady=4)

        row1 = tk.Frame(opt_frame)
        row1.pack(fill="x", padx=6, pady=2)
        tk.Label(row1, text="検出しきい値 (低いほど拾いやすい):",
                 font=("", 9)).pack(side="left")
        thr_var = tk.DoubleVar(value=self.threshold)
        tk.Scale(row1, from_=0.05, to=0.9, resolution=0.05, variable=thr_var,
                 orient=tk.HORIZONTAL, length=160).pack(side="left", padx=6)

        row2 = tk.Frame(opt_frame)
        row2.pack(fill="x", padx=6, pady=2)
        tk.Label(row2, text="マスク拡張 (px):", font=("", 9)).pack(side="left")
        margin_var = tk.IntVar(value=self.margin_px)
        tk.Spinbox(row2, from_=0, to=50, width=5,
                   textvariable=margin_var).pack(side="left", padx=6)
        tk.Label(row2, text="検出輪郭の外側に余裕を持たせます",
                 font=("", 8), fg="#888").pack(side="left")

        # ---- 動画 ----
        process_all_var = tk.BooleanVar(value=False)
        if self.is_video_mode:
            vid_frame = tk.LabelFrame(win, text="処理対象 (動画)")
            vid_frame.pack(fill="x", padx=10, pady=4)
            tk.Radiobutton(vid_frame, text="現在のフレームのみ",
                           variable=process_all_var, value=False).pack(anchor="w", padx=6)
            tk.Radiobutton(vid_frame,
                           text="動画全体 (一度検出した対象を全フレーム追跡)",
                           variable=process_all_var, value=True, wraplength=500,
                           justify="left").pack(anchor="w", padx=6)

        # ---- フォルダ ----
        overwrite_var = tk.BooleanVar(value=False)
        if self.folder_mode:
            tk.Checkbutton(win, text="既存のマスクも上書きする",
                           variable=overwrite_var).pack(pady=(4, 0))

        tk.Label(win,
                 text=("検出: deepghs/anime_censor_detection (イラスト向けYOLOv8) / "
                       "輪郭化・動画追跡: SAM2.1\n"
                       "初回はモデルのダウンロードが発生します (~1GB)。"),
                 font=("", 8), fg="#888888", justify="left", wraplength=520
                 ).pack(pady=(8, 0), padx=10, anchor="w")

        btn_frm = tk.Frame(win)
        btn_frm.pack(pady=10)
        ok = {"v": False}

        def on_run():
            ok["v"] = True
            win.destroy()

        run_label = "適用する" if self.folder_mode else "検出開始"
        tk.Button(btn_frm, text=run_label, command=on_run,
                  bg="#3a7bd5", fg="white", relief="flat",
                  padx=12, pady=4).pack(side="left", padx=6)
        tk.Button(btn_frm, text="キャンセル", command=win.destroy,
                  relief="flat", padx=8, pady=4).pack(side="left", padx=6)

        win.wait_window()
        if not ok["v"]:
            return None

        for k, v in cls_vars.items():
            self.enabled_keys[k] = v.get()

        cfg = DetectConfig()
        cfg.categories = [c for c in self.all_categories
                          if cls_vars[c.key].get()]
        cfg.threshold = float(thr_var.get())
        cfg.margin_px = int(margin_var.get())
        cfg.process_all_frames = bool(process_all_var.get())
        cfg.overwrite = bool(overwrite_var.get())
        self.result = cfg
        return cfg


def show_detection_results(parent, detections: List[Detection]) -> Optional[List[Detection]]:
    """検出結果のレビュー。採用する検出のリストを返す (キャンセル時 None)."""
    dlg = tk.Toplevel(parent)
    dlg.title(f"検出結果 ({len(detections)}件)")
    dlg.geometry("560x420")
    safe_grab(dlg)

    tk.Label(dlg, text="チェックした項目をモザイク範囲に追加します",
             font=("", 9), fg="gray").pack(pady=(6, 2))

    frm = tk.Frame(dlg)
    frm.pack(fill="both", expand=True, padx=10, pady=4)
    canvas_sc = tk.Canvas(frm)
    scrollbar = ttk.Scrollbar(frm, orient="vertical", command=canvas_sc.yview)
    inner = tk.Frame(canvas_sc)
    inner.bind("<Configure>",
               lambda e: canvas_sc.configure(scrollregion=canvas_sc.bbox("all")))
    canvas_sc.create_window((0, 0), window=inner, anchor="nw")
    canvas_sc.configure(yscrollcommand=scrollbar.set)
    canvas_sc.pack(side="left", fill="both", expand=True)
    scrollbar.pack(side="right", fill="y")

    vars_: List[tk.BooleanVar] = []
    for d in detections:
        v = tk.BooleanVar(value=True)
        vars_.append(v)
        x1, y1, x2, y2 = d.bbox
        seg_tag = "[輪郭]" if d.mask is not None else "[矩形]"
        score_tag = f" {d.score:.2f}" if d.score < 1.0 else ""
        tk.Checkbutton(
            inner,
            text=f"{seg_tag} {d.label}{score_tag}  ({x1},{y1})-({x2},{y2})",
            variable=v, anchor="w",
        ).pack(fill="x", padx=6, pady=1)

    btn_frame = tk.Frame(dlg)
    btn_frame.pack(pady=8)
    ok = {"v": False}

    def on_apply():
        ok["v"] = True
        dlg.destroy()

    tk.Button(btn_frame, text="選択範囲に追加", command=on_apply,
              bg="#3a7bd5", fg="white", relief="flat",
              padx=10, pady=4).pack(side="left", padx=6)
    tk.Button(btn_frame, text="キャンセル", command=dlg.destroy,
              relief="flat", padx=10, pady=4).pack(side="left", padx=6)

    dlg.wait_window()
    if not ok["v"]:
        return None
    return [d for v, d in zip(vars_, detections) if v.get()]
