"""自動検出の設定ダイアログと検出結果レビューダイアログ."""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk
from typing import Dict, List, Optional

from ..core.categories import Category, make_custom_category
from ..detect.base import Detection
from ..detect.pipeline import BACKENDS


class DetectConfig:
    """ユーザーが選んだ検出設定 (ダイアログの戻り値)."""

    def __init__(self):
        self.categories: List[Category] = []
        self.backend: str = "la_sam2"
        self.threshold: float = 0.4
        self.generation_mode: str = "hybrid"
        self.margin_px: int = 4
        self.process_all_frames: bool = False
        self.overwrite: bool = False


class DetectConfigDialog:
    """検出設定ダイアログ。状態 (チェック・追加クラス等) は呼び出し側が保持する."""

    def __init__(self, parent,
                 all_categories: List[Category],
                 enabled_keys: Dict[str, bool],
                 backend: str = "la_sam2",
                 threshold: float = 0.4,
                 generation_mode: str = "hybrid",
                 margin_px: int = 4,
                 is_video_mode: bool = False,
                 folder_mode: bool = False,
                 folder_image_count: int = 0):
        self.parent = parent
        self.all_categories = list(all_categories)
        self.enabled_keys = enabled_keys
        self.backend = backend
        self.threshold = threshold
        self.generation_mode = generation_mode
        self.margin_px = margin_px
        self.is_video_mode = is_video_mode
        self.folder_mode = folder_mode
        self.folder_image_count = folder_image_count
        self.result: Optional[DetectConfig] = None

    def show(self) -> Optional[DetectConfig]:
        win = tk.Toplevel(self.parent)
        title = "自動モザイク (フォルダ一括)" if self.folder_mode else "自動検出"
        win.title(title)
        win.geometry("520x640")
        win.resizable(False, True)
        win.grab_set()

        if self.folder_mode:
            tk.Label(win,
                     text=f"フォルダ内の画像 {self.folder_image_count} 枚に自動モザイクを適用します",
                     font=("", 10), pady=6).pack()

        # ---- カテゴリ ----
        cat_frame = tk.LabelFrame(win, text="検出対象")
        cat_frame.pack(fill="x", padx=10, pady=(8, 4))
        cls_vars: Dict[str, tk.BooleanVar] = {}
        cls_inner = tk.Frame(cat_frame)
        cls_inner.pack(fill="x", padx=6, pady=4)

        def _rebuild_cls():
            for w in cls_inner.winfo_children():
                w.destroy()
            cls_vars.clear()
            for i, cat in enumerate(self.all_categories):
                var = tk.BooleanVar(
                    value=self.enabled_keys.get(cat.key, cat.enabled_default))
                cls_vars[cat.key] = var
                text = cat.label
                if cat.note:
                    text += f"  ({cat.note})"
                tk.Checkbutton(cls_inner, text=text, variable=var, anchor="w"
                               ).grid(row=i, column=0, sticky="w", padx=4)

        _rebuild_cls()

        add_frm = tk.Frame(cat_frame)
        add_frm.pack(fill="x", padx=6, pady=(0, 6))
        tk.Label(add_frm, text="追加クラス (英語):", font=("", 9)).pack(side="left")
        new_cls_var = tk.StringVar()
        tk.Entry(add_frm, textvariable=new_cls_var, width=22).pack(side="left", padx=4)

        def _on_add_cls():
            name = new_cls_var.get().strip()
            if not name:
                return
            cat = make_custom_category(name)
            if all(c.key != cat.key for c in self.all_categories):
                self.all_categories.append(cat)
                self.enabled_keys[cat.key] = True
            new_cls_var.set("")
            for k, v in cls_vars.items():
                self.enabled_keys[k] = v.get()
            _rebuild_cls()

        tk.Button(add_frm, text="追加", command=_on_add_cls,
                  relief="flat", padx=8).pack(side="left")

        # ---- 検出エンジン ----
        eng_frame = tk.LabelFrame(win, text="検出エンジン")
        eng_frame.pack(fill="x", padx=10, pady=4)
        backend_var = tk.StringVar(value=self.backend)
        for key, label in BACKENDS.items():
            tk.Radiobutton(eng_frame, text=label, variable=backend_var,
                           value=key, anchor="w").pack(fill="x", padx=6)

        # ---- 詳細設定 ----
        opt_frame = tk.LabelFrame(win, text="詳細設定")
        opt_frame.pack(fill="x", padx=10, pady=4)

        row1 = tk.Frame(opt_frame)
        row1.pack(fill="x", padx=6, pady=2)
        tk.Label(row1, text="検出しきい値 (低いほど拾いやすい):",
                 font=("", 9)).pack(side="left")
        thr_var = tk.DoubleVar(value=self.threshold)
        tk.Scale(row1, from_=0.1, to=0.9, resolution=0.05, variable=thr_var,
                 orient=tk.HORIZONTAL, length=160).pack(side="left", padx=6)

        row2 = tk.Frame(opt_frame)
        row2.pack(fill="x", padx=6, pady=2)
        tk.Label(row2, text="マスク拡張 (px):", font=("", 9)).pack(side="left")
        margin_var = tk.IntVar(value=self.margin_px)
        tk.Spinbox(row2, from_=0, to=50, width=5,
                   textvariable=margin_var).pack(side="left", padx=6)
        tk.Label(row2, text="検出輪郭の外側に余裕を持たせます",
                 font=("", 8), fg="#888").pack(side="left")

        row3 = tk.Frame(opt_frame)
        row3.pack(fill="x", padx=6, pady=2)
        tk.Label(row3, text="LocateAnything 検出強度:", font=("", 9)).pack(side="left")
        mode_var = tk.StringVar(value=self.generation_mode)
        for m in ("fast", "hybrid", "slow"):
            tk.Radiobutton(row3, text=m, variable=mode_var, value=m
                           ).pack(side="left", padx=4)

        # ---- 動画 ----
        process_all_var = tk.BooleanVar(value=False)
        if self.is_video_mode:
            vid_frame = tk.LabelFrame(win, text="処理対象 (動画)")
            vid_frame.pack(fill="x", padx=10, pady=4)
            tk.Radiobutton(vid_frame, text="現在のフレームのみ",
                           variable=process_all_var, value=False).pack(anchor="w", padx=6)
            tk.Radiobutton(vid_frame,
                           text="動画全体 (SAM3 トラッキング — 一度検出した対象を全フレーム追跡)",
                           variable=process_all_var, value=True).pack(anchor="w", padx=6)

        # ---- フォルダ ----
        overwrite_var = tk.BooleanVar(value=False)
        if self.folder_mode:
            tk.Checkbutton(win, text="既存のマスクも上書きする",
                           variable=overwrite_var).pack(pady=(4, 0))

        tk.Label(win,
                 text=("初回はモデルのダウンロードが発生します "
                       "(SAM3: ~3.4GB / LocateAnything-3B: ~8GB)。\n"
                       "SAM3 (facebook/sam3) は Hugging Face で利用許諾の同意が必要です。\n"
                       "LocateAnything-3B は非商用研究目的のみ利用可。"),
                 font=("", 8), fg="#888888", justify="left", wraplength=490
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
                          if cls_vars.get(c.key) and cls_vars[c.key].get()]
        cfg.backend = backend_var.get()
        cfg.threshold = float(thr_var.get())
        cfg.generation_mode = mode_var.get()
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
    dlg.grab_set()

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
