"""メインエディタウィンドウ.

旧版 (mosaic.py) の編集機能を引き継ぎつつ、自動検出を
SAM3 / LocateAnything-3B の新パイプラインに置き換えたもの。
"""
from __future__ import annotations

import os
import shutil
import subprocess
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Dict, List, Optional

import cv2
import numpy as np
from PIL import Image, ImageTk

from ..core import masking
from ..core.categories import DEFAULT_CATEGORIES, Category
from ..core.session import (ALL_SUPPORTED_EXT, SUPPORTED_EXT,
                            SUPPORTED_VIDEO_EXT, Session)
from ..detect.pipeline import DetectionPipeline
from .detect_dialog import DetectConfig, DetectConfigDialog, show_detection_results
from .progress import ensure_deps, show_progress_window


class MosaicEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Mosaic Editor")

        self.session = Session()
        self.image: Optional[Image.Image] = None
        self.original_image: Optional[Image.Image] = None
        self.tk_image: Optional[ImageTk.PhotoImage] = None

        # ---- 動画状態 ----
        self.is_video: bool = False
        self.video_cap: Optional[cv2.VideoCapture] = None
        self.video_total_frames: int = 0
        self.video_fps: float = 30.0
        self.video_frame_index: int = 0
        self.video_masks: Dict[int, np.ndarray] = {}

        # ---- 編集状態 ----
        self.mode = tk.StringVar(value="pen")
        self.threshold = tk.IntVar(value=40)
        self.pen_brush_size = tk.IntVar(value=40)
        self.wand_brush_size = tk.IntVar(value=40)
        self.mosaic_size = tk.IntVar(value=10)
        self.mosaic_size.trace_add("write", lambda *_: self._safe_update_view())
        self.auto_mosaic = tk.BooleanVar(value=True)
        self.show_mask = tk.BooleanVar(value=True)

        self.mosaic_mask: Optional[np.ndarray] = None
        self.undo_stack: List[np.ndarray] = []
        self.redo_stack: List[np.ndarray] = []

        self.zoom: float = 1.0
        self._canvas_xview: float = 0.0
        self._canvas_yview: float = 0.0

        self.cursor_tag = "cursor_preview"
        self.preview_rect_tag = "preview_rect"
        self.selection_tag = "selection_rect"
        self.selection_rect: Optional[tuple] = None
        self.rect_start_x = None
        self.rect_start_y = None

        self._wand_img_bgr_cache: Optional[np.ndarray] = None
        self._wand_last_ix: int = -9999
        self._wand_last_iy: int = -9999
        self._pen_last_ix: Optional[int] = None
        self._pen_last_iy: Optional[int] = None

        # ---- 検出状態 ----
        self.pipeline = DetectionPipeline()
        self._detect_categories: List[Category] = list(DEFAULT_CATEGORIES)
        self._detect_enabled: Dict[str, bool] = {
            c.key: c.enabled_default for c in DEFAULT_CATEGORIES}
        self._detect_backend: str = "sam3"
        self._detect_threshold: float = 0.4
        self._detect_gen_mode: str = "hybrid"
        self._detect_margin: int = 4
        self._detect_cancel: bool = False

        self._skip_btn: Optional[tk.Button] = None
        self.frame_label: Optional[tk.Label] = None

        self.canvas = tk.Canvas(self.root, cursor="crosshair", bg="gray")
        self.build_menu()
        self.build_ui()
        self._setup_drag_drop()

    # ================= D&D =================

    def _setup_drag_drop(self):
        try:
            self.canvas.drop_target_register("DND_Files")  # type: ignore
            self.canvas.dnd_bind("<<Drop>>", self._on_dnd_drop)  # type: ignore
            self.root.drop_target_register("DND_Files")  # type: ignore
            self.root.dnd_bind("<<Drop>>", self._on_dnd_drop)  # type: ignore
        except Exception:
            pass

    def _on_dnd_drop(self, event):
        try:
            import re
            raw = event.data.strip()
            bracketed = re.findall(r'\{([^}]+)\}', raw)
            paths = bracketed if bracketed else raw.split()

            self.save_current(show_dialog=False)
            if not self.session.open_paths(paths):
                messagebox.showwarning("D&D", "対応ファイルが見つかりませんでした")
                return
            self.load_current_file()
            self.root.after(200, self._offer_folder_auto_detect)
        except Exception as e:
            messagebox.showerror("D&Dエラー", str(e))

    # ================= 座標変換 =================

    def canvas_to_image(self, event_x, event_y):
        cx = self.canvas.canvasx(event_x)
        cy = self.canvas.canvasy(event_y)
        return int(cx / self.zoom), int(cy / self.zoom)

    # ================= Undo / Redo =================

    def push_history(self):
        if self.mosaic_mask is not None:
            self.undo_stack.append(self.mosaic_mask.copy())
            if len(self.undo_stack) > 20:
                self.undo_stack.pop(0)
            self.redo_stack.clear()

    def undo(self, event=None):
        if not self.undo_stack:
            return
        if self.mosaic_mask is not None:
            self.redo_stack.append(self.mosaic_mask.copy())
        self.mosaic_mask = self.undo_stack.pop()
        self.update_view()

    def redo(self, event=None):
        if not self.redo_stack:
            return
        if self.mosaic_mask is not None:
            self.undo_stack.append(self.mosaic_mask.copy())
        self.mosaic_mask = self.redo_stack.pop()
        self.update_view()

    # ================= UI =================

    def build_menu(self):
        menubar = tk.Menu(self.root)
        filemenu = tk.Menu(menubar, tearoff=0)
        filemenu.add_command(label="画像を開く", command=self.open_image)
        filemenu.add_command(label="動画を開く", command=self.open_video)
        filemenu.add_command(label="フォルダを開く", command=self.open_folder)
        filemenu.add_command(label="保存", command=self.save_current)
        menubar.add_cascade(label="ファイル", menu=filemenu)

        detectmenu = tk.Menu(menubar, tearoff=0)
        detectmenu.add_command(label="自動検出 (SAM3 / LocateAnything)",
                               command=self.auto_detect_open_dialog)
        menubar.add_cascade(label="自動検出", menu=detectmenu)
        self.root.config(menu=menubar)

    def build_ui(self):
        top = tk.Frame(self.root)
        top.pack(fill="x", padx=5, pady=5)

        tk.Button(top, text="←", command=self.prev_image).pack(side="left")
        tk.Button(top, text="→", command=self.next_image).pack(side="left")
        tk.Button(top, text="戻す", command=self.undo).pack(side="left", padx=5)
        tk.Button(top, text="やり直す", command=self.redo).pack(side="left")
        tk.Button(top, text="拡大", command=self.zoom_in).pack(side="left", padx=(10, 0))
        tk.Button(top, text="縮小", command=self.zoom_out).pack(side="left", padx=(2, 0))
        tk.Button(top, text="100%", command=self.zoom_custom).pack(side="left", padx=(2, 0))

        tk.Button(top, text="自動検出", command=self.auto_detect_open_dialog,
                  bg="#4a90d9", fg="white", relief="flat", padx=6
                  ).pack(side="left", padx=(10, 0))

        self._skip_btn = tk.Button(top, text="作成しない", command=self.toggle_skip_jpg,
                                   bg="#555555", fg="white", relief="flat", padx=6)
        self._skip_btn.pack(side="left", padx=(10, 0))

        tk.Label(top, text="モザイク強度").pack(side="left", padx=(10, 2))
        self._mosaic_spinbox = tk.Spinbox(top, from_=4, to=100, width=5,
                                          textvariable=self.mosaic_size)
        self._mosaic_spinbox.pack(side="left")

        def _on_auto_toggle(*_):
            state = "disabled" if self.auto_mosaic.get() else "normal"
            self._mosaic_spinbox.config(state=state)
            self.update_view()

        self.auto_mosaic.trace_add("write", _on_auto_toggle)
        tk.Checkbutton(top, text="自動(規定)", variable=self.auto_mosaic
                       ).pack(side="left", padx=(2, 4))
        tk.Checkbutton(top, text="範囲表示", variable=self.show_mask,
                       command=self.update_view).pack(side="left", padx=4)
        _on_auto_toggle()

        self.frame_label = tk.Label(top, text="", font=("Consolas", 9), fg="#555")
        self.frame_label.pack(side="right", padx=8)

        settings_frame = tk.Frame(self.root)
        settings_frame.pack(fill="x", padx=5, pady=5)

        mode_frame = tk.LabelFrame(settings_frame, text="モード")
        mode_frame.pack(side="left", padx=5)
        for text, val in [("ペン", "pen"), ("選択", "rect"),
                          ("魔法の杖", "wand"), ("消しゴム", "eraser")]:
            tk.Radiobutton(mode_frame, text=text,
                           variable=self.mode, value=val).pack(side="left")
        tk.Button(mode_frame, text="解除", command=self.clear_selection
                  ).pack(side="left", padx=(5, 5))

        sliders_frame = tk.Frame(settings_frame)
        sliders_frame.pack(side="left", fill="x", expand=True, padx=5)
        tk.Label(sliders_frame, text="許容値 (魔法の杖)").grid(row=0, column=0, sticky="e")
        tk.Scale(sliders_frame, from_=0, to=255, variable=self.threshold,
                 orient=tk.HORIZONTAL, length=100).grid(row=0, column=1, sticky="w")
        tk.Label(sliders_frame, text="ペンサイズ").grid(row=0, column=2, sticky="e")
        tk.Scale(sliders_frame, from_=1, to=200, variable=self.pen_brush_size,
                 orient=tk.HORIZONTAL, length=100).grid(row=0, column=3, sticky="w")
        tk.Label(sliders_frame, text="魔法の杖サイズ").grid(row=0, column=4, sticky="e")
        tk.Scale(sliders_frame, from_=1, to=200, variable=self.wand_brush_size,
                 orient=tk.HORIZONTAL, length=100).grid(row=0, column=5, sticky="w")

        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<ButtonPress-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Motion>", self.on_mouse_move)
        self.canvas.bind("<MouseWheel>", self.on_mousewheel)
        self.canvas.bind("<Button-4>", lambda e: self._navigate_image_or_frame(e, -1))
        self.canvas.bind("<Button-5>", lambda e: self._navigate_image_or_frame(e, 1))
        self.canvas.bind("<Control-Button-4>", lambda e: self._zoom_at(e, 1.1))
        self.canvas.bind("<Control-Button-5>", lambda e: self._zoom_at(e, 1 / 1.1))
        self.canvas.bind("<ButtonPress-2>", self.start_pan)
        self.canvas.bind("<B2-Motion>", self.do_pan)

        self.root.bind("<Control-z>", self.undo)
        self.root.bind("<Control-y>", self.redo)
        self.root.bind("<Left>", lambda e: self.prev_image())
        self.root.bind("<Right>", lambda e: self.next_image())
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def clear_selection(self):
        self.selection_rect = None
        self.canvas.delete(self.selection_tag)

    def on_closing(self):
        if self.is_video and self.video_cap is not None:
            self.video_cap.release()
            self.video_cap = None
        self.save_current(show_dialog=False)
        self.root.destroy()

    def _safe_update_view(self):
        try:
            if self.mosaic_size.get() < 2:
                return
            self.update_view()
        except tk.TclError:
            pass

    # ================= Zoom / Pan =================

    def _zoom_to_fit(self):
        if self.original_image is None:
            return
        self.root.update_idletasks()
        cw = self.canvas.winfo_width()
        ch = self.canvas.winfo_height()
        if cw <= 1 or ch <= 1:
            self.zoom = 1.0
            return
        iw, ih = self.original_image.size
        self.zoom = min(cw / iw, ch / ih)

    def on_mousewheel(self, event):
        ctrl = (event.state & 0x4) != 0
        if ctrl:
            self._zoom_at(event, 1.1 if event.delta > 0 else 1 / 1.1)
        else:
            self._navigate_image_or_frame(event, -1 if event.delta > 0 else 1)

    def _navigate_image_or_frame(self, event, delta: int):
        if self.is_video:
            self._navigate_frame(delta)
        elif delta < 0:
            self.prev_image()
        else:
            self.next_image()

    def _zoom_at(self, event, factor):
        if self.image is None:
            return
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        old_zoom = self.zoom
        self.zoom = max(0.05, min(20.0, self.zoom * factor))
        scale = self.zoom / old_zoom
        new_cx, new_cy = cx * scale, cy * scale

        self._display_image_preserving_pos()
        img = self.image
        if img:
            total_w = img.width * self.zoom
            total_h = img.height * self.zoom
            sx = max(0.0, min(1.0, (new_cx - event.x) / total_w if total_w > 0 else 0))
            sy = max(0.0, min(1.0, (new_cy - event.y) / total_h if total_h > 0 else 0))
            self.canvas.xview_moveto(sx)
            self.canvas.yview_moveto(sy)
        self.update_cursor_preview(event)

    def zoom_in(self):
        if self.image is not None:
            self.zoom *= 1.25
            self._display_image_preserving_pos()

    def zoom_out(self):
        if self.image is not None:
            self.zoom /= 1.25
            self._display_image_preserving_pos()

    def zoom_custom(self):
        if self.image is None:
            return
        dlg = tk.Toplevel(self.root)
        dlg.title("ズーム率を指定")
        dlg.geometry("260x100")
        dlg.resizable(False, False)
        dlg.grab_set()
        tk.Label(dlg, text="ズーム率 (%)を入力 (10〜2000)", font=("", 9)).pack(pady=(10, 4))
        val = tk.IntVar(value=int(self.zoom * 100))
        sb = tk.Spinbox(dlg, from_=10, to=2000, width=8, textvariable=val, font=("", 10))
        sb.pack()
        sb.focus_set()
        sb.selection_range(0, "end")

        def apply(event=None):
            try:
                self.zoom = max(0.1, min(20.0, int(val.get()) / 100))
                self._display_image_preserving_pos()
            except (ValueError, tk.TclError):
                pass
            dlg.destroy()

        sb.bind("<Return>", apply)
        tk.Button(dlg, text="OK", command=apply, bg="#3a7bd5", fg="white",
                  relief="flat", padx=12, pady=3).pack(pady=6)

    def start_pan(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def do_pan(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)
        self.update_cursor_preview(event)

    # ================= マウス / プレビュー =================

    def on_mouse_move(self, event):
        self.update_cursor_preview(event)

    def update_cursor_preview(self, event):
        self.canvas.delete(self.cursor_tag)
        if self.image is None or not event:
            return
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        mode = self.mode.get()
        r = (self.wand_brush_size.get() if mode == "wand"
             else self.pen_brush_size.get()) * self.zoom
        color = {"eraser": "white", "wand": "cyan"}.get(mode, "green")
        if mode != "rect":
            self.canvas.create_oval(cx - r, cy - r, cx + r, cy + r,
                                    outline=color, width=2, tags=self.cursor_tag)

    def on_click(self, event):
        if self.image is None:
            return
        ix, iy = self.canvas_to_image(event.x, event.y)
        if self.mode.get() == "wand":
            self.push_history()
            img_np = np.array(self.original_image)
            self._wand_img_bgr_cache = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
            self._wand_last_ix, self._wand_last_iy = ix, iy
            self.apply_wand_flood(ix, iy)
        elif self.mode.get() == "rect":
            self.rect_start_x, self.rect_start_y = ix, iy
        else:
            self.push_history()
            self._pen_last_ix, self._pen_last_iy = ix, iy
            self.apply_brush(ix, iy)

    def on_drag(self, event):
        if self.image is None:
            return
        self.update_cursor_preview(event)
        ix, iy = self.canvas_to_image(event.x, event.y)

        if self.mode.get() in ("pen", "eraser"):
            self.apply_brush(ix, iy, self._pen_last_ix, self._pen_last_iy)
            self._pen_last_ix, self._pen_last_iy = ix, iy
        elif self.mode.get() == "wand":
            min_dist = max(3, self.wand_brush_size.get() // 3)
            dx, dy = ix - self._wand_last_ix, iy - self._wand_last_iy
            if dx * dx + dy * dy >= min_dist * min_dist:
                self._wand_last_ix, self._wand_last_iy = ix, iy
                self.apply_wand_flood(ix, iy)
        elif self.mode.get() == "rect":
            if self.rect_start_x is not None:
                self.canvas.delete(self.preview_rect_tag)
                self.canvas.create_rectangle(
                    self.rect_start_x * self.zoom, self.rect_start_y * self.zoom,
                    self.canvas.canvasx(event.x), self.canvas.canvasy(event.y),
                    outline="red", width=2, dash=(4, 4), tags=self.preview_rect_tag)

    def on_release(self, event):
        if self.mode.get() == "wand":
            self._wand_img_bgr_cache = None
        if self.mode.get() == "rect" and self.rect_start_x is not None:
            ix, iy = self.canvas_to_image(event.x, event.y)
            x1, x2 = sorted((self.rect_start_x, ix))
            y1, y2 = sorted((self.rect_start_y, iy))
            self.canvas.delete(self.preview_rect_tag)
            if abs(x2 - x1) < 2 and abs(y2 - y1) < 2:
                self.clear_selection()
            elif self.mosaic_mask is not None:
                h, w = self.mosaic_mask.shape[:2]
                x1, x2 = max(0, min(w, x1)), max(0, min(w, x2))
                y1, y2 = max(0, min(h, y1)), max(0, min(h, y2))
                self.selection_rect = (x1, y1, x2, y2)
                self.canvas.delete(self.selection_tag)
                self.canvas.create_rectangle(
                    x1 * self.zoom, y1 * self.zoom, x2 * self.zoom, y2 * self.zoom,
                    outline="red", width=2, dash=(4, 4), tags=self.selection_tag)
            self.rect_start_x = None
            self.rect_start_y = None

    # ================= 魔法の杖 =================

    def apply_wand_flood(self, x, y):
        if self.original_image is None or self.mosaic_mask is None:
            return
        if self._wand_img_bgr_cache is not None:
            img_bgr = self._wand_img_bgr_cache
        else:
            img_bgr = cv2.cvtColor(np.array(self.original_image), cv2.COLOR_RGB2BGR)
        h, w = img_bgr.shape[:2]
        if not (0 <= x < w and 0 <= y < h):
            return
        tol = self.threshold.get()

        mask = np.zeros((h + 2, w + 2), np.uint8)
        flags = 8 | (255 << 8) | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE
        cv2.floodFill(img_bgr, mask, (x, y), (255, 255, 255),
                      (tol, tol, tol), (tol, tol, tol), flags)
        flood_mask = mask[1:-1, 1:-1]

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        flood_bool = cv2.morphologyEx(flood_mask, cv2.MORPH_CLOSE, kernel).astype(bool)

        r = self.wand_brush_size.get()
        brush_limit = np.zeros((h, w), dtype=np.uint8)
        cv2.circle(brush_limit, (x, y), r, 255, -1)
        flood_bool &= brush_limit > 0

        if self.selection_rect is not None:
            sx1, sy1, sx2, sy2 = self.selection_rect
            sel = np.zeros_like(self.mosaic_mask, dtype=bool)
            sel[sy1:sy2, sx1:sx2] = True
            flood_bool &= sel

        self.mosaic_mask[flood_bool] = 255
        self._update_view_preserve_pos()

    # ================= ペン / 消しゴム =================

    def apply_brush(self, x, y, prev_x=None, prev_y=None):
        if self.mosaic_mask is None:
            return
        r = self.pen_brush_size.get()
        color = 0 if self.mode.get() == "eraser" else 255

        if self.selection_rect is not None:
            temp = self.mosaic_mask.copy()
            self._draw_brush_segment(temp, prev_x, prev_y, x, y, r, color)
            sx1, sy1, sx2, sy2 = self.selection_rect
            self.mosaic_mask[sy1:sy2, sx1:sx2] = temp[sy1:sy2, sx1:sx2]
        else:
            self._draw_brush_segment(self.mosaic_mask, prev_x, prev_y, x, y, r, color)
        self.update_view()

    @staticmethod
    def _draw_brush_segment(mask, x0, y0, x1, y1, r, color):
        if x0 is not None and y0 is not None and (x0 != x1 or y0 != y1):
            cv2.line(mask, (x0, y0), (x1, y1), (color,), r * 2)
            cv2.circle(mask, (x0, y0), r, (color,), -1)
        cv2.circle(mask, (x1, y1), r, (color,), -1)

    # ================= 表示合成 =================

    def get_block_size(self) -> int:
        if self.auto_mosaic.get():
            if self.original_image is None:
                return 4
            w, h = self.original_image.size
            return masking.auto_block_size(w, h)
        return max(4, self.mosaic_size.get())

    def generate_mosaic_image(self) -> Optional[np.ndarray]:
        if self.original_image is None or self.mosaic_mask is None:
            return None
        return masking.apply_mosaic(np.array(self.original_image),
                                    self.mosaic_mask, self.get_block_size())

    def _save_canvas_scroll(self):
        try:
            self._canvas_xview = self.canvas.xview()[0]
            self._canvas_yview = self.canvas.yview()[0]
        except Exception:
            self._canvas_xview = self._canvas_yview = 0.0

    def _restore_canvas_scroll(self):
        try:
            self.canvas.xview_moveto(self._canvas_xview)
            self.canvas.yview_moveto(self._canvas_yview)
        except Exception:
            pass

    def _update_view_preserve_pos(self):
        self._save_canvas_scroll()
        self.update_view()
        self._restore_canvas_scroll()

    def _display_image_preserving_pos(self):
        self._save_canvas_scroll()
        self.display_image()
        self._restore_canvas_scroll()

    def update_view(self):
        final_img = self.generate_mosaic_image()
        if final_img is None:
            return
        if self.show_mask.get() and self.mosaic_mask is not None:
            final_img = masking.overlay_mask(final_img, self.mosaic_mask)
        self.image = Image.fromarray(final_img)
        self.display_image()

    def display_image(self):
        if self.image is None:
            return
        new_w = int(self.image.width * self.zoom)
        new_h = int(self.image.height * self.zoom)
        resized = self.image.resize((max(1, new_w), max(1, new_h)), Image.NEAREST)
        self.tk_image = ImageTk.PhotoImage(resized)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        self.canvas.config(scrollregion=(0, 0, new_w, new_h))

        if self.selection_rect is not None:
            x1, y1, x2, y2 = self.selection_rect
            self.canvas.create_rectangle(
                x1 * self.zoom, y1 * self.zoom, x2 * self.zoom, y2 * self.zoom,
                outline="red", width=2, dash=(4, 4), tags=self.selection_tag)

        if self.is_skip_jpg():
            margin = 20
            self.canvas.create_line(margin, margin, new_w - margin, new_h - margin,
                                    fill="#ff0000", width=8, tags="skip_x")
            self.canvas.create_line(new_w - margin, margin, margin, new_h - margin,
                                    fill="#ff0000", width=8, tags="skip_x")
            self.canvas.create_text(new_w // 2, new_h // 2, text="作成しない",
                                    fill="#ff0000",
                                    font=("", max(14, new_h // 12), "bold"),
                                    tags="skip_x")

    # ================= スキップ =================

    def is_skip_jpg(self) -> bool:
        path = self.session.current
        return self.session.is_skipped(path) if path else False

    def toggle_skip_jpg(self):
        path = self.session.current
        if path is None:
            return
        skipped = not self.session.is_skipped(path)
        self.session.set_skipped(path, skipped)
        if skipped:
            jpg = self.session.output_jpg_path(path)
            if jpg and os.path.exists(jpg):
                try:
                    os.remove(jpg)
                except Exception:
                    pass
        self._update_skip_btn()
        self.update_view()

    def _update_skip_btn(self):
        if self._skip_btn is None:
            return
        if self.is_skip_jpg():
            self._skip_btn.config(bg="#cc2222", relief="sunken", text="✕ 作成しない")
        else:
            self._skip_btn.config(bg="#555555", relief="flat", text="作成しない")

    # ================= Load / Save =================

    def open_image(self):
        self.save_current(show_dialog=False)
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp")])
        if not path:
            return
        self.session.open_paths([path])
        self.load_current_file()

    def open_video(self):
        self.save_current(show_dialog=False)
        path = filedialog.askopenfilename(
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv *.webm"),
                       ("All files", "*.*")])
        if not path:
            return
        self.session.open_paths([path])
        self.load_current_file()

    def open_folder(self):
        self.save_current(show_dialog=False)
        folder = filedialog.askdirectory()
        if not folder:
            return
        if not self.session.open_paths([folder]):
            messagebox.showwarning("フォルダ", "対応ファイルが見つかりませんでした")
            return
        self.load_current_file()
        self.root.after(200, self._offer_folder_auto_detect)

    def load_current_file(self):
        path = self.session.current
        if path is None:
            return
        if path.lower().endswith(SUPPORTED_VIDEO_EXT):
            self.load_video(path)
        else:
            self.load_current_image(path)

    def load_current_image(self, path: str):
        if self.video_cap is not None:
            self.video_cap.release()
            self.video_cap = None
        self.is_video = False
        self.video_masks = {}
        if self.frame_label:
            self.frame_label.config(text="")

        img = Image.open(path).convert("RGB")
        self.original_image = img.copy()
        w, h = img.size
        loaded = self.session.load_mask(path, w, h)
        self.mosaic_mask = loaded if loaded is not None else np.zeros((h, w), dtype=np.uint8)

        self.undo_stack.clear()
        self.redo_stack.clear()
        self.clear_selection()
        self._canvas_xview = self._canvas_yview = 0.0
        self.session.load_skip_state(path)
        self._update_skip_btn()
        self._zoom_to_fit()
        self.update_view()
        self.root.title(
            f"Mosaic Editor - {os.path.basename(path)} "
            f"({self.session.index + 1}/{len(self.session.files)})")

    def load_video(self, path: str):
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("エラー", f"動画を開けませんでした:\n{path}")
            return
        if self.video_cap is not None:
            self.video_cap.release()

        self.is_video = True
        self.video_cap = cap
        self.video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.video_frame_index = 0
        self.video_masks = {}
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.clear_selection()
        self._canvas_xview = self._canvas_yview = 0.0
        self._update_skip_btn()
        self.root.title(
            f"Mosaic Editor [VIDEO] - {os.path.basename(path)} "
            f"({self.session.index + 1}/{len(self.session.files)})")
        self.load_frame_at(0, fit_zoom=True)

    def load_frame_at(self, index: int, fit_zoom: bool = False):
        if self.video_cap is None:
            return
        index = max(0, min(self.video_total_frames - 1, index))
        self.video_frame_index = index
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame_bgr = self.video_cap.read()
        if not ret:
            return
        self.original_image = Image.fromarray(
            cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        h, w = frame_bgr.shape[:2]
        if fit_zoom:
            self._zoom_to_fit()
        self.mosaic_mask = self.video_masks.get(
            index, np.zeros((h, w), dtype=np.uint8)).copy()
        self.update_view()
        if self.frame_label:
            self.frame_label.config(
                text=f"フレーム: {index + 1} / {self.video_total_frames}  "
                     f"({index / self.video_fps:.2f}s)")

    def _stash_video_frame_mask(self):
        if self.mosaic_mask is not None and np.any(self.mosaic_mask):
            self.video_masks[self.video_frame_index] = self.mosaic_mask.copy()
        else:
            self.video_masks.pop(self.video_frame_index, None)

    def _navigate_frame(self, delta: int):
        if not self.is_video:
            return
        self._stash_video_frame_mask()
        self.load_frame_at(self.video_frame_index + delta)

    def next_image(self):
        if self.is_video:
            self._stash_video_frame_mask()
        else:
            self.save_current(show_dialog=False)
        if self.session.index < len(self.session.files) - 1:
            self.session.index += 1
            self.load_current_file()

    def prev_image(self):
        if self.is_video:
            self._stash_video_frame_mask()
        else:
            self.save_current(show_dialog=False)
        if self.session.index > 0:
            self.session.index -= 1
            self.load_current_file()

    def save_current(self, show_dialog=True):
        path = self.session.current
        if path is None or self.session.output_folder is None:
            return

        if self.is_video:
            self._stash_video_frame_mask()
            if show_dialog:
                self._save_video()
            return

        save_path = self.session.output_jpg_path(path)
        if save_path is None:
            return

        if self.session.is_skipped(path):
            for p in (save_path, self.session.mask_path(path)):
                if p and os.path.exists(p):
                    try:
                        os.remove(p)
                    except Exception:
                        pass
            return

        clean_np = self.generate_mosaic_image()
        if clean_np is None:
            return
        Image.fromarray(clean_np).save(save_path, "JPEG", quality=95)
        self.session.save_mask(path, self.mosaic_mask)

        if show_dialog:
            messagebox.showinfo("保存", f"{save_path} に保存しました")

    # ================= 動画書き出し =================

    def _save_video(self):
        if self.video_cap is None or self.session.current is None:
            return
        src_path = self.session.current
        base = os.path.splitext(os.path.basename(src_path))[0]
        save_path = os.path.join(self.session.output_folder, base + ".mp4")
        total = self.video_total_frames
        fps = self.video_fps

        win, status, bar = show_progress_window(
            self.root, "動画出力中...", f"保存先: {save_path}",
            with_progress_bar=True, maximum=total)

        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, first = self.video_cap.read()
        if not ret:
            win.destroy()
            messagebox.showerror("エラー", "動画のフレームを読み出せません")
            return
        h, w = first.shape[:2]
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))
        vid_masks = self.video_masks
        block = self.get_block_size()

        def worker():
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for fi in range(total):
                ret, frame_bgr = self.video_cap.read()
                if not ret:
                    break
                mask = vid_masks.get(fi)
                if mask is not None and np.any(mask):
                    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    rgb = masking.apply_mosaic(rgb, mask, block)
                    frame_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)
                if fi % 10 == 0:
                    self.root.after(0, lambda f=fi: (
                        bar.config(value=f),
                        status.config(text=f"{f + 1} / {total} フレーム")))
            writer.release()
            self.root.after(0, lambda: (win.destroy(),
                                        self._merge_audio(src_path, save_path)))

        threading.Thread(target=worker, daemon=True).start()

    def _merge_audio(self, orig_path: str, new_path: str):
        """ffmpeg で元動画の音声を出力動画に結合する (無ければ無音のまま)."""
        ffmpeg = shutil.which("ffmpeg")
        if ffmpeg is None:
            messagebox.showinfo(
                "保存完了",
                f"無音のまま保存しました (音声を残すには ffmpeg をインストールしてください):\n{new_path}")
            return

        win, status, _ = show_progress_window(
            self.root, "音声結合中...", "元動画の音声を結合しています...")

        def worker():
            import tempfile
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
            os.close(tmp_fd)
            cmd = [ffmpeg, "-y",
                   "-i", new_path, "-i", orig_path,
                   "-map", "0:v:0", "-map", "1:a:0?",
                   "-c:v", "libx264", "-crf", "18", "-preset", "medium",
                   "-c:a", "aac", "-shortest", tmp_path]
            try:
                proc = subprocess.run(cmd, capture_output=True, text=True)
                if proc.returncode == 0:
                    shutil.move(tmp_path, new_path)
                    msg = ("保存完了", f"音声付きで保存しました:\n{new_path}")
                else:
                    os.remove(tmp_path)
                    msg = ("保存完了",
                           f"音声結合に失敗したため無音で保存しました:\n{new_path}")
            except Exception as e:
                msg = ("保存完了", f"無音で保存しました ({e}):\n{new_path}")
            self.root.after(0, lambda: (win.destroy(),
                                        messagebox.showinfo(*msg)))

        threading.Thread(target=worker, daemon=True).start()

    # ================= 自動検出 =================

    def _open_config_dialog(self, **kwargs) -> Optional[DetectConfig]:
        dlg = DetectConfigDialog(
            self.root,
            all_categories=self._detect_categories,
            enabled_keys=self._detect_enabled,
            backend=self._detect_backend,
            threshold=self._detect_threshold,
            generation_mode=self._detect_gen_mode,
            margin_px=self._detect_margin,
            **kwargs)
        cfg = dlg.show()
        if cfg is not None:
            # ダイアログで増えた追加クラスと設定を保持
            self._detect_categories = dlg.all_categories
            self._detect_backend = cfg.backend
            self._detect_threshold = cfg.threshold
            self._detect_gen_mode = cfg.generation_mode
            self._detect_margin = cfg.margin_px
        return cfg

    def auto_detect_open_dialog(self):
        if self.original_image is None:
            messagebox.showwarning("警告", "画像または動画を開いてください")
            return
        cfg = self._open_config_dialog(is_video_mode=self.is_video)
        if cfg is None:
            return
        if not cfg.categories:
            messagebox.showwarning("自動検出", "検出対象を1つ以上選んでください")
            return

        def _go():
            if cfg.process_all_frames and self.is_video:
                self._run_video_tracking(cfg)
            else:
                self.push_history()
                self._run_detect_current(cfg)

        ensure_deps(self.root, _go)

    def _run_detect_current(self, cfg: DetectConfig):
        win, status, _ = show_progress_window(
            self.root, "自動検出中", "検出を開始しています...")
        current_img = self.original_image.copy()

        def progress(msg: str):
            self.root.after(0, lambda m=msg: status.config(text=m))

        def worker():
            try:
                detections = self.pipeline.detect(
                    current_img, cfg.categories,
                    backend=cfg.backend, threshold=cfg.threshold,
                    generation_mode=cfg.generation_mode, progress_cb=progress)
                self.root.after(0, lambda: self._review_detections(win, cfg, detections))
            except Exception as e:
                err = str(e)
                self.root.after(0, lambda: self._on_detect_error(win, err))

        threading.Thread(target=worker, daemon=True).start()

    def _review_detections(self, win, cfg: DetectConfig, detections):
        try:
            win.destroy()
        except Exception:
            pass
        if self.mosaic_mask is None:
            return
        if not detections:
            messagebox.showinfo(
                "自動検出",
                "検出結果がありませんでした。\n"
                "・しきい値を下げる\n"
                "・検出エンジンを「LocateAnything + SAM3」や「両方併用」に変える\n"
                "等を試してください。")
            return

        accepted = show_detection_results(self.root, detections)
        if accepted is None:
            return

        h, w = self.mosaic_mask.shape[:2]
        add_mask = DetectionPipeline.combine_masks(
            accepted, (w, h), margin_px=cfg.margin_px)
        self.mosaic_mask = masking.merge_masks(self.mosaic_mask, add_mask)
        self.show_mask.set(True)
        self.update_view()
        messagebox.showinfo(
            "適用完了",
            f"{len(accepted)} 箇所を選択範囲に追加しました。\n"
            "ペン・魔法の杖・消しゴムで微調整してから保存してください。")

    def _on_detect_error(self, win, err: str):
        try:
            win.destroy()
        except Exception:
            pass
        hint = ""
        if "gated" in err or "403" in err:
            hint = ("\n\nfacebook/sam3 へのアクセス許可が必要です:\n"
                    "1. https://huggingface.co/facebook/sam3 で利用規約に同意\n"
                    "2. `hf auth login` でログイン")
        messagebox.showerror("検出エラー", f"検出中にエラーが発生しました:\n{err}{hint}")

    # ---- 動画全体: SAM3 トラッキング ----

    def _run_video_tracking(self, cfg: DetectConfig):
        if self.session.current is None:
            return
        video_path = self.session.current
        win, status, _ = show_progress_window(
            self.root, "動画トラッキング中",
            "SAM3 Video で検出 + トラッキングしています...",
            cancelable=True,
            on_cancel=lambda: setattr(self, "_detect_cancel", True))
        self._detect_cancel = False

        def progress(msg: str):
            self.root.after(0, lambda m=msg: status.config(text=m))

        def worker():
            try:
                masks = self.pipeline.video_tracker.track_video(
                    video_path, cfg.categories,
                    progress_cb=progress,
                    cancel_check=lambda: self._detect_cancel)
                if cfg.margin_px > 0:
                    masks = {fi: masking.dilate_mask(m, cfg.margin_px)
                             for fi, m in masks.items()}
                self.root.after(0, lambda: self._finish_video_tracking(win, masks))
            except Exception as e:
                err = str(e)
                self.root.after(0, lambda: self._on_detect_error(win, err))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_video_tracking(self, win, masks: Dict[int, np.ndarray]):
        try:
            win.destroy()
        except Exception:
            pass
        for fi, m in masks.items():
            existing = self.video_masks.get(fi)
            self.video_masks[fi] = masking.merge_masks(existing, m)
        self.load_frame_at(self.video_frame_index)
        word = "キャンセルされるまでに" if self._detect_cancel else "全体で"
        messagebox.showinfo(
            "トラッキング完了",
            f"{word} {len(masks)} フレームにマスクを適用しました。\n"
            "ホイールでフレームを確認し、必要なら微調整してから保存してください。")

    # ---- フォルダ一括 ----

    def _offer_folder_auto_detect(self):
        img_files = [p for p in self.session.files
                     if p.lower().endswith(SUPPORTED_EXT)]
        if not img_files:
            return
        cfg = self._open_config_dialog(
            folder_mode=True, folder_image_count=len(img_files))
        if cfg is None:
            return
        if not cfg.categories:
            messagebox.showwarning("自動検出", "検出対象を1つ以上選んでください")
            return
        ensure_deps(self.root, lambda: self._run_folder_batch(img_files, cfg))

    def _run_folder_batch(self, img_files: List[str], cfg: DetectConfig):
        total = len(img_files)
        win, status, bar = show_progress_window(
            self.root, "フォルダ一括検出中", "準備中...",
            with_progress_bar=True, maximum=total,
            cancelable=True,
            on_cancel=lambda: setattr(self, "_detect_cancel", True))
        self._detect_cancel = False

        def set_bar(val: int):
            try:
                if win.winfo_exists():
                    bar.config(value=val)
            except Exception:
                pass

        def worker():
            applied = 0
            try:
                for fi, img_path in enumerate(img_files):
                    if self._detect_cancel:
                        break
                    mask_path = self.session.mask_path(img_path)
                    if not cfg.overwrite and mask_path and os.path.exists(mask_path):
                        self.root.after(0, lambda v=fi + 1: set_bar(v))
                        continue
                    self.root.after(0, lambda f=fi, p=img_path: status.config(
                        text=f"[{f + 1}/{total}] {os.path.basename(p)}"))
                    try:
                        img_pil = Image.open(img_path).convert("RGB")
                    except Exception:
                        self.root.after(0, lambda v=fi + 1: set_bar(v))
                        continue
                    detections = self.pipeline.detect(
                        img_pil, cfg.categories,
                        backend=cfg.backend, threshold=cfg.threshold,
                        generation_mode=cfg.generation_mode)
                    if detections:
                        w, h = img_pil.size
                        combined = DetectionPipeline.combine_masks(
                            detections, (w, h), margin_px=cfg.margin_px)
                        if np.any(combined):
                            np.savez_compressed(mask_path, mask=combined)
                            applied += 1
                    self.root.after(0, lambda v=fi + 1: set_bar(v))
                self.root.after(0, lambda: self._finish_folder_batch(win, applied, total))
            except Exception as e:
                err = str(e)
                self.root.after(0, lambda: self._on_detect_error(win, err))

        threading.Thread(target=worker, daemon=True).start()

    def _finish_folder_batch(self, win, applied: int, total: int):
        try:
            win.destroy()
        except Exception:
            pass
        self.load_current_file()
        if self._detect_cancel:
            messagebox.showinfo("キャンセル", f"{applied}/{total} 枚適用済みでキャンセルしました")
        else:
            messagebox.showinfo(
                "一括検出完了",
                f"全 {total} 枚中、{applied} 枚にモザイクを適用しました。\n"
                "← → キーで確認し、微調整してから保存してください。")
