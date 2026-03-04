import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import cv2
import os
import sys
import threading
from typing import Optional, List

SUPPORTED_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
SUPPORTED_VIDEO_EXT = (".mp4", ".avi", ".mov", ".mkv", ".webm")
ALL_SUPPORTED_EXT = SUPPORTED_EXT + SUPPORTED_VIDEO_EXT


class MosaicEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Mosaic Editor")

        self.image: Optional[Image.Image] = None
        self.original_image: Optional[Image.Image] = None
        self.tk_image: Optional[ImageTk.PhotoImage] = None

        self.image_list: List[str] = []
        self.current_index: int = 0
        self.current_path: Optional[str] = None
        self.output_folder: Optional[str] = None

        # ---- 動画状態 ----
        self.is_video: bool = False
        self.video_cap: Optional[cv2.VideoCapture] = None
        self.video_total_frames: int = 0
        self.video_fps: float = 30.0
        self.video_frame_index: int = 0
        self.video_masks: dict = {}  # {frame_idx: np.ndarray}
        self.frame_label: Optional[tk.Label] = None

        # Mode: pen, wand, eraser, rect
        self.mode = tk.StringVar(value="pen")
        self.threshold = tk.IntVar(value=40)
        self.brush_size = tk.IntVar(value=20)
        self.mosaic_size = tk.IntVar(value=10)
        self.mosaic_size.trace_add("write", self.on_mosaic_size_change)
        self.auto_mosaic = tk.BooleanVar(value=True)  # 規定準拠自動サイズ
        self.show_mask = tk.BooleanVar(value=True)
        self.use_edge_detection = tk.BooleanVar(value=False)

        self.undo_stack: List[np.ndarray] = []
        self.redo_stack: List[np.ndarray] = []

        self.zoom: float = 1.0
        # Canvas scroll offset (for pan position preservation)
        self._canvas_xview: float = 0.0
        self._canvas_yview: float = 0.0

        self.mosaic_mask: Optional[np.ndarray] = None

        self.cursor_tag = "cursor_preview"
        self.preview_rect_tag = "preview_rect"
        self.selection_tag = "selection_rect"
        self.selection_rect: Optional[tuple] = None
        self.rect_start_x = None
        self.rect_start_y = None

        self.canvas = tk.Canvas(self.root, cursor="crosshair", bg="gray")

        self.build_menu()
        self.build_ui()

        # Drag & Drop support (folder or file)
        self._setup_drag_drop()

    # ================= Drag & Drop =================

    def _setup_drag_drop(self):
        """tkinterdnd2を使ってD&Dを有効化する"""
        try:
            # キャンバスとルートウィンドウ両方をドロップターゲットに登録
            self.canvas.drop_target_register("DND_Files")  # type: ignore
            self.canvas.dnd_bind("<<Drop>>", self._on_dnd_drop)  # type: ignore
            self.root.drop_target_register("DND_Files")  # type: ignore
            self.root.dnd_bind("<<Drop>>", self._on_dnd_drop)  # type: ignore
        except Exception:
            pass

    def _on_dnd_drop(self, event):
        """tkinterdnd2のドロップイベントを処理する"""
        try:
            # event.data はスペース区切りのパス文字列（{}で囲まれる場合あり）
            raw = event.data.strip()
            # {} で囲まれた複数パスを分解
            import re
            # 例: {C:/foo/bar} {D:/baz} or C:/single
            bracketed = re.findall(r'\{([^}]+)\}', raw)
            if bracketed:
                paths = bracketed
            else:
                paths = raw.split()

            all_files = []
            for p in paths:
                p = p.strip()
                if os.path.isdir(p):
                    for root_dir, _, filenames in os.walk(p):
                        for f in filenames:
                            if f.lower().endswith(ALL_SUPPORTED_EXT):
                                all_files.append(os.path.join(root_dir, f))
                elif os.path.isfile(p) and p.lower().endswith(ALL_SUPPORTED_EXT):
                    all_files.append(p)

            if not all_files:
                messagebox.showwarning("D&D", "対応ファイルが見つかりませんでした")
                return

            all_files.sort()
            self.save_current(show_dialog=False)

            base = paths[0].strip()
            if os.path.isdir(base):
                out_folder = base.rstrip("/\\") + "_mc"
            else:
                out_folder = os.path.dirname(base) + "_mc"
            os.makedirs(out_folder, exist_ok=True)

            self.output_folder = out_folder
            self.image_list = all_files
            self.current_index = 0
            self.load_current_file()
            # フォルダD&D後に自動検出をオファー
            if len(all_files) > 1 or (len(all_files) == 1 and os.path.isdir(paths[0].strip())):
                self.root.after(200, self._offer_folder_auto_detect)
        except Exception as e:
            messagebox.showerror("D&Dエラー", str(e))

    # ================= 座標変換 =================

    def canvas_to_image(self, event_x, event_y):
        cx = self.canvas.canvasx(event_x)
        cy = self.canvas.canvasy(event_y)
        ix = int(cx / self.zoom)
        iy = int(cy / self.zoom)
        return ix, iy

    # ================= Utility =================

    def get_mask_path(self, img_path: str) -> str:
        """画像に対応するマスクNPZファイルのパスを返す"""
        if self.output_folder is None:
            return ""
        base = os.path.splitext(os.path.basename(img_path))[0]
        return os.path.join(self.output_folder, base + ".mask.npz")

    def get_block_size(self, img_w: int = 0, img_h: int = 0) -> int:
        """ブロックサイズを取得。自動モード時は視親規定に基づく計算。

        規定: 長辺が400px以上なら max(4, 長辺 // 100)、未満なら最小4px。
        """
        if self.auto_mosaic.get():
            # 画像サイズが渡されていない場合は現在画像から取得
            if img_w == 0 and img_h == 0 and self.original_image is not None:
                img_w, img_h = self.original_image.size
            long_side = max(img_w, img_h)
            if long_side >= 400:
                return max(4, long_side // 100)
            return 4  # 長辺400px未満は最小4px
        # 手動指定時は最小4pxを保証
        return max(4, self.mosaic_size.get())

    def push_history(self):
        mask = self.mosaic_mask
        if mask is not None:
            self.undo_stack.append(mask.copy())
            if len(self.undo_stack) > 20:
                self.undo_stack.pop(0)
            self.redo_stack.clear()

    def undo(self, event=None):
        if not self.undo_stack:
            return
        mask = self.mosaic_mask
        if mask is not None:
            self.redo_stack.append(mask.copy())
        self.mosaic_mask = self.undo_stack.pop()
        self.update_view()

    def redo(self, event=None):
        if not self.redo_stack:
            return
        mask = self.mosaic_mask
        if mask is not None:
            self.undo_stack.append(mask.copy())
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
        detectmenu.add_command(label="自動検出 (YOLO)", command=self.auto_detect_yolo)
        detectmenu.add_command(label="YOLOモデルを選択...", command=self.select_yolo_model)
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

        tk.Button(top, text="自動検出 (YOLO)", command=self.auto_detect_yolo,
                  bg="#4a90d9", fg="white", relief="flat", padx=6).pack(side="left", padx=(10, 0))

        # フレームナビラベル（動画時のみ使用）
        self.frame_label = tk.Label(top, text="", font=("Consolas", 9), fg="#555")
        self.frame_label.pack(side="right", padx=8)

        # Settings Area
        settings_frame = tk.Frame(self.root)
        settings_frame.pack(fill="x", padx=5, pady=5)

        # Mode Selection
        mode_frame = tk.LabelFrame(settings_frame, text="モード")
        mode_frame.pack(side="left", padx=5)

        for text, val in [("ペン", "pen"),
                          ("選択", "rect"),
                          ("魔法の杖", "wand"),
                          ("消しゴム", "eraser")]:
            tk.Radiobutton(mode_frame, text=text,
                           variable=self.mode, value=val).pack(side="left")

        tk.Button(mode_frame, text="解除", command=self.clear_selection).pack(side="left", padx=(5, 5))

        # Sliders
        sliders_frame = tk.Frame(settings_frame)
        sliders_frame.pack(side="left", fill="x", expand=True, padx=5)

        tk.Label(sliders_frame, text="許容値 (魔法の杖)").grid(row=0, column=0, sticky="e")
        tk.Spinbox(sliders_frame, from_=0, to=255, width=5,
                   textvariable=self.threshold).grid(row=0, column=1, sticky="w")

        tk.Label(sliders_frame, text="ブラシサイズ").grid(row=0, column=2, sticky="e")
        tk.Scale(sliders_frame, from_=1, to=200, variable=self.brush_size,
                 orient=tk.HORIZONTAL, length=100).grid(row=0, column=3, sticky="w")

        tk.Label(sliders_frame, text="モザイク強度").grid(row=0, column=4, sticky="e")
        self._mosaic_spinbox = tk.Spinbox(sliders_frame, from_=4, to=100, width=5,
                   textvariable=self.mosaic_size)
        self._mosaic_spinbox.grid(row=0, column=5, sticky="w")

        def _on_auto_toggle(*_):
            state = "disabled" if self.auto_mosaic.get() else "normal"
            self._mosaic_spinbox.config(state=state)
            self.update_view()

        self.auto_mosaic.trace_add("write", _on_auto_toggle)
        _auto_cb = tk.Checkbutton(sliders_frame, text="自動(規定)",
                                  variable=self.auto_mosaic,
                                  command=lambda: _on_auto_toggle())
        _auto_cb.grid(row=0, column=6, sticky="w", padx=(2, 4))
        # 初期状態を反映
        _on_auto_toggle()

        tk.Checkbutton(sliders_frame, text="範囲表示", variable=self.show_mask,
                       command=self.update_view).grid(row=0, column=7, sticky="w", padx=4)

        tk.Checkbutton(sliders_frame, text="境界線で止める", variable=self.use_edge_detection).grid(
            row=0, column=8, sticky="w", padx=4)

        if self.canvas is not None:
            self.canvas.pack(fill="both", expand=True)

            self.canvas.bind("<ButtonPress-1>", self.on_click)
            self.canvas.bind("<B1-Motion>", self.on_drag)
            self.canvas.bind("<ButtonRelease-1>", self.on_release)
            self.canvas.bind("<Motion>", self.on_mouse_move)

            # マウスホイールズーム (Windows: <MouseWheel>, Linux: <Button-4>/<Button-5>)
            self.canvas.bind("<MouseWheel>", self.on_mousewheel)
            self.canvas.bind("<Button-4>", lambda e: self._zoom_at(e, 1.1))
            self.canvas.bind("<Button-5>", lambda e: self._zoom_at(e, 1 / 1.1))

            self.canvas.bind("<ButtonPress-2>", self.start_pan)
            self.canvas.bind("<B2-Motion>", self.do_pan)

        self.root.bind("<Control-z>", self.undo)
        self.root.bind("<Control-y>", self.redo)
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def clear_selection(self):
        self.selection_rect = None
        if hasattr(self, 'canvas') and self.canvas:
            self.canvas.delete(self.selection_tag)

    def on_closing(self):
        if self.is_video and self.video_cap is not None:
            self.video_cap.release()
            self.video_cap = None
        self.save_current(show_dialog=False)
        self.root.destroy()


    def on_mosaic_size_change(self, *args):
        try:
            val = self.mosaic_size.get()
            if val < 2:
                return
            self.update_view()
        except tk.TclError:
            pass

    # ================= Zoom =================

    def on_mousewheel(self, event):
        """Windowsマウスホイール: 動画時はフレーム移動、画像時はズーム（Ctrl+ホイールは常にズーム）"""
        ctrl = (event.state & 0x4) != 0
        if self.is_video and not ctrl:
            # フレーム移動: ホイール下 = 次フレーム、ホイール上 = 前フレーム
            self._navigate_frame(1 if event.delta < 0 else -1)
        else:
            factor = 1.1 if event.delta > 0 else 1 / 1.1
            self._zoom_at(event, factor)

    def _zoom_at(self, event, factor):
        """マウスカーソル位置を中心にズーム"""
        if self.image is None:
            return

        # マウス位置のキャンバス座標（スクロール考慮）
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)

        old_zoom = self.zoom
        self.zoom *= factor
        self.zoom = max(0.05, min(20.0, self.zoom))

        # ズーム後にマウス位置が同じ画像座標を指すようにスクロール調整
        # cx_new = cx * (self.zoom / old_zoom)
        # スクロール位置の差分だけ移動
        scale = self.zoom / old_zoom
        new_cx = cx * scale
        new_cy = cy * scale

        self._display_image_preserving_pos()

        # キャンバスのスクロール位置を再設定
        img = self.image
        if img:
            canvas_w = self.canvas.winfo_width()
            canvas_h = self.canvas.winfo_height()
            total_w = img.width * self.zoom
            total_h = img.height * self.zoom

            # マウス直下の点をキャンバス内に維持
            scroll_x = (new_cx - event.x) / total_w if total_w > 0 else 0
            scroll_y = (new_cy - event.y) / total_h if total_h > 0 else 0
            scroll_x = max(0.0, min(1.0, scroll_x))
            scroll_y = max(0.0, min(1.0, scroll_y))
            self.canvas.xview_moveto(scroll_x)
            self.canvas.yview_moveto(scroll_y)

        self.update_cursor_preview(event)

    def zoom_in(self):
        if self.image is None:
            return
        self.zoom *= 1.25
        self._display_image_preserving_pos()

    def zoom_out(self):
        if self.image is None:
            return
        self.zoom /= 1.25
        self._display_image_preserving_pos()

    def zoom_custom(self):
        """ズーム率を数値入力ダイアログで設定"""
        if self.image is None:
            return
        dlg = tk.Toplevel(self.root)
        dlg.title("ズーム率を指定")
        dlg.geometry("260x100")
        dlg.resizable(False, False)
        dlg.grab_set()

        tk.Label(dlg, text="ズーム率 (%)を入力 (10〜2000)",
                 font=("", 9)).pack(pady=(10, 4))
        val = tk.IntVar(value=int(self.zoom * 100))
        sb = tk.Spinbox(dlg, from_=10, to=2000, width=8, textvariable=val, font=("", 10))
        sb.pack()
        sb.focus_set()
        sb.selection_range(0, "end")

        def apply(event=None):
            try:
                v = int(val.get())
                self.zoom = max(0.1, min(20.0, v / 100))
                self._display_image_preserving_pos()
            except (ValueError, tk.TclError):
                pass
            dlg.destroy()

        sb.bind("<Return>", apply)
        tk.Button(dlg, text="OK", command=apply,
                  bg="#3a7bd5", fg="white", relief="flat",
                  padx=12, pady=3).pack(pady=6)

    # ================= Pan =================

    def start_pan(self, event):
        if self.canvas:
            self.canvas.scan_mark(event.x, event.y)

    def do_pan(self, event):
        if self.canvas:
            self.canvas.scan_dragto(event.x, event.y, gain=1)
            self.update_cursor_preview(event)

    # ================= Mouse / Preview =================

    def on_mouse_move(self, event):
        self.update_cursor_preview(event)

    def update_cursor_preview(self, event):
        if self.canvas is None:
            return
        self.canvas.delete(self.cursor_tag)

        if self.image is None or not event:
            return

        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)

        r = self.brush_size.get() * self.zoom

        color = "green"
        if self.mode.get() == "eraser":
            color = "white"

        if self.mode.get() not in ("rect", "wand"):
            self.canvas.create_oval(
                cx - r, cy - r, cx + r, cy + r,
                outline=color, width=2, tags=self.cursor_tag
            )

    def on_click(self, event):
        if self.image is None:
            return

        ix, iy = self.canvas_to_image(event.x, event.y)

        if self.mode.get() == "wand":
            self.start_stroke()
            self.apply_wand_flood(ix, iy)
        elif self.mode.get() == "rect":
            self.rect_start_x, self.rect_start_y = ix, iy
        else:
            self.start_stroke()
            self.apply_brush(ix, iy)

    def on_drag(self, event):
        if self.image is None:
            return

        self.update_cursor_preview(event)
        ix, iy = self.canvas_to_image(event.x, event.y)

        if self.mode.get() in ("pen", "eraser"):
            self.apply_brush(ix, iy)
        elif self.mode.get() == "wand":
            self.apply_wand_flood(ix, iy)
        elif self.mode.get() == "rect":
            if self.rect_start_x is not None and self.rect_start_y is not None:
                self.canvas.delete(self.preview_rect_tag)
                start_cx = self.rect_start_x * self.zoom
                start_cy = self.rect_start_y * self.zoom
                cx = self.canvas.canvasx(event.x)
                cy = self.canvas.canvasy(event.y)
                self.canvas.create_rectangle(
                    start_cx, start_cy, cx, cy,
                    outline="red", width=2, dash=(4, 4), tags=self.preview_rect_tag
                )

    def on_release(self, event):
        if self.mode.get() == "rect":
            if self.rect_start_x is not None and self.rect_start_y is not None:
                ix, iy = self.canvas_to_image(event.x, event.y)
                x1, x2 = min(self.rect_start_x, ix), max(self.rect_start_x, ix)
                y1, y2 = min(self.rect_start_y, iy), max(self.rect_start_y, iy)

                self.canvas.delete(self.preview_rect_tag)

                if abs(x2 - x1) < 2 and abs(y2 - y1) < 2:
                    self.clear_selection()
                else:
                    h, w = self.mosaic_mask.shape[:2] if self.mosaic_mask is not None else (0, 0)
                    if w > 0 and h > 0:
                        x1 = max(0, min(w, x1))
                        y1 = max(0, min(h, y1))
                        x2 = max(0, min(w, x2))
                        y2 = max(0, min(h, y2))
                        self.selection_rect = (x1, y1, x2, y2)
                        self.canvas.delete(self.selection_tag)
                        self.canvas.create_rectangle(
                            x1 * self.zoom, y1 * self.zoom, x2 * self.zoom, y2 * self.zoom,
                            outline="red", width=2, dash=(4, 4), tags=self.selection_tag
                        )

                self.rect_start_x = None
                self.rect_start_y = None

    # ================= Wand (Flood Fill) =================

    def apply_wand_flood(self, x, y):
        if self.original_image is None or self.mosaic_mask is None:
            return

        img_np = np.array(self.original_image)
        h, w = img_np.shape[:2]
        if x < 0 or x >= w or y < 0 or y >= h:
            return

        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        tol = self.threshold.get()

        mask = np.zeros((h + 2, w + 2), np.uint8)

        connectivity = 8
        flags = connectivity | (255 << 8) | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE

        if self.use_edge_detection.get():
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            edges = cv2.Canny(gray, 100, 200)
            kernel_edge = np.ones((3, 3), np.uint8)
            edges_dilated = cv2.dilate(edges, kernel_edge, iterations=1)
            mask[1:-1, 1:-1][edges_dilated > 0] = 255

            if mask[y + 1, x + 1] == 0:
                cv2.floodFill(img_bgr, mask, (x, y), (255, 255, 255),
                              (tol, tol, tol), (tol, tol, tol), flags)
        else:
            cv2.floodFill(img_bgr, mask, (x, y), (255, 255, 255),
                          (tol, tol, tol), (tol, tol, tol), flags)

        flood_mask = mask[1:-1, 1:-1]

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        flood_mask_closed = cv2.morphologyEx(flood_mask, cv2.MORPH_CLOSE, kernel)
        flood_mask_bool = flood_mask_closed.astype(bool)

        m_mask = self.mosaic_mask
        if m_mask is not None:
            if hasattr(self, 'selection_rect') and self.selection_rect is not None:
                sx1, sy1, sx2, sy2 = self.selection_rect
                sel_mask = np.zeros_like(m_mask, dtype=bool)
                sel_mask[sy1:sy2, sx1:sx2] = True
                flood_mask_bool = flood_mask_bool & sel_mask

            m_mask[flood_mask_bool] = 255

        # バグ修正: update_viewはスクロール位置を保持して表示を更新する
        self._update_view_preserve_pos()

    # ================= Pen / Brush =================

    def apply_brush(self, x, y):
        m_mask = self.mosaic_mask
        if m_mask is None:
            return

        h, w = m_mask.shape[:2]
        r = self.brush_size.get()

        color = 255
        if self.mode.get() == "eraser":
            color = 0

        if hasattr(self, 'selection_rect') and self.selection_rect is not None:
            temp_mask = m_mask.copy()
            cv2.circle(temp_mask, (x, y), r, (color,), -1)
            sx1, sy1, sx2, sy2 = self.selection_rect
            m_mask[sy1:sy2, sx1:sx2] = temp_mask[sy1:sy2, sx1:sx2]
        else:
            cv2.circle(m_mask, (x, y), r, (color,), -1)

        self.update_view()

    def start_stroke(self):
        self.push_history()

    # ================= View Composition =================

    def generate_mosaic_image(self) -> Optional[np.ndarray]:
        orig_pil = self.original_image
        m_mask = self.mosaic_mask

        if orig_pil is None or m_mask is None:
            return None

        orig_np = np.array(orig_pil)
        h, w = orig_np.shape[:2]
        block = self.get_block_size()

        small_h, small_w = max(1, h // block), max(1, w // block)
        small = cv2.resize(orig_np, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        mosaic_full = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)

        final_img = orig_np.copy()
        mask_bool = m_mask > 0
        final_img[mask_bool] = mosaic_full[mask_bool]

        return final_img

    def _save_canvas_scroll(self):
        """現在のキャンバスのスクロール位置を保存"""
        try:
            self._canvas_xview = self.canvas.xview()[0]
            self._canvas_yview = self.canvas.yview()[0]
        except Exception:
            self._canvas_xview = 0.0
            self._canvas_yview = 0.0

    def _restore_canvas_scroll(self):
        """保存したスクロール位置を復元"""
        try:
            self.canvas.xview_moveto(self._canvas_xview)
            self.canvas.yview_moveto(self._canvas_yview)
        except Exception:
            pass

    def _update_view_preserve_pos(self):
        """スクロール位置を保持しながらビューを更新（魔法の杖用）"""
        self._save_canvas_scroll()
        self.update_view()
        self._restore_canvas_scroll()

    def _display_image_preserving_pos(self):
        """スクロール位置を保持しながら画像を表示"""
        self._save_canvas_scroll()
        self.display_image()
        self._restore_canvas_scroll()

    def update_view(self):
        final_img = self.generate_mosaic_image()
        if final_img is None:
            return

        if self.show_mask.get() and self.mosaic_mask is not None:
            m_mask = self.mosaic_mask
            mask_bool = m_mask > 0

            green_overlay = final_img.copy()
            green_overlay[mask_bool] = [0, 255, 0]

            blended = cv2.addWeighted(final_img, 0.7, green_overlay, 0.3, 0)
            final_img[mask_bool] = blended[mask_bool]

        self.image = Image.fromarray(final_img)
        self.display_image()

    # ================= Load / Save =================

    def open_image(self):
        self.save_current(show_dialog=False)
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp")]
        )
        if not path:
            return
        self.image_list = [path]
        self.current_index = 0
        self.current_path = path
        out_folder = os.path.dirname(path) + "_mc"
        os.makedirs(out_folder, exist_ok=True)
        self.output_folder = out_folder
        self.load_current_file()

    def open_video(self):
        self.save_current(show_dialog=False)
        path = filedialog.askopenfilename(
            filetypes=[("Videos", "*.mp4 *.avi *.mov *.mkv *.webm"),
                       ("All files", "*.*")]
        )
        if not path:
            return
        self.image_list = [path]
        self.current_index = 0
        self.current_path = path
        out_folder = os.path.dirname(path) + "_mc"
        os.makedirs(out_folder, exist_ok=True)
        self.output_folder = out_folder
        self.load_current_file()

    def open_folder(self):
        self.save_current(show_dialog=False)
        folder = filedialog.askdirectory()
        if not folder:
            return
        out_folder = folder + "_mc"
        os.makedirs(out_folder, exist_ok=True)
        self.output_folder = out_folder
        files = []
        for root_dir, _, filenames in os.walk(folder):
            for f in filenames:
                if f.lower().endswith(ALL_SUPPORTED_EXT):
                    files.append(os.path.join(root_dir, f))
        files.sort()
        self.image_list = files
        self.current_index = 0
        self.load_current_file()
        # フォルダ読込後に自動検出をオファー
        self.root.after(200, self._offer_folder_auto_detect)

    def load_current_file(self):
        """画像か動画かに応じてロード先を振り分ける"""
        if not self.image_list:
            return
        path = self.image_list[self.current_index]
        if path.lower().endswith(SUPPORTED_VIDEO_EXT):
            self.load_video(path)
        else:
            self.load_current_image(path)

    def load_current_image(self, path: str = None):
        """画像モードでファイルをロード"""
        # 動画状態をリセット
        if self.video_cap is not None:
            self.video_cap.release()
            self.video_cap = None
        self.is_video = False
        self.video_masks = {}
        if self.frame_label:
            self.frame_label.config(text="")

        if path is None:
            if not self.image_list:
                return
            path = self.image_list[self.current_index]

        self.current_path = path
        img = Image.open(path).convert("RGB")
        self.original_image = img.copy()
        w, h = img.size
        # 保存済みマスクがあれば復元
        mask_path = self.get_mask_path(path)
        if mask_path and os.path.exists(mask_path):
            try:
                data = np.load(mask_path)
                loaded = data["mask"]
                if loaded.shape == (h, w):
                    self.mosaic_mask = loaded.copy()
                else:
                    # サイズ不一致はリサイズして対応
                    self.mosaic_mask = cv2.resize(loaded, (w, h), interpolation=cv2.INTER_NEAREST)
            except Exception:
                self.mosaic_mask = np.zeros((h, w), dtype=np.uint8)
        else:
            self.mosaic_mask = np.zeros((h, w), dtype=np.uint8)
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.clear_selection()
        self.zoom = 1.0
        self._canvas_xview = 0.0
        self._canvas_yview = 0.0
        self.update_view()
        self.root.title(
            f"Mosaic Editor - {os.path.basename(path)} ({self.current_index + 1}/{len(self.image_list)})"
        )

    def load_video(self, path: str):
        """動画モードでファイルをロード"""
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            messagebox.showerror("エラー", f"動画を開けませんでした:\n{path}")
            return
        # 前の動画を解放
        if self.video_cap is not None:
            self.video_cap.release()

        self.is_video = True
        self.video_cap = cap
        self.video_total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        self.video_frame_index = 0
        self.video_masks = {}
        self.current_path = path
        self.undo_stack.clear()
        self.redo_stack.clear()
        self.clear_selection()
        self.zoom = 1.0
        self._canvas_xview = 0.0
        self._canvas_yview = 0.0
        self.root.title(
            f"Mosaic Editor [VIDEO] - {os.path.basename(path)} ({self.current_index + 1}/{len(self.image_list)})"
        )
        self.load_frame_at(0)

    def load_frame_at(self, index: int):
        """指定フレームを読み込んで表示"""
        if self.video_cap is None:
            return
        index = max(0, min(self.video_total_frames - 1, index))
        self.video_frame_index = index
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ret, frame_bgr = self.video_cap.read()
        if not ret:
            return
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        self.original_image = Image.fromarray(frame_rgb)
        h, w = frame_bgr.shape[:2]
        # そのフレームのマスクを取得（未編集なら零写起こし）
        self.mosaic_mask = self.video_masks.get(index, np.zeros((h, w), dtype=np.uint8)).copy()
        self.update_view()
        if self.frame_label:
            self.frame_label.config(
                text=f"フレーム: {index + 1} / {self.video_total_frames}  ({index / self.video_fps:.2f}s)"
            )

    def _navigate_frame(self, delta: int):
        """現在フレームのマスクを保存してdelta分移動"""
        if not self.is_video:
            return
        # 現在フレームのマスクを保存
        if self.mosaic_mask is not None and np.any(self.mosaic_mask):
            self.video_masks[self.video_frame_index] = self.mosaic_mask.copy()
        self.load_frame_at(self.video_frame_index + delta)

    def save_current(self, show_dialog=True):
        if self.current_path is None or self.output_folder is None:
            return

        if self.is_video:
            # 動画モード: 現在フレームのマスクを保存して動画をエクスポート
            if self.mosaic_mask is not None and np.any(self.mosaic_mask):
                self.video_masks[self.video_frame_index] = self.mosaic_mask.copy()
            if show_dialog:
                self._save_video()
            return

        # 画像モードの保存
        clean_np = self.generate_mosaic_image()
        if clean_np is None:
            return

        img_to_save = Image.fromarray(clean_np)
        path = self.current_path
        folder = self.output_folder

        # 拡張子をjpgに変更
        base_name = os.path.splitext(os.path.basename(path))[0]
        save_filename = base_name + ".jpg"
        save_path = os.path.join(folder, save_filename)

        # JPEG形式で保存（品質95）
        img_to_save.save(save_path, "JPEG", quality=95)

        # マスクをNPZに保存（空ならファイル削除）
        mask_path = self.get_mask_path(path)
        if mask_path:
            if self.mosaic_mask is not None and np.any(self.mosaic_mask):
                np.savez_compressed(mask_path, mask=self.mosaic_mask)
            elif os.path.exists(mask_path):
                try:
                    os.remove(mask_path)
                except Exception:
                    pass

        if show_dialog:
            messagebox.showinfo("保存", f"{save_path} に保存しました")

    def _save_video(self):
        """動画をモザイク適用してMP4で保存（バックグラウンド）"""
        if self.video_cap is None or self.current_path is None or self.output_folder is None:
            return

        base_name = os.path.splitext(os.path.basename(self.current_path))[0]
        save_path = os.path.join(self.output_folder, base_name + ".mp4")
        total = self.video_total_frames
        fps = self.video_fps

        # プログレスダイアログ
        prog_win = tk.Toplevel(self.root)
        prog_win.title("動画出力中...")
        prog_win.geometry("380x110")
        prog_win.resizable(False, False)
        prog_win.grab_set()
        tk.Label(prog_win, text=f"保存先: {save_path}",
                 font=("", 8), fg="gray", wraplength=360).pack(pady=(8, 2))
        bar = ttk.Progressbar(prog_win, maximum=total, length=340)
        bar.pack(padx=20, pady=4)
        pct_label = tk.Label(prog_win, text="0 / 0", font=("", 9))
        pct_label.pack()

        # 元画像サイズを取得
        self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        ret, first = self.video_cap.read()
        if not ret:
            prog_win.destroy()
            messagebox.showerror("エラー", "動画のフレームを読み出せません")
            return
        h, w = first.shape[:2]
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(save_path, fourcc, fps, (w, h))

        vid_masks = self.video_masks
        mosaic_sz = self.get_block_size()

        def worker():
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            for fi in range(total):
                ret, frame_bgr = self.video_cap.read()
                if not ret:
                    break
                mask = vid_masks.get(fi)
                if mask is not None and np.any(mask):
                    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                    block = max(2, mosaic_sz)
                    sh, sw = frame_rgb.shape[:2]
                    small = cv2.resize(frame_rgb, (max(1, sw // block), max(1, sh // block)),
                                       interpolation=cv2.INTER_LINEAR)
                    mosaic_full = cv2.resize(small, (sw, sh), interpolation=cv2.INTER_NEAREST)
                    mask_bool = mask > 0
                    frame_rgb[mask_bool] = mosaic_full[mask_bool]
                    frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
                writer.write(frame_bgr)
                if fi % 10 == 0:
                    self.root.after(0, lambda f=fi: (
                        bar.config(value=f),
                        pct_label.config(text=f"{f + 1} / {total}")
                    ))
            writer.release()
            self.root.after(0, lambda: _on_done())

        def _on_done():
            bar.config(value=total)
            prog_win.destroy()
            # 動画保存後、音声マージ処理に移行
            self._merge_audio(self.current_path, save_path)

        threading.Thread(target=worker, daemon=True).start()

    def _merge_audio(self, orig_path: str, new_path: str):
        """保存された無音動画に元の音声を結合する"""
        try:
            import moviepy.editor as mp  # type: ignore
            self._run_audio_merge(orig_path, new_path)
        except ImportError:
            if messagebox.askyesno(
                "moviepy 自動インストール",
                "動画の音声を維持するために 'moviepy' ライブラリが必要です。\n"
                "自動的にインストールしますか？\n\n"
                "（「いいえ」を選ぶと無音のまま完了します）"
            ):
                self._install_moviepy_and_merge(orig_path, new_path)
            else:
                messagebox.showinfo("保存完了", f"無音のまま動画を保存しました:\n{new_path}")

    def _install_moviepy_and_merge(self, orig_path: str, new_path: str):
        import subprocess
        import queue

        progress_win = tk.Toplevel(self.root)
        progress_win.title("moviepy インストール中")
        progress_win.geometry("520x300")
        progress_win.resizable(True, True)
        progress_win.grab_set()

        tk.Label(progress_win, text="moviepy をインストール中...",
                 font=("", 10, "bold"), pady=6).pack()

        bar = ttk.Progressbar(progress_win, mode="indeterminate", length=490)
        bar.pack(padx=10)
        bar.start(12)

        frame = tk.Frame(progress_win)
        frame.pack(fill="both", expand=True, padx=10, pady=6)
        log_text = tk.Text(frame, height=12, wrap="word", state="disabled",
                           bg="#1e1e1e", fg="#cccccc", font=("Consolas", 9))
        log_text.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(frame, command=log_text.yview)
        sb.pack(side="right", fill="y")
        log_text.config(yscrollcommand=sb.set)

        log_queue: "queue.Queue[str]" = __import__("queue").Queue()

        def append_log(line: str):
            log_text.config(state="normal")
            log_text.insert("end", line)
            log_text.see("end")
            log_text.config(state="disabled")

        def poll_queue():
            try:
                while True:
                    line = log_queue.get_nowait()
                    append_log(line)
            except __import__("queue").Empty:
                pass
            if progress_win.winfo_exists():
                progress_win.after(100, poll_queue)

        def do_install():
            def run_pip(*args):
                proc = subprocess.Popen(
                    [sys.executable, "-m", "pip"] + list(args),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                for line in proc.stdout:  # type: ignore
                    log_queue.put(line)
                proc.wait()
                return proc.returncode == 0

            log_queue.put("\n⏳ moviepy をインストール中...\n")
            ok = run_pip("install", "--no-cache-dir", "moviepy")
            if not ok:
                log_queue.put("❌ moviepy インストール失敗\n")
                self.root.after(0, lambda: self._moviepy_install_fail(progress_win, new_path, "moviepy インストール失敗"))
                return

            log_queue.put("\n✅ インストール完了！音声合成を開始します。\n")
            self.root.after(0, lambda: self._moviepy_install_success(progress_win, orig_path, new_path))

        progress_win.after(100, poll_queue)
        threading.Thread(target=do_install, daemon=True).start()

    def _moviepy_install_fail(self, progress_win, new_path: str, err: str):
        progress_win.title("インストール失敗")
        for w in progress_win.winfo_children():
            if isinstance(w, ttk.Progressbar):
                w.stop()
        tk.Label(
            progress_win,
            text=f"❌ {err}\n\n手動インストール:\n  pip install moviepy\n\n（今回は無音のまま完了します）",
            fg="red", justify="left", wraplength=480, pady=6
        ).pack()
        tk.Button(progress_win, text="閉じる", command=lambda: (progress_win.destroy(), messagebox.showinfo("保存完了", f"無音のまま保存しました:\n{new_path}")),
                  bg="#cc4444", fg="white", relief="flat", padx=12, pady=4).pack(pady=6)

    def _moviepy_install_success(self, progress_win, orig_path: str, new_path: str):
        progress_win.destroy()
        self._run_audio_merge(orig_path, new_path)

    def _run_audio_merge(self, orig_path: str, new_path: str):
        wait_win = tk.Toplevel(self.root)
        wait_win.title("音声結合中...")
        wait_win.geometry("380x100")
        wait_win.resizable(False, False)
        wait_win.grab_set()
        tk.Label(wait_win, text="元の動画から音声を抽出して結合しています...\n（処理に時間がかかる場合があります）", pady=12).pack()
        bar = ttk.Progressbar(wait_win, mode="indeterminate", length=340)
        bar.pack(padx=20)
        bar.start(10)

        def merge_worker():
            try:
                import moviepy.editor as mp  # type: ignore
                import tempfile
                import shutil
                
                # moviepyで元動画と出力動画を開く
                orig_clip = mp.VideoFileClip(orig_path)
                new_clip = mp.VideoFileClip(new_path)
                
                if orig_clip.audio is None:
                    # 元から音声がない場合
                    orig_clip.close()
                    new_clip.close()
                    self.root.after(0, lambda: _on_merge_done(new_path, "元動画に音声がありませんでした。"))
                    return

                # 音声をセット
                final_clip = new_clip.set_audio(orig_clip.audio)

                # 一時ファイルとして書き出し
                tmp_fd, tmp_path = tempfile.mkstemp(suffix=".mp4")
                os.close(tmp_fd)
                
                # ロガー出力を抑制して書き出し
                final_clip.write_videofile(tmp_path, codec="libx264", audio_codec="aac", logger=None)
                
                orig_clip.close()
                new_clip.close()
                final_clip.close()

                # 上書き
                shutil.move(tmp_path, new_path)
                
                self.root.after(0, lambda: _on_merge_done(new_path, "音声の結合が完了しました。"))

            except Exception as e:
                err_msg = str(e)
                self.root.after(0, lambda emsg=err_msg: _on_merge_error(new_path, emsg))

        def _on_merge_done(saved_path, ext_msg):
            try:
                wait_win.destroy()
            except Exception:
                pass
            messagebox.showinfo("保存完了", f"{ext_msg}\n{saved_path}")

        def _on_merge_error(saved_path, err):
            try:
                wait_win.destroy()
            except Exception:
                pass
            messagebox.showwarning("音声結合エラー", f"音声の結合に失敗しました（無音で保存されます）:\n{err}")
            messagebox.showinfo("保存完了", f"{saved_path}")

        threading.Thread(target=merge_worker, daemon=True).start()

    def display_image(self, preview=None):
        img_src = preview if preview else self.image
        if img_src is None:
            return

        new_w = int(img_src.width * self.zoom)
        new_h = int(img_src.height * self.zoom)

        resized = img_src.resize((new_w, new_h), Image.NEAREST)
        self.tk_image = ImageTk.PhotoImage(resized)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        self.canvas.config(scrollregion=(0, 0, new_w, new_h))

        if hasattr(self, 'selection_rect') and self.selection_rect is not None:
            x1, y1, x2, y2 = self.selection_rect
            self.canvas.create_rectangle(
                x1 * self.zoom, y1 * self.zoom, x2 * self.zoom, y2 * self.zoom,
                outline="red", width=2, dash=(4, 4), tags=self.selection_tag
            )

    def next_image(self):
        if self.is_video:
            # 動画時は現在フレームのマスクを保存して次のファイルへ
            if self.mosaic_mask is not None and np.any(self.mosaic_mask):
                self.video_masks[self.video_frame_index] = self.mosaic_mask.copy()
        else:
            self.save_current(show_dialog=False)
        if hasattr(self, 'image_list') and self.current_index < len(self.image_list) - 1:
            self.current_index += 1
            self.load_current_file()

    def prev_image(self):
        if self.is_video:
            if self.mosaic_mask is not None and np.any(self.mosaic_mask):
                self.video_masks[self.video_frame_index] = self.mosaic_mask.copy()
        else:
            self.save_current(show_dialog=False)
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_file()

    # ================= 自動検出 (YOLO) =================

    # ── モデルパス管理 ──────────────────────────────────────

    def _find_yolo_model(self) -> str | None:
        """YOLOモデル(.pt)を自動探索する。見つからなければNoneを返す。"""
        import pathlib
        search_dirs = [
            pathlib.Path(__file__).parent,          # mosaic_editor/ フォルダ
            pathlib.Path.home() / "yolo_models",    # ~/yolo_models/
        ]
        for d in search_dirs:
            if not d.exists():
                continue
            pts = sorted(d.glob("*.pt"))
            if pts:
                return str(pts[0])
        return None

    def select_yolo_model(self):
        """ファイルダイアログでYOLOモデルを選択してインスタンス変数に保存"""
        path = filedialog.askopenfilename(
            title="YOLOモデル (.pt) を選択",
            filetypes=[("YOLO model", "*.pt"), ("All files", "*.*")]
        )
        if path:
            self._yolo_model_path = path
            messagebox.showinfo("モデル設定", f"モデルをセットしました:\n{path}")

    def _get_yolo_model_path(self) -> str | None:
        """現在設定済みのモデルパスを取得。なければ自動探索、それもなければダイアログ。"""
        # インスタンスに保存されていればそれを使う
        if hasattr(self, '_yolo_model_path') and self._yolo_model_path:
            import os
            if os.path.isfile(self._yolo_model_path):
                return self._yolo_model_path
        # 自動探索
        found = self._find_yolo_model()
        if found:
            self._yolo_model_path = found
            return found
        # ダイアログで選択を促す
        messagebox.showinfo(
            "YOLOモデルが見つかりません",
            "YOLOモデル (.pt) が見つかりませんでした。\n\n"
            "以下のいずれかに .pt ファイルを置いてください:\n"
            "  ① mosaic_editor/ フォルダ内\n"
            "  ② ~/yolo_models/ フォルダ内\n\n"
            "または次のダイアログでファイルを直接選択してください。"
        )
        path = filedialog.askopenfilename(
            title="YOLOモデル (.pt) を選択",
            filetypes=[("YOLO model", "*.pt"), ("All files", "*.*")]
        )
        if path:
            self._yolo_model_path = path
            return path
        return None

    # ── エントリーポイント ──────────────────────────────────

    def auto_detect_yolo(self):
        """YOLOモデルで自動検出してモザイクマスクに追加"""
        if self.original_image is None:
            messagebox.showwarning("警告", "画像を開いてください")
            return

        model_path = self._get_yolo_model_path()
        if not model_path:
            return

        conf_win = tk.Toplevel(self.root)
        conf_win.title("YOLO 検出設定")
        conf_win.geometry("360x280" if self.is_video else "360x240")
        conf_win.resizable(False, False)
        conf_win.grab_set()

        tk.Label(conf_win, text="信頼度スコア閾値 (推奨: 0.5〜0.8)",
                 font=("", 9)).pack(pady=(8, 0))
        conf_var = tk.DoubleVar(value=getattr(self, '_yolo_conf', 0.5))
        conf_scale = tk.Scale(conf_win, from_=0.1, to=1.0, resolution=0.05,
                              variable=conf_var, orient=tk.HORIZONTAL, length=260,
                              showvalue=True)
        conf_scale.pack(padx=20)

        # 対象クラス選択
        tk.Label(conf_win, text="検出対象:", font=("", 9)).pack(pady=(4, 0))
        cls_frm = tk.Frame(conf_win)
        cls_frm.pack(pady=2)
        
        # 過去の選択状態があれば復元、なければ全てTrue
        if not hasattr(self, '_yolo_target_classes'):
            self._yolo_target_classes = {
                "nipples": True, "pussy": True, "anus": True, "penis": True,
                "testicles": True, "x-ray": True, "cross-section": True
            }
        
        cls_vars = {}
        row, col = 0, 0
        for i, cname in enumerate(self._yolo_target_classes.keys()):
            var = tk.BooleanVar(value=self._yolo_target_classes[cname])
            cls_vars[cname] = var
            tk.Checkbutton(cls_frm, text=cname, variable=var).grid(row=row, column=col, sticky="w", padx=4)
            col += 1
            if col > 2:
                col = 0
                row += 1

        # 動画の場合のみ全フレーム選択UIを追加
        process_all_var = tk.BooleanVar(value=False)
        if self.is_video:
            tk.Label(conf_win, text="処理対象:", font=("", 9)).pack(pady=(6, 0))
            frm_rad = tk.Frame(conf_win)
            frm_rad.pack()
            tk.Radiobutton(frm_rad, text="現在のフレーム", variable=process_all_var, value=False).pack(side="left", padx=4)
            tk.Radiobutton(frm_rad, text="動画全フレーム", variable=process_all_var, value=True).pack(side="left", padx=4)

        btn_frm = tk.Frame(conf_win)
        btn_frm.pack(pady=8)
        do_run = {"ok": False}

        def on_run():
            do_run["ok"] = True
            conf_win.destroy()

        tk.Button(btn_frm, text="検出開始", command=on_run,
                  bg="#3a7bd5", fg="white", relief="flat",
                  padx=12, pady=4).pack(side="left", padx=6)
        tk.Button(btn_frm, text="キャンセル", command=conf_win.destroy,
                  relief="flat", padx=8, pady=4).pack(side="left", padx=6)

        conf_win.wait_window()
        if not do_run["ok"]:
            return

        conf = conf_var.get()
        process_all = process_all_var.get()
        self._yolo_conf = conf  # 次回起動時の初期値として記憶
        
        # 選択されたクラスを記憶
        target_classes = [cname for cname, var in cls_vars.items() if var.get()]
        for cname in self._yolo_target_classes:
            self._yolo_target_classes[cname] = cname in target_classes

        # ultralytics がインストール済みか確認
        try:
            import ultralytics  # type: ignore  # noqa: F401
            if not process_all:
                self.push_history()
            self._run_yolo_detection(model_path, conf=conf, process_all=process_all, target_classes=target_classes)
        except ImportError:
            if not messagebox.askyesno(
                "ultralytics 自動インストール",
                "ultralytics がインストールされていません。\n"
                "自動的にインストールしますか？\n\n"
                "（pip install ultralytics を実行します）"
            ):
                return
            self._install_and_detect_yolo(model_path, conf=conf, process_all=process_all, target_classes=target_classes)

    # ── インストール ────────────────────────────────────────

    def _install_and_detect_yolo(self, model_path: str, conf: float = 0.5, process_all: bool = False, target_classes: list = None):
        """バックグラウンドで ultralytics をインストールして検出を実行する"""
        import subprocess
        import queue

        progress_win = tk.Toplevel(self.root)
        progress_win.title("ultralytics インストール中")
        progress_win.geometry("520x300")
        progress_win.resizable(True, True)
        progress_win.grab_set()

        tk.Label(progress_win, text="ultralytics をインストール中...",
                 font=("", 10, "bold"), pady=6).pack()

        bar = ttk.Progressbar(progress_win, mode="indeterminate", length=490)
        bar.pack(padx=10)
        bar.start(12)

        frame = tk.Frame(progress_win)
        frame.pack(fill="both", expand=True, padx=10, pady=6)
        log_text = tk.Text(frame, height=12, wrap="word", state="disabled",
                           bg="#1e1e1e", fg="#cccccc", font=("Consolas", 9))
        log_text.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(frame, command=log_text.yview)
        sb.pack(side="right", fill="y")
        log_text.config(yscrollcommand=sb.set)

        log_queue: "queue.Queue[str]" = __import__("queue").Queue()

        def append_log(line: str):
            log_text.config(state="normal")
            log_text.insert("end", line)
            log_text.see("end")
            log_text.config(state="disabled")

        def poll_queue():
            try:
                while True:
                    line = log_queue.get_nowait()
                    append_log(line)
            except __import__("queue").Empty:
                pass
            if progress_win.winfo_exists():
                progress_win.after(100, poll_queue)

        def do_install():
            def run_pip(*args):
                proc = subprocess.Popen(
                    [sys.executable, "-m", "pip"] + list(args),
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    encoding="utf-8",
                    errors="replace",
                )
                for line in proc.stdout:  # type: ignore
                    log_queue.put(line)
                proc.wait()
                return proc.returncode == 0

            log_queue.put("\n⏳ ultralytics をインストール中...\n")
            ok = run_pip("install", "--upgrade", "--no-cache-dir", "ultralytics")
            if not ok:
                log_queue.put("❌ ultralytics インストール失敗\n")
                self.root.after(0, lambda: self._yolo_install_fail(progress_win, "ultralytics インストール失敗"))
                return

            log_queue.put("\n✅ インストール完了！検出を開始します。\n")
            self.root.after(0, lambda: self._yolo_install_success(progress_win, model_path, conf, process_all, target_classes))

        progress_win.after(100, poll_queue)
        threading.Thread(target=do_install, daemon=True).start()

    def _yolo_install_success(self, progress_win, model_path: str, conf: float = 0.5, process_all: bool = False, target_classes: list = None):
        progress_win.destroy()
        messagebox.showinfo("完了", "ultralytics のインストールが完了しました！\n検出を開始します。")
        if not process_all:
            self.push_history()
        self._run_yolo_detection(model_path, conf=conf, process_all=process_all, target_classes=target_classes)

    def _yolo_install_fail(self, progress_win, err: str):
        progress_win.title("インストール失敗")
        for w in progress_win.winfo_children():
            if isinstance(w, ttk.Progressbar):
                w.stop()
        tk.Label(
            progress_win,
            text=f"❌ {err}\n\n手動インストール:\n  pip install ultralytics",
            fg="red", justify="left", wraplength=480, pady=6
        ).pack()
        tk.Button(progress_win, text="閉じる", command=progress_win.destroy,
                  bg="#cc4444", fg="white", relief="flat", padx=12, pady=4).pack(pady=6)

    # ── 推論 ────────────────────────────────────────────────

    def _run_yolo_detection(self, model_path: str, conf: float = 0.5, process_all: bool = False, target_classes: list = None):
        """YOLOで推論実行（セグメンテーション対応 / バックグラウンドスレッド）"""
        if self.original_image is None:
            return

        wait_win = tk.Toplevel(self.root)
        wait_win.title("YOLO 自動検出中")
        wait_win.geometry("360x130" if process_all else "360x100")
        wait_win.resizable(False, False)
        wait_win.grab_set()

        msg = "全フレーム一括検出中..." if process_all else "検出中です。しばらくお待ちください..."
        tk.Label(wait_win, text=msg, pady=12).pack()

        if process_all:
            bar = ttk.Progressbar(wait_win, maximum=max(1, self.video_total_frames), length=320)
            bar.pack(padx=20)
            pct_label = tk.Label(wait_win, text=f"0 / {self.video_total_frames}", font=("", 9))
            pct_label.pack()
        else:
            bar = ttk.Progressbar(wait_win, mode="indeterminate", length=320)
            bar.pack(padx=20)
            bar.start(10)

        tk.Label(wait_win, text=f"conf={conf:.1f}  model={os.path.basename(model_path)}",
                 font=("", 8), fg="gray").pack()

        # キャンセル用フラグ
        self._yolo_cancel = False
        if process_all:
            def _cancel():
                self._yolo_cancel = True
                wait_win.destroy()
            tk.Button(wait_win, text="キャンセル", command=_cancel,
                      relief="flat", padx=8, pady=0).pack(pady=4)

        current_img_copy = self.original_image.copy()

        def detect_worker():
            try:
                from ultralytics import YOLO  # type: ignore
                import tempfile

                model = YOLO(model_path)

                if not process_all:
                    # 単一フレーム処理
                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                        tmp_path = tmp.name
                        current_img_copy.save(tmp_path, "JPEG", quality=95)

                    results = model(tmp_path, verbose=False, conf=conf, imgsz=640)
                    os.unlink(tmp_path)
                    if not self._yolo_cancel:
                        self.root.after(0, lambda: self._apply_yolo_result(
                            wait_win, results, model_path, current_img_copy, target_classes))
                else:
                    # 全フレーム一括処理
                    vid_masks = self.video_masks
                    total = self.video_total_frames
                    w, h = self.original_image.size
                    self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                        tmp_path = tmp.name

                    applied_frames = 0
                    for fi in range(total):
                        if self._yolo_cancel:
                            break
                        ret, frame_bgr = self.video_cap.read()
                        if not ret:
                            break
                        
                        # RGB変換してから一時保存
                        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
                        pil_img = Image.fromarray(frame_rgb)
                        pil_img.save(tmp_path, "JPEG", quality=95)

                        # 推論
                        results = model(tmp_path, verbose=False, conf=conf, imgsz=640)
                        
                        # 自動マスク生成
                        mask_np = np.zeros((h, w), dtype=np.uint8)
                        has_applied = False
                        
                        try:
                            boxes = results[0].boxes
                            masks_data = results[0].masks
                            class_names = results[0].names
                            
                            if masks_data is not None and len(masks_data) > 0:
                                # セグメント適用
                                seg_masks = masks_data.data.cpu().numpy()
                                for i in range(len(seg_masks)):
                                    cls_id = int(boxes.cls[i]) if boxes is not None else 0
                                    cls_name = class_names.get(cls_id, str(cls_id))
                                    if target_classes and cls_name not in target_classes:
                                        continue
                                        
                                    if float(boxes.conf[i]) >= conf:
                                        m_f = seg_masks[i]
                                        m_u8 = (m_f * 255).astype(np.uint8)
                                        m_res = cv2.resize(m_u8, (w, h), interpolation=cv2.INTER_LINEAR)
                                        mask_np[m_res > 127] = 255
                                        has_applied = True
                            elif boxes is not None and len(boxes) > 0:
                                # バウディングボックスフォールバック
                                for i in range(len(boxes)):
                                    cls_id = int(boxes.cls[i])
                                    cls_name = class_names.get(cls_id, str(cls_id))
                                    if target_classes and cls_name not in target_classes:
                                        continue

                                    if float(boxes.conf[i]) >= conf:
                                        xyxy = boxes.xyxy[i].tolist()
                                        bx1, by1, bx2, by2 = xyxy
                                        if max(abs(bx1), abs(by1), abs(bx2), abs(by2)) <= 1.0:
                                            bx1, by1 = bx1 * w, by1 * h
                                            bx2, by2 = bx2 * w, by2 * h
                                        x1, y1 = max(0, int(min(bx1, bx2))), max(0, int(min(by1, by2)))
                                        x2, y2 = min(w, int(max(bx1, bx2))), min(h, int(max(by1, by2)))
                                        mask_np[y1:y2, x1:x2] = 255
                                        has_applied = True
                        except Exception:
                            pass
                        
                        if has_applied:
                            # 既存の手動マスクがあれば合成、なければ新規
                            if self.video_masks.get(fi) is not None:
                                self.video_masks[fi] = np.maximum(self.video_masks[fi], mask_np)
                            else:
                                self.video_masks[fi] = mask_np
                            applied_frames += 1

                        if fi % 5 == 0:
                            self.root.after(0, lambda f=fi: (
                                bar.config(value=f+1),
                                pct_label.config(text=f"{f+1} / {total}")
                            ))

                    try:
                        os.unlink(tmp_path)
                    except Exception:
                        pass
                    
                    if not self._yolo_cancel:
                        self.root.after(0, lambda: _on_batch_done(applied_frames))
                        
            except Exception as e:
                err_msg = str(e)
                self.root.after(0, lambda emsg=err_msg: self._yolo_error(wait_win, emsg))

        def _on_batch_done(applied_count):
            try:
                wait_win.destroy()
            except Exception:
                pass
            # カレントフレームの表示を再読み込みして反映
            self.load_frame_at(self.video_frame_index)
            messagebox.showinfo("一括検出完了", f"全 {self.video_total_frames} フレーム中、\n{applied_count} フレームにモザイクを適用しました。")

        threading.Thread(target=detect_worker, daemon=True).start()

    # ── 結果適用 ────────────────────────────────────────────

    def _apply_yolo_result(self, wait_win, results, model_path: str,
                           orig_pil=None, target_classes: list = None):
        """検出結果をプレビューダイアログで確認後にマスクへ反映（seg/det両対応）"""
        try:
            wait_win.destroy()
        except Exception:
            pass

        if self.mosaic_mask is None or self.original_image is None:
            return

        img_h, img_w = self.mosaic_mask.shape[:2]

        # クラス名マップ
        class_names: dict = {}
        try:
            class_names = results[0].names
        except Exception:
            pass

        # ── セグメンテーションマスクを解析 ──────────────────
        candidates = []
        has_seg = False
        try:
            masks_data = results[0].masks  # None if detection-only
            boxes = results[0].boxes
            if masks_data is not None and len(masks_data) > 0:
                has_seg = True
                # masks.data : (N, mask_h, mask_w) float tensor 0~1
                seg_masks = masks_data.data.cpu().numpy()  # (N, mh, mw)
                for i in range(len(seg_masks)):
                    conf_val = float(boxes.conf[i]) if boxes is not None else 0.0
                    cls_id   = int(boxes.cls[i]) if boxes is not None else 0
                    cls_name = class_names.get(cls_id, str(cls_id))
                    
                    if target_classes and cls_name not in target_classes:
                        continue

                    # バウンディングボックス（表示用）
                    xyxy = boxes.xyxy[i].tolist() if boxes is not None else [0, 0, img_w, img_h]
                    bx1, by1, bx2, by2 = xyxy
                    if max(abs(bx1), abs(by1), abs(bx2), abs(by2)) <= 1.0:
                        bx1, by1 = bx1 * img_w, by1 * img_h
                        bx2, by2 = bx2 * img_w, by2 * img_h
                    x1 = max(0, min(img_w, int(min(bx1, bx2))))
                    y1 = max(0, min(img_h, int(min(by1, by2))))
                    x2 = max(0, min(img_w, int(max(bx1, bx2))))
                    y2 = max(0, min(img_h, int(max(by1, by2))))

                    candidates.append({
                        "cls_id": cls_id,
                        "cls_name": cls_name,
                        "conf": conf_val,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "seg_idx": i,  # seg_masks のインデックス
                    })
            elif boxes is not None and len(boxes) > 0:
                # detection only fallback
                for i in range(len(boxes)):
                    xyxy = boxes.xyxy[i].tolist()
                    conf_val = float(boxes.conf[i])
                    cls_id   = int(boxes.cls[i])
                    cls_name = class_names.get(cls_id, str(cls_id))
                    
                    if target_classes and cls_name not in target_classes:
                        continue
                        
                    bx1, by1, bx2, by2 = xyxy
                    if max(abs(bx1), abs(by1), abs(bx2), abs(by2)) <= 1.0:
                        bx1, by1 = bx1 * img_w, by1 * img_h
                        bx2, by2 = bx2 * img_w, by2 * img_h
                    x1 = max(0, min(img_w, int(min(bx1, bx2))))
                    y1 = max(0, min(img_h, int(min(by1, by2))))
                    x2 = max(0, min(img_w, int(max(bx1, bx2))))
                    y2 = max(0, min(img_h, int(max(by1, by2))))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    candidates.append({
                        "cls_id": cls_id, "cls_name": cls_name, "conf": conf_val,
                        "x1": x1, "y1": y1, "x2": x2, "y2": y2,
                        "seg_idx": None,
                    })
        except Exception as e:
            messagebox.showerror("YOLO エラー", f"結果の解析に失敗しました:\n{e}")
            return

        model_name = os.path.basename(model_path)
        mode_label = "セグメント" if has_seg else "バウンディングボックス"

        if not candidates:
            messagebox.showinfo(
                f"自動検出 (YOLO / {model_name}) — 0件",
                "検出結果が見つかりませんでした。\n\n"
                "・信頼度が低い可能性があります（閾値を下げてみてください）。\n"
                f"・使用モデル: {model_name}\n"
                f"・画像サイズ: {img_w}x{img_h}px"
            )
            return

        # ── プレビューダイアログ ─────────────────────────────
        dlg = tk.Toplevel(self.root)
        dlg.title(f"検出結果 ({len(candidates)}件 / {mode_label}) — {model_name}")
        dlg.geometry("560x400")
        dlg.grab_set()

        tk.Label(dlg, text="チェックしたものをモザイク選択範囲に追加します",
                 font=("", 9), fg="gray").pack(pady=(6, 0))
        tk.Label(dlg, text=f"モデル: {model_name}  [{mode_label}]",
                 font=("", 8), fg="#666666").pack()

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

        vars_ = []
        for c in candidates:
            v = tk.BooleanVar(value=True)
            vars_.append(v)
            seg_tag = "[seg]" if c["seg_idx"] is not None else "[box]"
            tk.Checkbutton(
                inner,
                text=f"{seg_tag} {c['cls_name']}  conf={c['conf']:.0%}  "
                     f"({c['x1']},{c['y1']})-({c['x2']},{c['y2']})",
                variable=v,
                anchor="w",
            ).pack(fill="x", padx=6, pady=1)

        btn_frame = tk.Frame(dlg)
        btn_frame.pack(pady=8)
        result = {"ok": False}

        def on_apply():
            result["ok"] = True
            dlg.destroy()

        tk.Button(btn_frame, text="選択範囲に追加", command=on_apply,
                  bg="#3a7bd5", fg="white", relief="flat", padx=10, pady=4).pack(side="left", padx=6)
        tk.Button(btn_frame, text="キャンセル", command=dlg.destroy,
                  relief="flat", padx=10, pady=4).pack(side="left", padx=6)

        dlg.wait_window()

        if not result["ok"]:
            return

        # ── マスクへ反映 ──────────────────────────────────────
        applied = 0
        seg_masks_np = None
        if has_seg:
            try:
                seg_masks_np = results[0].masks.data.cpu().numpy()
            except Exception:
                seg_masks_np = None

        for v, c in zip(vars_, candidates):
            if not v.get():
                continue
            idx = c["seg_idx"]
            if has_seg and seg_masks_np is not None and idx is not None:
                # セグメントマスクを元画像サイズにリサイズして適用
                mask_f = seg_masks_np[idx]  # (mh, mw) float 0~1
                mask_u8 = (mask_f * 255).astype(np.uint8)
                mask_resized = cv2.resize(mask_u8, (img_w, img_h),
                                          interpolation=cv2.INTER_LINEAR)
                self.mosaic_mask[mask_resized > 127] = 255
            else:
                # バウンディングボックスで矩形塗り
                self.mosaic_mask[c["y1"]:c["y2"], c["x1"]:c["x2"]] = 255
            applied += 1

        self.show_mask.set(True)
        self.update_view()

        messagebox.showinfo(
            "適用完了",
            f"{applied} 箇所をモザイク選択範囲に追加しました。\n"
            f"({mode_label}モードで適用)\n"
            "ペン・魔法の杖・消しゴムで微調整してから保存してください。"
        )

    def _yolo_error(self, wait_win, err: str):
        try:
            wait_win.destroy()
        except Exception:
            pass
        messagebox.showerror("YOLO エラー", f"検出中にエラーが発生しました:\n{err}")

    # ================= フォルダ一括自動検出 =================

    def _offer_folder_auto_detect(self):
        """フォルダ読込後に自動検出をオファーする。YOLOモデルが見つからなければ何もしない。"""
        if not self.image_list:
            return
        # 画像ファイルのみ対象
        img_files = [p for p in self.image_list if p.lower().endswith(SUPPORTED_EXT)]
        if not img_files:
            return
        # YOLOモデルを自動探索（ダイアログは出さない）
        model_path = None
        if hasattr(self, '_yolo_model_path') and self._yolo_model_path:
            if os.path.isfile(self._yolo_model_path):
                model_path = self._yolo_model_path
        if model_path is None:
            model_path = self._find_yolo_model()
        if model_path is None:
            return  # モデルが無ければ静かにスキップ

        # 確認ダイアログ
        conf_dlg = tk.Toplevel(self.root)
        conf_dlg.title("自動モザイク")
        conf_dlg.geometry("380x220")
        conf_dlg.resizable(False, False)
        conf_dlg.grab_set()

        tk.Label(conf_dlg,
                 text=f"フォルダ内の画像 {len(img_files)} 枚に\n自動モザイク (YOLO) を適用しますか？",
                 font=("", 10), pady=8).pack()
        tk.Label(conf_dlg, text=f"モデル: {os.path.basename(model_path)}",
                 font=("", 8), fg="gray").pack()

        tk.Label(conf_dlg, text="信頼度閾値:", font=("", 9)).pack(pady=(6, 0))
        conf_var = tk.DoubleVar(value=getattr(self, '_yolo_conf', 0.5))
        tk.Scale(conf_dlg, from_=0.1, to=1.0, resolution=0.05,
                 variable=conf_var, orient=tk.HORIZONTAL, length=260,
                 showvalue=True).pack()

        overwrite_var = tk.BooleanVar(value=False)
        tk.Checkbutton(conf_dlg, text="既存のマスクも上書きする",
                       variable=overwrite_var).pack(pady=(4, 0))

        btn_frm = tk.Frame(conf_dlg)
        btn_frm.pack(pady=8)
        do_run = {"ok": False}

        def on_yes():
            do_run["ok"] = True
            conf_dlg.destroy()

        tk.Button(btn_frm, text="はい、適用する", command=on_yes,
                  bg="#3a7bd5", fg="white", relief="flat",
                  padx=12, pady=4).pack(side="left", padx=6)
        tk.Button(btn_frm, text="スキップ", command=conf_dlg.destroy,
                  relief="flat", padx=8, pady=4).pack(side="left", padx=6)

        conf_dlg.wait_window()
        if not do_run["ok"]:
            return

        conf = conf_var.get()
        self._yolo_conf = conf
        overwrite = overwrite_var.get()

        # ultralyticsが入っていなければインストール確認
        try:
            import ultralytics  # type: ignore  # noqa: F401
            self._auto_detect_folder_batch(model_path, img_files, conf, overwrite)
        except ImportError:
            if messagebox.askyesno(
                "ultralytics 自動インストール",
                "ultralytics がインストールされていません。\n自動的にインストールしますか？"
            ):
                self._install_and_folder_batch(model_path, img_files, conf, overwrite)

    def _install_and_folder_batch(self, model_path, img_files, conf, overwrite):
        """ultralyticsをインストールして一括検出を実行"""
        import subprocess, queue
        progress_win = tk.Toplevel(self.root)
        progress_win.title("ultralytics インストール中")
        progress_win.geometry("520x300")
        progress_win.resizable(True, True)
        progress_win.grab_set()
        tk.Label(progress_win, text="ultralytics をインストール中...",
                 font=("", 10, "bold"), pady=6).pack()
        bar = ttk.Progressbar(progress_win, mode="indeterminate", length=490)
        bar.pack(padx=10)
        bar.start(12)
        frame = tk.Frame(progress_win)
        frame.pack(fill="both", expand=True, padx=10, pady=6)
        log_text = tk.Text(frame, height=12, wrap="word", state="disabled",
                           bg="#1e1e1e", fg="#cccccc", font=("Consolas", 9))
        log_text.pack(side="left", fill="both", expand=True)
        sb = ttk.Scrollbar(frame, command=log_text.yview)
        sb.pack(side="right", fill="y")
        log_text.config(yscrollcommand=sb.set)
        log_queue: "queue.Queue[str]" = __import__("queue").Queue()

        def append_log(line):
            log_text.config(state="normal")
            log_text.insert("end", line)
            log_text.see("end")
            log_text.config(state="disabled")

        def poll_queue():
            try:
                while True:
                    line = log_queue.get_nowait()
                    append_log(line)
            except __import__("queue").Empty:
                pass
            if progress_win.winfo_exists():
                progress_win.after(100, poll_queue)

        def do_install():
            proc = subprocess.Popen(
                [sys.executable, "-m", "pip", "install", "--upgrade", "--no-cache-dir", "ultralytics"],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace"
            )
            for line in proc.stdout:
                log_queue.put(line)
            proc.wait()
            if proc.returncode == 0:
                log_queue.put("\n✅ インストール完了！検出を開始します。\n")
                self.root.after(0, lambda: (progress_win.destroy(),
                                            self._auto_detect_folder_batch(model_path, img_files, conf, overwrite)))
            else:
                log_queue.put("❌ インストール失敗\n")

        progress_win.after(100, poll_queue)
        threading.Thread(target=do_install, daemon=True).start()

    def _auto_detect_folder_batch(self, model_path: str, img_files: list,
                                   conf: float = 0.5, overwrite: bool = False):
        """全画像にYOLO自動検出を実行してマスクNPZを保存する"""
        total = len(img_files)

        wait_win = tk.Toplevel(self.root)
        wait_win.title("フォルダ一括自動検出中...")
        wait_win.geometry("400x140")
        wait_win.resizable(False, False)
        wait_win.grab_set()
        tk.Label(wait_win, text="全画像にYOLO自動モザイクを適用中...", pady=8).pack()
        bar = ttk.Progressbar(wait_win, maximum=total, length=360)
        bar.pack(padx=20)
        pct_label = tk.Label(wait_win, text=f"0 / {total}", font=("", 9))
        pct_label.pack(pady=2)

        self._batch_cancel = False

        def _cancel():
            self._batch_cancel = True
            wait_win.destroy()

        tk.Button(wait_win, text="キャンセル", command=_cancel,
                  relief="flat", padx=8, pady=2).pack(pady=4)

        def worker():
            try:
                from ultralytics import YOLO  # type: ignore
                import tempfile
                model = YOLO(model_path)
                applied = 0

                for fi, img_path in enumerate(img_files):
                    if self._batch_cancel:
                        break

                    mask_path = self.get_mask_path(img_path)
                    # 既存マスクがあってoverwriteでなければスキップ
                    if not overwrite and mask_path and os.path.exists(mask_path):
                        self.root.after(0, lambda f=fi: (
                            bar.config(value=f + 1),
                            pct_label.config(text=f"{f + 1} / {total}")
                        ))
                        continue

                    try:
                        img_pil = Image.open(img_path).convert("RGB")
                        w, h = img_pil.size
                    except Exception:
                        continue

                    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
                        tmp_path = tmp.name
                        img_pil.save(tmp_path, "JPEG", quality=95)

                    try:
                        results = model(tmp_path, verbose=False, conf=conf, imgsz=640)
                    except Exception:
                        results = None
                    finally:
                        try:
                            os.unlink(tmp_path)
                        except Exception:
                            pass

                    if results is None:
                        continue

                    mask_np = np.zeros((h, w), dtype=np.uint8)
                    has_applied = False
                    try:
                        boxes = results[0].boxes
                        masks_data = results[0].masks
                        class_names = results[0].names
                        if masks_data is not None and len(masks_data) > 0:
                            seg_masks = masks_data.data.cpu().numpy()
                            for i in range(len(seg_masks)):
                                cls_id = int(boxes.cls[i]) if boxes is not None else 0
                                cls_name = class_names.get(cls_id, str(cls_id))
                                if float(boxes.conf[i]) >= conf:
                                    m_f = seg_masks[i]
                                    m_u8 = (m_f * 255).astype(np.uint8)
                                    m_res = cv2.resize(m_u8, (w, h), interpolation=cv2.INTER_LINEAR)
                                    mask_np[m_res > 127] = 255
                                    has_applied = True
                        elif boxes is not None and len(boxes) > 0:
                            for i in range(len(boxes)):
                                if float(boxes.conf[i]) >= conf:
                                    xyxy = boxes.xyxy[i].tolist()
                                    bx1, by1, bx2, by2 = xyxy
                                    if max(abs(bx1), abs(by1), abs(bx2), abs(by2)) <= 1.0:
                                        bx1, by1 = bx1 * w, by1 * h
                                        bx2, by2 = bx2 * w, by2 * h
                                    x1 = max(0, int(min(bx1, bx2)))
                                    y1 = max(0, int(min(by1, by2)))
                                    x2 = min(w, int(max(bx1, bx2)))
                                    y2 = min(h, int(max(by1, by2)))
                                    mask_np[y1:y2, x1:x2] = 255
                                    has_applied = True
                    except Exception:
                        pass

                    if has_applied and mask_path:
                        np.savez_compressed(mask_path, mask=mask_np)
                        applied += 1

                    self.root.after(0, lambda f=fi: (
                        bar.config(value=f + 1),
                        pct_label.config(text=f"{f + 1} / {total}")
                    ))

                self.root.after(0, lambda: _on_done(applied))

            except Exception as e:
                err_msg = str(e)
                self.root.after(0, lambda emsg=err_msg: _on_error(emsg))

        def _on_done(applied_count):
            try:
                wait_win.destroy()
            except Exception:
                pass
            # 現在表示中の画像にマスクを反映
            if not self._batch_cancel:
                self.load_current_file()
                messagebox.showinfo(
                    "一括検出完了",
                    f"全 {total} 枚中、{applied_count} 枚にモザイクを適用しました。\n"
                    "手動で範囲を微調整してから保存してください。"
                )

        def _on_error(err):
            try:
                wait_win.destroy()
            except Exception:
                pass
            messagebox.showerror("自動検出エラー", f"一括検出中にエラーが発生しました:\n{err}")

        threading.Thread(target=worker, daemon=True).start()

def _ensure_tkinterdnd2():
    """tkinterdnd2がなければ自動インストールしてからimportする"""
    try:
        import tkinterdnd2  # type: ignore
        return tkinterdnd2
    except ImportError:
        import subprocess
        print("[MosaicEditor] tkinterdnd2をインストール中...")
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "tkinterdnd2"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            import importlib
            tkinterdnd2 = importlib.import_module("tkinterdnd2")
            print("[MosaicEditor] tkinterdnd2インストール完了")
            return tkinterdnd2
        except Exception as e:
            print(f"[MosaicEditor] tkinterdnd2インストール失敗: {e}")
            return None


if __name__ == "__main__":
    dnd = _ensure_tkinterdnd2()
    if dnd is not None:
        root = dnd.TkinterDnD.Tk()
    else:
        root = tk.Tk()
    root.geometry("1000x800")
    app = MosaicEditor(root)
    root.mainloop()
