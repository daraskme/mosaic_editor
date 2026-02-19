import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk, ImageDraw
import numpy as np
import cv2
import os
from typing import Optional, List

SUPPORTED_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


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

        # Mode: pen, wand, eraser (rect removed)
        self.mode = tk.StringVar(value="pen")
        self.threshold = tk.IntVar(value=40)
        self.brush_size = tk.IntVar(value=20)
        self.mosaic_size = tk.IntVar(value=10)  # User configurable mosaic block size
        # Trace mosaic size to update view on change
        self.mosaic_size.trace_add("write", self.on_mosaic_size_change)
        # Trace mosaic size to update view on change
        self.mosaic_size.trace_add("write", self.on_mosaic_size_change)
        self.show_mask = tk.BooleanVar(value=True) # Default to showing mask as requested "displayed in green"
        self.use_edge_detection = tk.BooleanVar(value=False) # New: Stop at edges

        self.undo_stack: List[np.ndarray] = []
        self.redo_stack: List[np.ndarray] = []

        self.zoom: float = 1.0
        # pan_x/y are practically unused with scan_dragto, removing tracked state
        
        self.mosaic_mask: Optional[np.ndarray] = None
        
        # Cursor preview tag
        self.cursor_tag = "cursor_preview"

        # Initialize canvas early to satisfy type checker (packed later in build_ui)
        self.canvas = tk.Canvas(self.root, cursor="crosshair", bg="gray")

        self.build_menu()
        self.build_ui()

    # ================= 座標変換 =================

    def canvas_to_image(self, event_x, event_y):
        # canvasx/canvasy converts window coordinate to canvas coordinate (accounting for scroll)
        cx = self.canvas.canvasx(event_x)
        cy = self.canvas.canvasy(event_y)
        
        # Convert canvas coordinate to image coordinate (accounting for zoom)
        ix = int(cx / self.zoom)
        iy = int(cy / self.zoom)
        return ix, iy

    # ================= Utility =================

    def get_block_size(self):
        # Use simple fixed size from slider
        size = self.mosaic_size.get()
        return max(2, size)

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
        filemenu.add_command(label="フォルダを開く", command=self.open_folder)
        filemenu.add_command(label="保存", command=self.save_current)
        menubar.add_cascade(label="ファイル", menu=filemenu)
        self.root.config(menu=menubar)

    def build_ui(self):
        top = tk.Frame(self.root)
        top.pack(fill="x", padx=5, pady=5)

        tk.Button(top, text="←", command=self.prev_image).pack(side="left")
        tk.Button(top, text="→", command=self.next_image).pack(side="left")
        
        tk.Button(top, text="戻す", command=self.undo).pack(side="left", padx=5)
        tk.Button(top, text="やり直す", command=self.redo).pack(side="left")
        
        tk.Button(top, text="＋", command=self.zoom_in).pack(side="left", padx=(10, 0))
        tk.Button(top, text="−", command=self.zoom_out).pack(side="left")

        # Settings Area
        settings_frame = tk.Frame(self.root)
        settings_frame.pack(fill="x", padx=5, pady=5)

        # Mode Selection
        mode_frame = tk.LabelFrame(settings_frame, text="モード")
        mode_frame.pack(side="left", padx=5)

        for text, val in [("ペン", "pen"),
                          ("魔法の杖", "wand"),
                          ("消しゴム", "eraser")]:
            tk.Radiobutton(mode_frame, text=text,
                           variable=self.mode, value=val).pack(side="left")

        # Sliders
        sliders_frame = tk.Frame(settings_frame)
        sliders_frame.pack(side="left", fill="x", expand=True, padx=5)

        # Threshold (Wand)
        tk.Label(sliders_frame, text="許容値 (魔法の杖)").grid(row=0, column=0, sticky="e")
        tk.Spinbox(sliders_frame, from_=0, to=255, width=5,
                   textvariable=self.threshold).grid(row=0, column=1, sticky="w")

        # Brush Size
        tk.Label(sliders_frame, text="ブラシサイズ").grid(row=0, column=2, sticky="e")
        tk.Spinbox(sliders_frame, from_=1, to=200, width=5,
                   textvariable=self.brush_size).grid(row=0, column=3, sticky="w")

        # Mosaic Size
        tk.Label(sliders_frame, text="モザイク強度").grid(row=0, column=4, sticky="e")
        tk.Spinbox(sliders_frame, from_=2, to=100, width=5,
                   textvariable=self.mosaic_size).grid(row=0, column=5, sticky="w")

        # Show Mask Toggle
        tk.Checkbutton(sliders_frame, text="範囲表示", variable=self.show_mask, 
                       command=self.update_view).grid(row=0, column=6, sticky="w", padx=4)

        # Edge Detection Toggle
        tk.Checkbutton(sliders_frame, text="境界線で止める", variable=self.use_edge_detection).grid(row=0, column=7, sticky="w", padx=4)

        # Canvas already initialized in __init__
        if self.canvas is not None:
             self.canvas.pack(fill="both", expand=True)

             self.canvas.bind("<ButtonPress-1>", self.on_click)
             self.canvas.bind("<B1-Motion>", self.on_drag)
             self.canvas.bind("<ButtonRelease-1>", self.on_release)
             self.canvas.bind("<Motion>", self.on_mouse_move)  # For cursor preview

             self.canvas.bind("<Button-4>", lambda e: self.zoom_in())
             self.canvas.bind("<Button-5>", lambda e: self.zoom_out())

             self.canvas.bind("<ButtonPress-2>", self.start_pan)
             self.canvas.bind("<B2-Motion>", self.do_pan)
        
        # Shortcuts
        self.root.bind("<Control-z>", self.undo)
        self.root.bind("<Control-y>", self.redo)
        
        # Bind window close event for auto-save
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def on_closing(self):
        self.save_current(show_dialog=False)
        self.root.destroy()
    
    def on_mosaic_size_change(self, *args):
        try:
             # Ensure valid integer
             val = self.mosaic_size.get()
             if val < 2: return
             self.update_view()
        except tk.TclError:
             pass # Ignore empty or invalid input while typing

    # ================= Zoom =================

    def zoom_in(self):
        if self.image is None:
            return
        self.zoom *= 1.25
        self.display_image()
        self.update_cursor_preview(None) # Force update or hide

    def zoom_out(self):
        if self.image is None:
            return
        self.zoom /= 1.25
        self.display_image()
        self.update_cursor_preview(None)

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
        if self.canvas is None: return
        self.canvas.delete(self.cursor_tag)
        
        if self.image is None or not event:
            return

        # Calculate canvas coordinates
        cx = self.canvas.canvasx(event.x)
        cy = self.canvas.canvasy(event.y)
        
        r = self.brush_size.get() * self.zoom # Scale radius by zoom
        
        color = "green"
        if self.mode.get() == "eraser":
            color = "white"
        
        # Draw circle
        self.canvas.create_oval(
            cx - r, cy - r, cx + r, cy + r,
            outline=color, width=2, tags=self.cursor_tag
        )

    def on_click(self, event):
        if self.image is None:
            return

        ix, iy = self.canvas_to_image(event.x, event.y)

        if self.mode.get() == "wand":
            self.start_stroke() # Push history for undo
            self.apply_wand_flood(ix, iy)
        else:
            self.apply_brush(ix, iy)

    def on_drag(self, event):
        if self.image is None:
            return

        # Update preview while dragging
        self.update_cursor_preview(event)
        
        ix, iy = self.canvas_to_image(event.x, event.y)

        if self.mode.get() in ("pen", "eraser"):
            self.apply_brush(ix, iy)
        elif self.mode.get() == "wand":
             self.apply_wand_flood(ix, iy)

    def on_release(self, event):
        # No specific action needed for pen/eraser as they apply instantly
        pass

    # ================= Wand (Flood Fill) =================

    # ================= Wand (Flood Fill) =================

    def apply_wand_flood(self, x, y):
        # We need to base flood fill on the ORIGINAL image
        if self.original_image is None or self.mosaic_mask is None: return
        
        img_np = np.array(self.original_image)
        h, w = img_np.shape[:2]
        if x < 0 or x >= w or y < 0 or y >= h:
            return
            
        img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
        tol = self.threshold.get()
        
        # Mask for floodFill must be 2 pixels larger than image
        mask = np.zeros((h + 2, w + 2), np.uint8)
        
        # Floodfill flags
        # 4 means 4-connectivity, 8 means 8-connectivity (diagonals included)
        # Changing to 8 to reduce isolated unselected pixels
        connectivity = 8
        flags = connectivity | (255 << 8) | cv2.FLOODFILL_MASK_ONLY | cv2.FLOODFILL_FIXED_RANGE
        
        # If edge detection is enabled, apply it
        if self.use_edge_detection.get():
            # Create gray image for Canny
            gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
            # Use fixed thresholds for now, or could add slider later
            # 100, 200 is a standard starting point
            edges = cv2.Canny(gray, 100, 200)
            
            # Dilate edges slightly to close gaps (make lines "thicker" barrier)
            kernel_edge = np.ones((3,3), np.uint8)
            edges_dilated = cv2.dilate(edges, kernel_edge, iterations=1)
            
            # Embed edges into the mask
            # mask is (h+2, w+2), edges is (h, w)
            # Place edges into mask at [1:-1, 1:-1]
            mask[1:-1, 1:-1][edges_dilated > 0] = 255  # Mark as visited/barrier
            
            # Ensure seed point is not blocked (otherwise floodfill fails immediately)
            if mask[y+1, x+1] == 0:
                 cv2.floodFill(img_bgr, mask, (x, y), (255, 255, 255), 
                               (tol, tol, tol), (tol, tol, tol), flags)
        else:
             # Standard floodfill
             cv2.floodFill(img_bgr, mask, (x, y), (255, 255, 255), 
                           (tol, tol, tol), (tol, tol, tol), flags)
        
        # Extract the relevant part of the mask
        flood_mask = mask[1:-1, 1:-1]
        
        # Morphological Closing to fill small holes ("dots")
        # Kernel size 3x3 or 5x5
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        flood_mask_closed = cv2.morphologyEx(flood_mask, cv2.MORPH_CLOSE, kernel)
        
        flood_mask_bool = flood_mask_closed.astype(bool)
        
        # Apply to mosaic_mask directly
        m_mask = self.mosaic_mask
        if m_mask is not None:
             m_mask[flood_mask_bool] = 255
        
        self.update_view()

    # ================= Pen / Brush =================

    def apply_brush(self, x, y):
        m_mask = self.mosaic_mask
        if m_mask is None: return
        assert m_mask is not None
        
        h, w = m_mask.shape[:2]
        r = self.brush_size.get()
        
        # Draw on the mask
        color = 255 # Mosaic
        if self.mode.get() == "eraser":
             color = 0 # No Mosaic

        # cv2.circle matches the visual appearance better than manual grid check
        cv2.circle(m_mask, (x, y), r, (color,), -1)
        
        self.update_view()

    def start_stroke(self):
        self.push_history()

    # ================= View Composition =================
    
    def generate_mosaic_image(self) -> Optional[np.ndarray]:
        """Generates the mosaic image without any overlays (for saving/display base)."""
        orig_pil = self.original_image
        m_mask = self.mosaic_mask
        
        if orig_pil is None or m_mask is None: return None
        
        # Explicit assertions for type checkers
        assert orig_pil is not None
        assert m_mask is not None
        
        orig_np = np.array(orig_pil)
        h, w = orig_np.shape[:2]
        block = self.get_block_size()
        
        # Generate full mosaic image
        small_h, small_w = max(1, h // block), max(1, w // block)
        small = cv2.resize(orig_np, (small_w, small_h), interpolation=cv2.INTER_LINEAR)
        mosaic_full = cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)
        
        # Composite: Copy Original, then overwrite where mask is 255
        final_img = orig_np.copy()
        mask_bool = m_mask > 0
        final_img[mask_bool] = mosaic_full[mask_bool]
        
        return final_img

    def update_view(self):
        final_img = self.generate_mosaic_image()
        if final_img is None: return
        
        # Add Green Overlay if requested (ONLY for display)
        if self.show_mask.get() and self.mosaic_mask is not None:
             m_mask = self.mosaic_mask
             assert m_mask is not None
             mask_bool = m_mask > 0
             
             # Create green overlay
             # final_img is RGB (from PIL)
             green_overlay = final_img.copy()
             green_overlay[mask_bool] = [0, 255, 0] # Green
             
             blended = cv2.addWeighted(final_img, 0.7, green_overlay, 0.3, 0)
             final_img[mask_bool] = blended[mask_bool]

        
        self.image = Image.fromarray(final_img)
        self.display_image()

    # ================= Load / Save =================

    def open_image(self):
        self.save_current(show_dialog=False) # Auto-save previous
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

        self.load_current_image()

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
                if f.lower().endswith(SUPPORTED_EXT):
                    files.append(os.path.join(root_dir, f))

        files.sort()
        self.image_list = files
        self.current_index = 0
        self.load_current_image()

    def load_current_image(self):
        if not self.image_list: return
        
        path = self.image_list[self.current_index]
        self.current_path = path
        
        img = Image.open(path).convert("RGB")
        self.original_image = img.copy() # Authentic source
        
        # Initialize Mask (Blank)
        w, h = img.size
        # self.mosaic_mask is uint8 (0 or 255)
        self.mosaic_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Clear Undo/Redo
        self.undo_stack.clear()
        
        self.update_view() # Generates self.image
        
        self.zoom = 1.0 
        self.root.title(f"Mosaic Editor - {os.path.basename(path)} ({self.current_index + 1}/{len(self.image_list)})")

    def save_current(self, show_dialog=True):
        # Auto save logic
        if self.current_path is None or self.output_folder is None:
            return

        # Generate clean image for saving (ignore self.image which might have overlay)
        clean_np = self.generate_mosaic_image()
        if clean_np is None:
             return
             
        img_to_save = Image.fromarray(clean_np)
        
        path = self.current_path
        folder = self.output_folder
        
        assert path is not None
        assert folder is not None

        filename = os.path.basename(path)
        save_path = os.path.join(folder, filename)
        img_to_save.save(save_path)
        
        if show_dialog:
            messagebox.showinfo("保存", f"{save_path} に保存しました")

    def display_image(self, preview=None):
        img_src = preview if preview else self.image
        if img_src is None: return

        new_w = int(img_src.width * self.zoom)
        new_h = int(img_src.height * self.zoom)

        resized = img_src.resize((new_w, new_h), Image.NEAREST)
        self.tk_image = ImageTk.PhotoImage(resized)

        self.canvas.delete("all")
        self.canvas.create_image(0, 0, anchor="nw", image=self.tk_image)
        self.canvas.config(scrollregion=(0, 0, new_w, new_h))

    def next_image(self):
        self.save_current(show_dialog=False)
        if hasattr(self, 'image_list') and self.current_index < len(self.image_list) - 1:
            self.current_index += 1
            self.load_current_image()
            
    def prev_image(self):
        self.save_current(show_dialog=False)
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()

if __name__ == "__main__":
    root = tk.Tk()
    root.geometry("1000x800")
    app = MosaicEditor(root)
    root.mainloop()


