import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import numpy as np
import os

SUPPORTED_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp")


class MosaicEditor:
    def __init__(self, root):
        self.root = root
        self.root.title("Mosaic Editor")

        self.image = None
        self.original_image = None
        self.tk_image = None

        self.image_list = []
        self.current_index = 0
        self.current_path = None
        self.output_folder = None

        self.mode = tk.StringVar(value="rect")
        self.threshold = tk.IntVar(value=40)
        self.brush_size = tk.IntVar(value=20)

        self.undo_stack = []
        self.redo_stack = []

        self.zoom = 1.0
        self.pan_x = 0
        self.pan_y = 0

        self.start_x = None
        self.start_y = None

        self.wand_mask = None

        self.build_menu()
        self.build_ui()

    # ================= 座標変換 =================

    def canvas_to_image(self, x, y):
        ix = int((x - self.pan_x) / self.zoom)
        iy = int((y - self.pan_y) / self.zoom)
        return ix, iy

    # ================= Utility =================

    def get_block_size(self):
        w, h = self.image.size
        long_side = max(w, h)
        if long_side >= 400:
            return max(4, long_side // 100)
        return 4

    def push_history(self):
        self.undo_stack.append(self.image.copy())
        if len(self.undo_stack) > 20:
            self.undo_stack.pop(0)
        self.redo_stack.clear()

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
        top.pack(fill="x")

        tk.Button(top, text="←", command=self.prev_image).pack(side="left")
        tk.Button(top, text="→", command=self.next_image).pack(side="left")
        tk.Button(top, text="＋", command=self.zoom_in).pack(side="left")
        tk.Button(top, text="−", command=self.zoom_out).pack(side="left")
        tk.Button(top, text="魔法の杖適用", command=self.confirm_wand).pack(side="left")

        mode_frame = tk.Frame(self.root)
        mode_frame.pack(fill="x")

        for text, val in [("矩形", "rect"),
                          ("ペン", "pen"),
                          ("魔法の杖", "wand"),
                          ("消しゴム", "eraser")]:
            tk.Radiobutton(mode_frame, text=text,
                           variable=self.mode, value=val).pack(side="left")

        tk.Label(self.root, text="許容値").pack()
        tk.Scale(self.root, from_=0, to=255,
                 orient="horizontal",
                 variable=self.threshold).pack(fill="x")

        tk.Label(self.root, text="ブラシサイズ").pack()
        tk.Scale(self.root, from_=5, to=100,
                 orient="horizontal",
                 variable=self.brush_size).pack(fill="x")

        self.canvas = tk.Canvas(self.root, cursor="cross")
        self.canvas.pack(fill="both", expand=True)

        self.canvas.bind("<ButtonPress-1>", self.on_click)
        self.canvas.bind("<B1-Motion>", self.on_drag)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        self.canvas.bind("<Button-4>", lambda e: self.next_image())
        self.canvas.bind("<Button-5>", lambda e: self.prev_image())

        self.canvas.bind("<ButtonPress-2>", self.start_pan)
        self.canvas.bind("<B2-Motion>", self.do_pan)

    # ================= Zoom =================

    def zoom_in(self):
        if not self.image:
            return
        self.zoom *= 1.25
        self.display_image()

    def zoom_out(self):
        if not self.image:
            return
        self.zoom /= 1.25
        self.display_image()

    # ================= Pan =================

    def start_pan(self, event):
        self.canvas.scan_mark(event.x, event.y)

    def do_pan(self, event):
        self.canvas.scan_dragto(event.x, event.y, gain=1)

    # ================= Mouse =================

    def on_click(self, event):
        if not self.image:
            return

        ix, iy = self.canvas_to_image(event.x, event.y)

        if self.mode.get() == "wand":
            self.preview_wand(ix, iy)
        else:
            self.start_x = ix
            self.start_y = iy

    def on_drag(self, event):
        if not self.image:
            return

        ix, iy = self.canvas_to_image(event.x, event.y)

        if self.mode.get() == "rect":
            self.canvas.delete("preview")
            self.canvas.create_rectangle(
                event.x, event.y,
                self.start_x * self.zoom,
                self.start_y * self.zoom,
                outline="green", width=2, tag="preview"
            )
        elif self.mode.get() in ("pen", "eraser"):
            self.apply_brush(ix, iy)

    def on_release(self, event):
        if not self.image:
            return

        ix, iy = self.canvas_to_image(event.x, event.y)

        if self.mode.get() == "rect":
            self.push_history()
            self.apply_rect(self.start_x, self.start_y, ix, iy)
            self.canvas.delete("preview")

    # ================= Wand =================

    def preview_wand(self, x, y):
        img = np.array(self.image)
        if not (0 <= x < img.shape[1] and 0 <= y < img.shape[0]):
            return

        target = img[y, x]
        tol = self.threshold.get()

        diff = np.sum(np.abs(img - target), axis=2)
        self.wand_mask = diff <= tol

        overlay = img.copy()
        overlay[self.wand_mask] = [0, 255, 0]
        self.display_image(Image.fromarray(overlay))

    def confirm_wand(self):
        if self.wand_mask is None:
            return

        self.push_history()
        img = np.array(self.image)
        block = self.get_block_size()
        img[self.wand_mask] = (img[self.wand_mask] // block) * block
        self.image = Image.fromarray(img)
        self.wand_mask = None
        self.display_image()

    # ================= Mosaic =================

    def apply_rect(self, x1, y1, x2, y2):
        img = np.array(self.image)
        x1, x2 = sorted([x1, x2])
        y1, y2 = sorted([y1, y2])
        block = self.get_block_size()

        for y in range(y1, y2, block):
            for x in range(x1, x2, block):
                img[y:y+block, x:x+block] = img[y, x]

        self.image = Image.fromarray(img)
        self.display_image()

    def apply_brush(self, x, y):
        self.push_history()
        img = np.array(self.image)
        orig = np.array(self.original_image)
        r = self.brush_size.get()
        block = self.get_block_size()

        y1 = max(0, y-r)
        y2 = min(img.shape[0], y+r)
        x1 = max(0, x-r)
        x2 = min(img.shape[1], x+r)

        if self.mode.get() == "eraser":
            img[y1:y2, x1:x2] = orig[y1:y2, x1:x2]
        else:
            region = img[y1:y2, x1:x2]
            region = (region // block) * block
            img[y1:y2, x1:x2] = region

        self.image = Image.fromarray(img)
        self.display_image()

    # ================= Load / Save =================

    def open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.webp")]
        )
        if not path:
            return

        self.image_list = [path]
        self.current_index = 0
        self.output_folder = os.path.dirname(path) + "_mc"
        os.makedirs(self.output_folder, exist_ok=True)
        self.load_current_image()

    def open_folder(self):
        folder = filedialog.askdirectory()
        if not folder:
            return

        self.output_folder = folder + "_mc"
        os.makedirs(self.output_folder, exist_ok=True)

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
        self.current_path = self.image_list[self.current_index]
        self.image = Image.open(self.current_path).convert("RGB")
        self.original_image = self.image.copy()
        self.zoom = 1.0
        self.display_image()

    def save_current(self):
        filename = os.path.basename(self.current_path)
        save_path = os.path.join(self.output_folder, filename)
        self.image.save(save_path)
        messagebox.showinfo("保存", f"{save_path} に保存しました")

    # ================= Display =================

    def display_image(self, preview=None):
        img = preview if preview else self.image
        if img is None:
            return

        img = img.resize(
            (int(img.width * self.zoom),
             int(img.height * self.zoom)),
            Image.NEAREST
        )

        self.tk_image = ImageTk.PhotoImage(img)
        self.canvas.delete("all")
        self.canvas.create_image(self.pan_x, self.pan_y,
                                 anchor="nw", image=self.tk_image)

    # ================= Navigation =================

    def next_image(self):
        if self.current_index < len(self.image_list) - 1:
            self.current_index += 1
            self.load_current_image()

    def prev_image(self):
        if self.current_index > 0:
            self.current_index -= 1
            self.load_current_image()


if __name__ == "__main__":
    root = tk.Tk()
    app = MosaicEditor(root)
    root.mainloop()

