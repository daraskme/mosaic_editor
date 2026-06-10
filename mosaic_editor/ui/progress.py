"""進捗ダイアログ・依存パッケージ自動インストール UI."""
from __future__ import annotations

import queue
import subprocess
import sys
import threading
import tkinter as tk
from tkinter import messagebox, ttk
from typing import Callable, List, Optional, Tuple


def safe_grab(win: tk.Toplevel) -> None:
    """ウィンドウが表示されてから grab_set する (未表示だと TclError になる)."""
    def _try(attempts: int = 20):
        try:
            win.grab_set()
        except tk.TclError:
            if attempts > 0 and win.winfo_exists():
                win.after(50, lambda: _try(attempts - 1))

    try:
        win.update_idletasks()
    except tk.TclError:
        return
    _try()


def show_progress_window(root, title: str, msg: str,
                         with_progress_bar: bool = False,
                         maximum: int = 100,
                         cancelable: bool = False,
                         on_cancel: Optional[Callable] = None
                         ) -> Tuple[tk.Toplevel, tk.Label, Optional[ttk.Progressbar]]:
    """待機ダイアログを生成して (window, status_label, bar) を返す."""
    win = tk.Toplevel(root)
    win.title(title)
    win.geometry("440x170")
    win.resizable(False, False)
    safe_grab(win)

    status_label = tk.Label(win, text=msg, pady=8, wraplength=420,
                            justify="left", font=("", 9))
    status_label.pack(fill="x", padx=10)

    if with_progress_bar:
        bar = ttk.Progressbar(win, maximum=max(1, maximum), length=400)
        bar.pack(padx=20, pady=4)
    else:
        bar = ttk.Progressbar(win, mode="indeterminate", length=400)
        bar.pack(padx=20, pady=4)
        bar.start(10)

    if cancelable:
        tk.Button(win, text="キャンセル",
                  command=on_cancel if on_cancel else win.destroy,
                  relief="flat", padx=8).pack(pady=4)

    return win, status_label, bar


REQUIRED_PACKAGES = ("torch", "transformers", "accelerate")


def check_missing_packages(packages=REQUIRED_PACKAGES) -> List[str]:
    missing: List[str] = []
    for pkg in packages:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    return missing


def ensure_deps(root, on_ready: Callable, packages=REQUIRED_PACKAGES) -> None:
    """依存が揃っていれば on_ready() を即実行。なければ自動インストールを提案."""
    missing = check_missing_packages(packages)
    if not missing:
        on_ready()
        return
    if not messagebox.askyesno(
        "依存パッケージのインストール",
        "自動検出には以下のパッケージが必要です:\n\n"
        f"  {', '.join(missing)}\n\n"
        "自動的にインストールしますか？\n"
        "（torch は大きいため初回は数GBダウンロードされます）"
    ):
        return
    install_packages_then(root, missing, on_ready)


def install_packages_then(root, packages: List[str], on_ready: Callable) -> None:
    """pip インストールをログ表示付きで実行し、成功したら on_ready() を呼ぶ."""
    win = tk.Toplevel(root)
    win.title("依存パッケージをインストール中")
    win.geometry("560x320")
    safe_grab(win)

    tk.Label(win, text=f"インストール中: {', '.join(packages)}",
             font=("", 10, "bold"), pady=6).pack()

    bar = ttk.Progressbar(win, mode="indeterminate", length=520)
    bar.pack(padx=10)
    bar.start(12)

    frame = tk.Frame(win)
    frame.pack(fill="both", expand=True, padx=10, pady=6)
    log_text = tk.Text(frame, height=14, wrap="word", state="disabled",
                       bg="#1e1e1e", fg="#cccccc", font=("Consolas", 9))
    log_text.pack(side="left", fill="both", expand=True)
    sb = ttk.Scrollbar(frame, command=log_text.yview)
    sb.pack(side="right", fill="y")
    log_text.config(yscrollcommand=sb.set)

    log_queue: "queue.Queue[str]" = queue.Queue()

    def append_log(line: str):
        log_text.config(state="normal")
        log_text.insert("end", line)
        log_text.see("end")
        log_text.config(state="disabled")

    def poll_queue():
        try:
            while True:
                append_log(log_queue.get_nowait())
        except queue.Empty:
            pass
        if win.winfo_exists():
            win.after(120, poll_queue)

    def do_install():
        args = [sys.executable, "-m", "pip", "install", "--upgrade",
                "--no-cache-dir"] + list(packages)
        log_queue.put(f"$ {' '.join(args)}\n")
        proc = subprocess.Popen(
            args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            text=True, encoding="utf-8", errors="replace",
        )
        for line in proc.stdout:  # type: ignore
            log_queue.put(line)
        proc.wait()
        if proc.returncode == 0:
            log_queue.put("\n✅ インストール完了！\n")
            root.after(0, lambda: (win.destroy(), on_ready()))
        else:
            log_queue.put("\n❌ インストール失敗\n")

            def fail():
                win.title("インストール失敗")
                bar.stop()
                tk.Label(win,
                         text=("❌ インストールに失敗しました。手動で実行してください:\n"
                               f"  pip install {' '.join(packages)}"),
                         fg="red", justify="left").pack(pady=4)
            root.after(0, fail)

    win.after(120, poll_queue)
    threading.Thread(target=do_install, daemon=True).start()
