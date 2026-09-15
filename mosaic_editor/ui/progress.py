"""進捗ダイアログ・依存パッケージ自動インストール UI."""
from __future__ import annotations

import ctypes
import importlib
import os
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


def close_window(win: tk.Toplevel) -> None:
    """破棄済みでもエラーにならないようにウィンドウを閉じる."""
    try:
        win.destroy()
    except tk.TclError:
        pass


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


REQUIRED_PACKAGES = ("torch", "transformers", "imgutils")


def check_missing_packages(packages=REQUIRED_PACKAGES) -> List[str]:
    missing: List[str] = []
    for pkg in packages:
        try:
            __import__(pkg)
        except Exception:  # DLL 読み込み失敗 (OSError) も未使用扱いにする
            missing.append(pkg)
    return missing


def _local_app_dir() -> str:
    """exe 版で実行環境を置く永続ディレクトリ (%LOCALAPPDATA%\\MosaicEditor)."""
    base = os.environ.get("LOCALAPPDATA") or os.path.join(
        os.path.expanduser("~"), "AppData", "Local")
    return os.path.join(base, "MosaicEditor")


def _deps_dir() -> str:
    return os.path.join(_local_app_dir(), "pydeps")


def _use_deps_dir() -> None:
    """前回インストール済みの実行環境があれば sys.path に追加する."""
    d = _deps_dir()
    if os.path.isdir(d) and d not in sys.path:
        # 先頭に挿入: exe 同梱の部分的なパッケージ (packaging 等) より
        # pydeps 側の一貫したセットを優先させる
        sys.path.insert(0, d)
        importlib.invalidate_caches()


def ensure_deps(root, on_ready: Callable, packages=REQUIRED_PACKAGES) -> None:
    """依存が揃っていれば on_ready() を即実行。なければ自動インストールを提案."""
    _use_deps_dir()
    missing = check_missing_packages(packages)
    if not missing:
        on_ready()
        return
    if getattr(sys, "frozen", False):
        if not messagebox.askyesno(
            "自動検出のセットアップ",
            "自動検出の実行環境 (~1.5GB) が見つかりません。\n\n"
            "今すぐダウンロードしてインストールしますか？\n"
            "（初回のみ・インターネット接続が必要です）"
        ):
            return
        install_frozen_deps_then(root, on_ready)
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
    args = [sys.executable, "-m", "pip", "install", "--upgrade",
            "--no-cache-dir"] + list(packages)
    _run_steps_window(
        root,
        title="依存パッケージをインストール中",
        header=f"インストール中: {', '.join(packages)}",
        steps=[(f"pip install {' '.join(packages)}", args, False)],
        on_success=on_ready,
        fail_hint=f"pip install {' '.join(packages)}",
    )


_UV_ZIP_URL = ("https://github.com/astral-sh/uv/releases/latest/download/"
               "uv-x86_64-pc-windows-msvc.zip")
_VCREDIST_URL = "https://aka.ms/vs/17/release/vc_redist.x64.exe"


def _probe_torch(deps_dir: str) -> bool:
    """deps_dir の torch DLL がロードできるか試す (VC++ ランタイム不足の検出用)."""
    c10 = os.path.join(deps_dir, "torch", "lib", "c10.dll")
    if not os.path.exists(c10):
        return False
    try:
        ctypes.CDLL(c10)
        return True
    except OSError:
        return False


def install_frozen_deps_then(root, on_ready: Callable) -> None:
    """exe 版: uv で実行環境を pydeps にインストールしてから on_ready() を呼ぶ."""
    base = _local_app_dir()
    uv_dir = os.path.join(base, "uv")
    uv_exe = os.path.join(uv_dir, "uv.exe")
    uv_zip = uv_exe + ".zip"
    redist = os.path.join(base, "vc_redist.x64.exe")
    deps = _deps_dir()
    py_ver = f"{sys.version_info.major}.{sys.version_info.minor}"

    try:
        os.makedirs(uv_dir, exist_ok=True)
        os.makedirs(deps, exist_ok=True)
    except OSError as e:
        messagebox.showerror(
            "セットアップ失敗",
            f"インストール先を作成できませんでした:\n{e}")
        return

    steps: List[Tuple[str, List[str], bool]] = []
    if not os.path.exists(uv_exe):
        steps += [
            ("uv のダウンロード (~20MB)",
             ["curl.exe", "-fsSL", "-o", uv_zip, _UV_ZIP_URL], False),
            ("uv の展開",
             ["powershell", "-NoProfile", "-Command",
              f"Expand-Archive -Force '{uv_zip}' '{uv_dir}'"], False),
        ]
    steps += [
        ("Visual C++ ランタイムのダウンロード (~25MB)",
         ["curl.exe", "-fsSL", "-o", redist, _VCREDIST_URL], False),
        ("Visual C++ ランタイムのインストール (要管理者権限・失敗時は後で手動)",
         [redist, "/install", "/quiet", "/norestart"], True),
        ("torch のダウンロード/インストール (~800MB)",
         [uv_exe, "pip", "install", "--python", py_ver, "--target", deps,
          "--index", "https://download.pytorch.org/whl/cpu",
          "torch", "torchvision"], False),
        ("残りのパッケージのインストール (~500MB)",
         [uv_exe, "pip", "install", "--python", py_ver, "--target", deps,
          "transformers", "dghs-imgutils", "onnxruntime"], False),
    ]

    def _after_install() -> None:
        _use_deps_dir()
        if not _probe_torch(deps):
            messagebox.showerror(
                "実行環境の読み込み失敗",
                "インストールは完了しましたが、torch の DLL を読み込めませんでした。\n"
                "最新の Visual C++ 再頒布パッケージが必要です。\n\n"
                "管理者権限で以下を実行してからアプリを再起動してください:\n"
                f"  {_VCREDIST_URL}")
            return
        on_ready()

    _run_steps_window(
        root,
        title="自動検出の実行環境をダウンロード中",
        header="自動検出の実行環境をセットアップしています (~1.5GB)",
        steps=steps,
        on_success=_after_install,
        fail_hint="画面上部の「自動検出」から再度セットアップを実行",
    )


def _run_steps_window(root, title: str, header: str,
                      steps: List[Tuple[str, List[str], bool]],
                      on_success: Callable,
                      fail_hint: str) -> None:
    """ログ表示付きでコマンド列を順に実行し、全成功したら on_success() を呼ぶ."""
    win = tk.Toplevel(root)
    win.title(title)
    win.geometry("560x320")
    safe_grab(win)

    tk.Label(win, text=header, font=("", 10, "bold"), pady=6).pack()

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
        ok = True
        for label, args, ignore_rc in steps:
            log_queue.put(f"\n== {label} ==\n$ {' '.join(args)}\n")
            try:
                proc = subprocess.Popen(
                    args, stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True, encoding="utf-8", errors="replace",
                    creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                )
            except OSError as e:
                log_queue.put(f"実行に失敗しました: {e}\n")
                if not ignore_rc:
                    ok = False
                    break
                continue
            for line in proc.stdout or []:
                log_queue.put(line)
            proc.wait()
            if proc.returncode != 0:
                log_queue.put(f"(exit code {proc.returncode})\n")
                if not ignore_rc:
                    ok = False
                    break
        if ok:
            log_queue.put("\n✅ セットアップ完了！\n")
            root.after(0, lambda: (close_window(win), on_success()))
        else:
            log_queue.put("\n❌ セットアップ失敗\n")

            def fail():
                win.title("セットアップ失敗")
                bar.stop()
                tk.Label(win,
                         text=("❌ セットアップに失敗しました。\n"
                               f"次を確認して再試行してください: {fail_hint}"),
                         fg="red", justify="left").pack(pady=4)
            root.after(0, fail)

    win.after(120, poll_queue)
    threading.Thread(target=do_install, daemon=True).start()
