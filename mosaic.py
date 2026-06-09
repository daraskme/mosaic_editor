"""Mosaic Editor 起動エントリポイント."""
import sys
import tkinter as tk


def _ensure_tkinterdnd2():
    """tkinterdnd2 がなければ自動インストールしてから import する."""
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
            mod = importlib.import_module("tkinterdnd2")
            print("[MosaicEditor] tkinterdnd2インストール完了")
            return mod
        except Exception as e:
            print(f"[MosaicEditor] tkinterdnd2インストール失敗: {e}")
            return None


def main():
    from mosaic_editor.ui.editor import MosaicEditor

    dnd = _ensure_tkinterdnd2()
    root = dnd.TkinterDnD.Tk() if dnd is not None else tk.Tk()
    root.geometry("1100x850")
    MosaicEditor(root)
    root.mainloop()


if __name__ == "__main__":
    main()
