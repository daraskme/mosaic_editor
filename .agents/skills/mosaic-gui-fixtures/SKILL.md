---
name: test-mosaic-editor-windows-exe
description: Verify the Windows Tkinter lite executable and optional detection runtime through the real desktop without reinstalling heavy dependencies.
---

# Windows executable GUI testing

## Devin Secrets Needed
None for local mosaic detection, runtime confirmation cancellation, or cached AnimeCensor detection. First-time model/runtime downloads require internet access.

## Desktop and launch
- Use the requested prebuilt `dist/MosaicEditor.exe`, not the Python entry point, when testing frozen-executable dependency handling.
- Existing blueprint installs pyautogui and pygetwindow in `venv`. If computer clicks do not affect Tkinter, drive the actual desktop using those modules via `venv/Scripts/python.exe`.
- Set `PYTHONIOENCODING=utf-8` when printing Japanese window titles; default Windows console encoding may reject them.
- Query `pyautogui.size()`; screenshot-tool coordinates may be scaled differently. On a 1280x720 desktop, convert from a 1024x768 tool screenshot independently on each axis.
- Activate the specific main/modal window with pygetwindow before interacting. Maximize the main window before recording. Allow the one-file executable to unpack and create its window before querying it.
- Use backslash Windows paths in native file dialogs. A portable Python expression is `path.replace('/', chr(92))`; it also avoids shell/Python backslash-escaping problems.

## Runtime states
- Optional ML dependencies live under `%LOCALAPPDATA%\MosaicEditor\pydeps`.
- Always close all exe instances before temporarily renaming pydeps; loaded modules otherwise hide missing-runtime behavior.
- Use a unique backup name and never overwrite an existing backup. Record file-relative paths and sizes, rename rather than copy/delete, restore after the test, and compare the manifest.
- With only モザイク checked, expect no ML setup prompt. With 男性器 only and installed runtime, expect detection without setup.
- With pydeps absent and a fresh exe, expect `自動検出のセットアップ` mentioning ~1.5GB. Click No to test cancellation without reinstalling. Buttons may be English Yes/No despite Japanese app text.
- Full downloads are large; do not repeat them when cancellation/reuse is the requested scope.

## Fixtures and UI
- Use isolated test folders: opening/navigating/closing images can autosave JPEGs and masks to a sibling `_mc` folder.
- Reliable mosaic fixture: seeded RGB noise at 32x32, bicubic upscale to 512x512, average each 16px cell inside (128,96)-(384,352), save PNG. Keep the unpixelated base for negative checks.
- `ファイル → 画像を開く`, then toolbar `自動検出`; uncheck 男性器/女性器/乳首, check モザイク; `検出開始`.
- Review is a coordinate list. `選択範囲に追加` applies a green overlay; expect an application-complete dialog and matching rectangle.
- `ファイル → フォルダを開く` automatically opens batch configuration. Use one mosaic and one plain PNG; expect one of two images masked.
- Keep negative/setup dialogs and applied overlays as screenshots. Record visible GUI tests, not filesystem backup pauses.
