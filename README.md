# Mosaic Editor

PythonとTkinterで作られた画像モザイク編集ツールです。
手動ブラシ編集に加え、**LocateAnything-3B (NVIDIA) による自動検出モザイク**に対応しています。

## 主な機能

- **ペン/消しゴム** : ブラシを使って手書き感覚で編集（ブラシサイズ変更可能）
- **矩形選択** : ドラッグで範囲を選択し、その範囲内だけにモザイクを適用
- **魔法の杖** : 色の類似度に基づいて範囲を自動選択（許容値調整可能・エッジ停止オプションあり）
- **モザイク強度** : モザイクの粗さをスピンボックスで調整
- **ズーム/パン** : マウスホイールズーム（カーソル位置を中心）、中ボタンドラッグでパン
- **D&D対応** : 画像ファイルまたはフォルダをウィンドウにドラッグ＆ドロップして読み込み
- **一括処理** : フォルダを開いて `←` / `→` で順送り、保存は `_mc` フォルダに自動出力（JPEG）
- **自動検出** : LocateAnything-3B で部位を検出 → SAM で輪郭マスクに再セグメント

## 自動検出

### 使用モデル

| モデル | 役割 | サイズ | ライセンス |
|-------|------|--------|----------|
| [`nvidia/LocateAnything-3B`](https://huggingface.co/nvidia/LocateAnything-3B) | 検出 (画像 + テキスト → bbox) | 約 6GB | **非商用研究目的のみ** |
| [`facebook/sam-vit-base`](https://huggingface.co/facebook/sam-vit-base) | bbox → 輪郭マスクへ再セグメント | 約 360MB | Apache 2.0 |

初回実行時にモデルが Hugging Face Hub からダウンロードされ、
`~/.cache/huggingface/` にキャッシュされます。

### ハードウェア要件

| 環境 | 推論時間 (画像1枚あたり) |
|------|-------------------------|
| **NVIDIA GPU (Ampere 以降, VRAM 16GB+)** | 数秒 |
| CPU only / Apple Silicon (MPS) | 数十秒〜数分 |

> LocateAnything-3B は BF16 推論を前提に設計されているため、
> **NVIDIA GPU + CUDA** での実行が強く推奨されます。
> CPU/MPS でも動作するよう自動フォールバックしますが、実用速度は出ません。

### 検出可能なクラス

LocateAnything-3B は汎用 Vision-Language モデルなので、テキストプロンプトで任意の対象を指定できます。
デフォルトでは以下のクラスを検出対象として候補に出します:

- `penis`
- `pussy`
- `vagina`
- `anus`
- `nipples`
- `testicles`

検出ダイアログでチェックボックスにより取捨選択、または「追加クラス」入力で任意のクラスを追加できます。

### 自動検出の手順

1. 画像 (または動画) を開く
2. ツールバーの **「自動検出」** ボタンをクリック
3. 検出対象クラスを選び、検出強度 (fast/hybrid/slow) を選択 → **「検出開始」**
4. 検出結果の一覧から取り込みたい候補をチェック
5. **「選択範囲に追加」** でモザイクマスクに反映
6. ペン・魔法の杖・消しゴムで微調整 → 保存

> SAM 再セグメントを有効にすると `[seg]` (輪郭マスク)、無効にすると `[box]` (矩形塗りつぶし) で適用されます。

### 検出強度 (generation_mode)

| モード | 速度 | 精度 | 用途 |
|--------|------|------|------|
| `fast` | 速い | 標準 | シンプルなシーン |
| `hybrid` | 中 | 高 | **推奨** |
| `slow` | 遅い | 最高 | 複雑/曖昧なシーン |

### 依存パッケージの自動インストール

初回実行時に `transformers` / `torch` / `peft` / `accelerate` が未インストールであれば、
インストール確認ダイアログが表示されます。事前に手動でインストールしたい場合:

```bash
pip install -r requirements.txt
```

## 必要要件

- Python 3.10+
- 依存ライブラリ（`requirements.txt`）:
  - `Pillow`
  - `numpy`
  - `opencv-python`
  - `torch` / `torchvision` （自動検出を使う場合）
  - `transformers>=4.57.1` （自動検出を使う場合）
  - `peft`, `accelerate` （自動検出を使う場合）
  - `tkinterdnd2`（D&D対応、起動時に自動インストール）

## インストールと実行

### 自動セットアップ（推奨）

**Windows** - `mosaic.bat` をダブルクリック、またはファイル/フォルダをD&D:

```
mosaic.bat
```

**Linux / Mac**:
```bash
chmod +x mosaic.sh
./mosaic.sh
```

### 手動セットアップ

```bash
python -m venv venv
# Windows: venv\Scripts\activate / Mac・Linux: source venv/bin/activate
pip install -r requirements.txt
python mosaic.py
```

## 操作まとめ

| 操作 | 方法 |
|------|------|
| ズームイン/アウト | マウスホイール / `+` `-` ボタン |
| パン（画像移動） | 中ボタンドラッグ |
| undo / redo | `Ctrl+Z` / `Ctrl+Y` |
| 矩形選択解除 | 「解除」ボタン |
| 保存 | メニュー「ファイル」>「保存」 / 次の画像へ移動時に自動保存 |

## 出力ファイル

保存先は入力フォルダと同じ階層に `_mc` サフィックスのフォルダが自動生成されます。
保存形式は **JPEG（品質95）** です。
