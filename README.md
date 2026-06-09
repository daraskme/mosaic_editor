# Mosaic Editor

Python と Tkinter で作られた画像/動画モザイク編集ツールです。
手動ブラシ編集に加え、**SAM3 (Meta)** と **LocateAnything-3B (NVIDIA)** による自動検出モザイクに対応しています。

## 主な機能

- **ペン/消しゴム** : ブラシを使って手書き感覚で編集（ブラシサイズ変更可能）
- **矩形選択** : ドラッグで範囲を選択し、その範囲内だけにモザイクを適用
- **魔法の杖** : 色の類似度に基づいて範囲を自動選択（許容値調整可能）
- **モザイク強度** : 自動（審査規定準拠: 長辺/100, 最小4px）または手動指定
- **ズーム/パン** : Ctrl+ホイールズーム（カーソル中心）、中ボタンドラッグでパン
- **D&D対応** : 画像/動画/フォルダをウィンドウにドラッグ＆ドロップして読み込み
- **一括処理** : フォルダを開いて `←` / `→` で順送り、保存は `_mc` フォルダに自動出力（JPEG）
- **自動検出 (画像)** : SAM3 のテキストプロンプトで検出 → 輪郭マスクを直接取得
- **自動検出 (動画)** : SAM3 Video が対象を**全フレームにわたり追跡**してマスク生成
- **音声保持** : 動画書き出し時に ffmpeg があれば元動画の音声を自動結合

## 自動検出

### 使用モデル

| モデル | 役割 | サイズ | ライセンス |
|-------|------|--------|----------|
| [`facebook/sam3`](https://huggingface.co/facebook/sam3) | テキスト→検出+輪郭マスク (画像/動画追跡) | ~3.4GB | SAM License (商用可) |
| [`nvidia/LocateAnything-3B`](https://huggingface.co/nvidia/LocateAnything-3B) | 検出 (画像+テキスト→bbox)。条件付き概念に強い | ~8GB | **非商用研究目的のみ** |

初回実行時にモデルが Hugging Face Hub からダウンロードされ、`~/.cache/huggingface/` にキャッシュされます。

> **重要**: `facebook/sam3` は gated モデルです。初回利用前に
> 1. https://huggingface.co/facebook/sam3 を開いて利用規約に同意
> 2. `hf auth login` (または `huggingface-cli login`) でログイン
> しておいてください。

### 検出エンジン

| エンジン | 構成 | 特徴 |
|---------|------|------|
| **SAM3 のみ** (推奨) | テキスト→マスク直接 | 高速。1モデルで検出+輪郭。スコア付き |
| **LocateAnything + SAM3** | VLM で bbox → SAM3 Tracker で輪郭化 | 「挿入されたアナル」のような**条件付き・文章的概念**に強い |
| **両方併用** | 上記2つを実行して統合 | 取りこぼし最小。最も遅い |

### 検出カテゴリ

デフォルトの検出対象（ダイアログで変更可能）:

| カテゴリ | デフォルト | 備考 |
|---------|-----------|------|
| 男性器 | ✅ | |
| 女性器 | ✅ | |
| 睾丸 | ✅ | |
| 結合部 (挿入) | ✅ | 挿入中の性器結合部 |
| アナル (挿入時のみ) | ✅ | **挿入されている場合のみ**検出・モザイク |
| アナル (常時) | ❌ | |
| 乳首 | ❌ | モザイク不要のためデフォルトOFF |

「追加クラス」に英語の名詞句を入力すれば任意の対象も検出できます。

### 動画の自動モザイク

動画を開いて「自動検出」→「動画全体」を選ぶと、SAM3 Video が対象を検出して
**全フレームにわたって追跡**します。旧版のフレーム毎検出と違い、
時間方向に一貫したマスクが高速に得られます。

### 詳細設定

- **検出しきい値** : 低いほど拾いやすい（誤検出も増える）。既定 0.4
- **マスク拡張** : 検出輪郭の外側に余裕を持たせる px 数。既定 4px
- **LocateAnything 検出強度** : fast / hybrid (推奨) / slow

## ハードウェア要件

| 環境 | 推奨 |
|------|------|
| NVIDIA GPU (VRAM 8GB+) | SAM3 のみ → 快適 |
| NVIDIA GPU (VRAM 16GB+) | 全エンジン快適 |
| CPU only | 動作はするが 1枚あたり数十秒〜 |

## インストールと実行

### 自動セットアップ（推奨）

**Windows** — `mosaic.bat` をダブルクリック:

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
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128  # GPU 環境
pip install -r requirements.txt
python mosaic.py
```

- Python 3.10+（SAM3 利用時は transformers>=5.0 が必要）
- 動画の音声保持には `ffmpeg` がパスにあること（無ければ無音で保存）

## 操作まとめ

| 操作 | 方法 |
|------|------|
| 前後の画像/フレーム | ホイール / `←` `→` |
| ズームイン/アウト | Ctrl+ホイール / 拡大・縮小ボタン |
| パン（画像移動） | 中ボタンドラッグ |
| undo / redo | `Ctrl+Z` / `Ctrl+Y` |
| 保存 | メニュー「ファイル」>「保存」 / 次の画像へ移動時に自動保存 |

## 出力ファイル

- 保存先: 入力フォルダと同じ階層の `_mc` サフィックス付きフォルダ
- 画像: JPEG（品質95）、動画: MP4 (H.264 + AAC)
- 編集マスク: `_mc/masks/*.mask.npz` に保存され、再編集時に復元されます

## 技術メモ

- LocateAnything-3B の remote code は transformers 4.57.1 向けのため、
  v5 で動かす互換パッチを `mosaic_editor/detect/locate_anything.py` で適用しています
- 検出はバックエンド非依存の `Detection` 型に正規化され、
  `DetectionPipeline.combine_masks()` で1枚のマスクに統合されます
