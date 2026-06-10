# Mosaic Editor

Python と Tkinter で作られた画像/動画モザイク編集ツールです。
手動ブラシ編集に加え、**AnimeCensor (deepghs) + SAM2 (Meta)** による自動検出モザイクに対応しています。

## 主な機能

- **ペン/消しゴム** : ブラシを使って手書き感覚で編集（ブラシサイズ変更可能）
- **矩形選択** : ドラッグで範囲を選択し、その範囲内だけにモザイクを適用
- **魔法の杖** : 色の類似度に基づいて範囲を自動選択（許容値調整可能）
- **モザイク強度** : 自動（審査規定準拠: 長辺/100, 最小4px）または手動指定
- **ズーム/パン** : Ctrl+ホイールズーム（カーソル中心）、中ボタンドラッグでパン
- **D&D対応** : 画像/動画/フォルダをウィンドウにドラッグ＆ドロップして読み込み
- **一括処理** : フォルダを開いて `←` / `→` で順送り、保存は `_mc` フォルダに自動出力（JPEG）
- **自動検出 (画像)** : イラスト特化の YOLOv8 で検出 → SAM2 で輪郭マスク化
- **自動検出 (動画)** : 検出した対象を SAM2 Video が**全フレームにわたり追跡**
- **音声保持** : 動画書き出し時に ffmpeg があれば元動画の音声を自動結合

## 自動検出

### 使用モデル

| モデル | 役割 | サイズ | ライセンス |
|-------|------|--------|----------|
| [`deepghs/anime_censor_detection`](https://huggingface.co/deepghs/anime_censor_detection) | イラスト/アニメ絵の検出 (YOLOv8, ONNX) | ~50MB | Apache 2.0 |
| [`facebook/sam2.1-hiera-large`](https://huggingface.co/facebook/sam2.1-hiera-large) | bbox→輪郭マスク化・動画追跡 | ~900MB | Apache 2.0 |

初回実行時にモデルが Hugging Face Hub からダウンロードされ、`~/.cache/huggingface/` にキャッシュされます。
**HF アカウントやログインは不要**です。

検出は booru 系大規模アノテーションで学習されたアニメ絵専用 YOLOv8 で、
1枚あたり数十ms と高速。デフォルメされた描写にも強いのが特徴です。

### 検出カテゴリ

| カテゴリ | デフォルト | 備考 |
|---------|-----------|------|
| 男性器 | ✅ | 挿入中の露出部分も検出されます |
| 女性器 | ✅ | |
| 乳首 | ❌ | モザイク不要のためデフォルトOFF |

### 動画の自動モザイク

動画を開いて「自動検出」→「動画全体」を選ぶと、150フレームごとのキーフレームで
検出し、SAM2.1 Video が各対象をチャンク内全フレームに伝播・追跡します。
途中から映り込む対象もチャンク境界の再検出で拾えます。

### 詳細設定

- **検出しきい値** : 低いほど拾いやすい（誤検出も増える）。既定 0.3。
  拾い漏れがある場合は 0.15〜0.25 に下げる
- **マスク拡張** : 検出輪郭の外側に余裕を持たせる px 数。既定 4px

## ハードウェア要件

| 環境 | 目安 |
|------|------|
| NVIDIA GPU (VRAM 4GB+) | 快適 (検出は CPU でも数十ms) |
| CPU only | 検出は高速。SAM2 の輪郭化・動画追跡は遅め |

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

- Python 3.10+
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

- 検出はバックエンド非依存の `Detection` 型に正規化され、
  `DetectionPipeline.combine_masks()` で1枚のマスクに統合されます
- SAM2 Video は bf16 だと長尺伝播で数値的に不安定なため fp32 固定です
