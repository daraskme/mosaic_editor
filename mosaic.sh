#!/bin/bash
# カレントディレクトリをこの .sh ファイルの場所にする
cd "$(dirname "$0")"

# 初回起動時に venv 環境を作成
if [ ! -d "venv" ]; then
  python3 -m venv venv
fi

# venv を有効化
source venv/bin/activate

# 依存パッケージをインストール
pip install -r requirements.txt

# 引数（ドラッグ＆ドロップされたファイル or フォルダ）を渡して実行
python3 mosaic.py "$@"
