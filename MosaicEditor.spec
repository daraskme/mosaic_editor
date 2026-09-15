# -*- mode: python ; coding: utf-8 -*-
import importlib.util
import sys

from PyInstaller.utils.hooks import collect_all, collect_submodules

datas = []
binaries = []
hiddenimports = []
tmp_ret = collect_all('tkinterdnd2')
datas += tmp_ret[0]; binaries += tmp_ret[1]; hiddenimports += tmp_ret[2]

# 実行環境 (torch/transformers 等) は pydeps に後から追加されるため、
# それらが必要とする標準ライブラリはアプリ側で全量同梱しておく。
_STDLIB_SKIP = {'antigravity', 'ensurepip', 'idlelib', 'lib2to3', 'test',
                'turtledemo', 'venv', 'this'}
_stdlib = [
    m for m in sys.stdlib_module_names
    if m not in _STDLIB_SKIP and importlib.util.find_spec(m) is not None
]
hiddenimports += _stdlib
for _m in _stdlib:
    _spec = importlib.util.find_spec(_m)
    if _spec and _spec.submodule_search_locations:
        hiddenimports += collect_submodules(_m)

# pydeps 側のライブラリ (imgutils 等) が使う PIL サブモジュールも全量同梱
hiddenimports += collect_submodules('PIL')


a = Analysis(
    ['mosaic.py'],
    pathex=[],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # 重い自動検出依存は exe に含めず pydeps に後から入れる。
        # exe 内に部分的なコピーがあると FrozenImporter がそちらを優先して
        # pydeps 側を隠すため、pydeps で提供されるパッケージは全て除外する
        # (アプリ本体が使う numpy / PIL / cv2 は除く)。
        'torch', 'torchvision', 'transformers', 'imgutils',
        'dghs-imgutils', 'onnxruntime', 'huggingface_hub', 'sam2',
        'annotated_doc', 'anyio', 'bchlib', 'bitmath', 'bracex',
        'cachetools', 'certifi', 'chardet', 'charset_normalizer', 'click',
        'cloudpickle', 'colorama', 'dateutil', 'emoji', 'filelock',
        'flatbuffers', 'fsspec', 'functorch', 'google', 'h11', 'hbutils',
        'hf_xet', 'hfutils', 'httpcore', 'httpx', 'idna', 'jinja2',
        'joblib', 'markdown_it', 'markupsafe', 'mdurl', 'mpmath',
        'narwhals', 'natsort', 'networkx', 'packaging', 'pandas', 'piexif',
        'pilmoji', 'pkg_resources', 'pyclipper', 'pygments', 'pyparsing',
        'pytimeparse', 'random_user_agent', 'regex', 'requests', 'rich',
        'safetensors', 'scipy', 'setuptools', 'shapely', 'shellingham',
        'sklearn', 'sympy', 'tokenizers', 'torchgen', 'tqdm', 'typer',
        'tzdata', 'tzlocal', 'urllib3', 'urlobject', 'wcmatch', 'yaml',
        '_yaml', '_distutils_hack',
    ],
    noarchive=False,
    optimize=0,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.datas,
    [],
    name='MosaicEditor',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
)
