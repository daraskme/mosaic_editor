"""検出カテゴリ定義.

カテゴリごとに:
- key: 内部キー
- label: UI 表示名 (日本語)
- prompts: 検出モデルに渡すテキストプロンプト群。
  SAM3 は短い名詞句 (noun phrase) を、LocateAnything-3B は
  自然文に近い記述も受け付けるので、複数プロンプトを OR で投げて
  マスクを統合する。
- enabled_default: デフォルトでチェックを入れるか
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Category:
    key: str
    label: str
    prompts: List[str]
    enabled_default: bool = True
    note: str = ""


DEFAULT_CATEGORIES: List[Category] = [
    Category(
        key="penis",
        label="男性器",
        prompts=["penis", "erect penis"],
    ),
    Category(
        key="vagina",
        label="女性器",
        prompts=["vagina", "vulva", "pussy"],
    ),
    Category(
        key="testicles",
        label="睾丸",
        prompts=["testicles", "scrotum"],
    ),
    Category(
        key="sex_act",
        label="結合部 (挿入)",
        prompts=[
            "penis penetrating vagina",
            "genitals during sexual penetration",
        ],
        note="挿入中の性器結合部を検出",
    ),
    Category(
        key="anus_penetrated",
        label="アナル (挿入時のみ)",
        prompts=[
            "penis penetrating anus",
            "object inserted in anus",
        ],
        note="挿入されている場合のみ検出・モザイク対象",
    ),
    Category(
        key="anus",
        label="アナル (常時)",
        prompts=["anus"],
        enabled_default=False,
    ),
    Category(
        key="nipples",
        label="乳首",
        prompts=["nipple"],
        enabled_default=False,
        note="通常モザイク不要のためデフォルトOFF",
    ),
]


def make_custom_category(text: str) -> Category:
    """ユーザー入力の追加クラスからカテゴリを生成する."""
    text = text.strip()
    return Category(key=f"custom:{text}", label=text, prompts=[text])
