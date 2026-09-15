"""検出カテゴリ定義.

男性器 / 女性器 / 乳首は AnimeCensor、既存のモザイクはブロック格子で検出。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class Category:
    key: str
    label: str
    prompts: List[str] = field(default_factory=list)
    enabled_default: bool = True
    note: str = ""


DEFAULT_CATEGORIES: List[Category] = [
    Category(
        key="penis",
        label="男性器",
        note="挿入中の露出部分も penis として検出されます",
    ),
    Category(
        key="vagina",
        label="女性器",
    ),
    Category(
        key="nipples",
        label="乳首",
        enabled_default=False,
        note="通常モザイク不要のためデフォルトOFF",
    ),
    Category(
        key="mosaic",
        label="モザイク",
        enabled_default=False,
        note="既存のブロックモザイクを検出（ドット絵等の誤検出に注意）",
    ),
]
