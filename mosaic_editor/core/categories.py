"""検出カテゴリ定義.

検出器は deepghs/anime_censor_detection (YOLOv8)。
モデルが検出できるのは 男性器 / 女性器 / 乳首 の3クラス。
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
]
