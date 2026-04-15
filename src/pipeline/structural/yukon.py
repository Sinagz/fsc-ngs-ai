"""Yukon Physician Fee Guide — structural extractor.
Verified against the 2024 edition (Nov 2025 reprint).

Subclasses BCExtractor with a 4-digit code regex and a 9.0pt body font
(BC is 10.0pt; Yukon's typesetting uses the smaller size).
"""
from __future__ import annotations

import re
from typing import ClassVar

from src.pipeline.schema import Province
from src.pipeline.structural.bc import BCExtractor

YT_CODE_REGEX = re.compile(r"\d{4}")


class YukonExtractor(BCExtractor):
    PROVINCE: ClassVar[Province] = "YT"
    CODE_REGEX: ClassVar[re.Pattern[str]] = YT_CODE_REGEX
    BODY_FONT_SIZE: ClassVar[float] = 9.0
