"""Yukon Physician Fee Guide — structural extractor.
Verified against the 2024 edition.

Subclasses BCExtractor: Yukon's PDF layout closely mirrors BC's (same chapter
font size 13.0, same body size 10.0). The only distinctive feature is the
4-digit code regex.
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
