"""BC MSC Payment Schedule — structural extractor.
Verified against the 2024-2025 edition.

Subclasses OntarioExtractor: BC uses the same single-pass extract, row grouping,
price detection, and _try_row behavior. The only differences are the code regex
(5 digits instead of [A-Z]\\d{3,4}), the chapter font size (13.0 vs 14.0), and
the PROVINCE label.
"""
from __future__ import annotations

import re
from typing import ClassVar

from src.pipeline.schema import Province
from src.pipeline.structural.ontario import OntarioExtractor

BC_CODE_REGEX = re.compile(r"\d{5}")


class BCExtractor(OntarioExtractor):
    PROVINCE: ClassVar[Province] = "BC"
    CODE_REGEX: ClassVar[re.Pattern[str]] = BC_CODE_REGEX
    CHAPTER_FONT_SIZE: ClassVar[float] = 13.0
