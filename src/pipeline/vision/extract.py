"""Per-window extraction call.

Responsibility:
- Build the prompt (via :mod:`prompts`).
- Dispatch one ``chat_vision_json`` call via ``asyncio.to_thread`` so the
  sync OpenAI SDK cooperates with our asyncio fan-out.
- Retry exactly once on :class:`pydantic.ValidationError` with a small
  temperature nudge (0.0 -> 0.2). On second failure, emit zero records
  (regression gate catches cumulative damage).
- Log failures to the diagnostics path for post-mortem.
"""
from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from pydantic import ValidationError

from src.openai_client import OpenAIClient
from src.pipeline.schema import Province
from src.pipeline.vision.prompts import build_prompt
from src.pipeline.vision.schema import VisionRecord, WindowExtraction
from src.pipeline.vision.windows import Window

logger = logging.getLogger(__name__)


async def extract_window(
    *,
    window: Window,
    province: Province,
    images: list[bytes],
    client: OpenAIClient,
    model: str = "gpt-5.4-mini",
    failure_log: Path | None = None,
) -> list[VisionRecord]:
    """Extract records from one window. Order: context image first, target second.

    Returns [] on two consecutive ValidationErrors (logs to failure_log if set).
    """
    prompt = build_prompt(
        province=province,
        target_page=window.target_page,
        section=window.section_hints,
    )

    last_err: ValidationError | None = None
    for attempt, temperature in enumerate([0.0, 0.2]):
        try:
            result: WindowExtraction = await asyncio.to_thread(
                client.chat_vision_json,
                prompt=prompt,
                images=images,
                schema=WindowExtraction,
                model=model,
                temperature=temperature,
            )
            return list(result.records)
        except ValidationError as e:
            logger.warning(
                "window %d (province %s) validation failed on attempt %d: %s",
                window.target_page, province, attempt + 1, e,
            )
            last_err = e

    if failure_log is not None:
        failure_log.parent.mkdir(parents=True, exist_ok=True)
        with failure_log.open("a", encoding="utf-8") as f:
            f.write(
                json.dumps(
                    {
                        "province": province,
                        "target_page": window.target_page,
                        "error_class": type(last_err).__name__,
                        "message": str(last_err),
                    }
                )
                + "\n"
            )
    return []
