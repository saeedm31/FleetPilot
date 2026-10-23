"""Structured (JSON) logs to stdout, so any log platform (CloudWatch, Loki, ELK) can index them."""

from __future__ import annotations

import json
import logging
import os
import time


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        out = dict(
            ts=time.strftime("%Y-%m-%dT%H:%M:%S", time.gmtime(record.created)),
            level=record.levelname,
            logger=record.name,
            msg=record.getMessage(),
        )
        if record.exc_info:
            out["exc"] = self.formatException(record.exc_info)
        return json.dumps(out)


def setup_logging() -> None:
    root = logging.getLogger()
    if any(isinstance(h.formatter, JsonFormatter) for h in root.handlers):
        return
    h = logging.StreamHandler()
    h.setFormatter(JsonFormatter())
    root.handlers = [h]
    root.setLevel(os.getenv("LOG_LEVEL", "INFO"))
