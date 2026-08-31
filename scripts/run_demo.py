#!/usr/bin/env python3
"""Start the local ServeScope demo on http://127.0.0.1:8080."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import uvicorn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
CONFIG = json.loads((ROOT / "configs" / "demo.json").read_text(encoding="utf-8"))


def main() -> None:
    host = CONFIG.get("demo_host", "127.0.0.1")
    port = int(CONFIG.get("demo_port", 8080))
    print(f"ServeScope demo: http://{host}:{port}", flush=True)
    print("Start vLLM separately with scripts/_p3_start_priority.sh", flush=True)
    from servescope.demo.app import app

    uvicorn.run(app, host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
