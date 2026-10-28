"""把审计日志导出成 CSV / JSON，供排查或合规留档。

用法：
    python scripts/export_audit.py                     # 最近 200 条 -> logs/audit-export.csv
    python scripts/export_audit.py --limit 0           # 全量
    python scripts/export_audit.py --format json --out logs/audit.json

字段固定（与 ask.py / graph.py 写入的一致）：
    ts, user_id, department, level, question, hit_ids, citations,
    sensitivity_max, answered, reason, degraded, latency_ms, answer_chars
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

from src.config import get_settings
from src.security.audit import audit_path, read_events

COLUMNS = [
    "ts",
    "user_id",
    "department",
    "level",
    "question",
    "hit_ids",
    "citations",
    "sensitivity_max",
    "answered",
    "reason",
    "degraded",
    "latency_ms",
    "answer_chars",
]


def main() -> int:
    parser = argparse.ArgumentParser(description="导出审计日志")
    parser.add_argument("--limit", type=int, default=200, help="导出最近 N 条；0 = 全量")
    parser.add_argument("--format", choices=("csv", "json"), default="csv")
    parser.add_argument("--out", default=None, help="输出路径；默认 logs/audit-export.<ext>")
    args = parser.parse_args()

    settings = get_settings()
    source = audit_path(settings)
    if not source.is_file():
        print(f"[错误] 审计日志不存在：{source}")
        return 1

    events = read_events(settings, limit=args.limit or 10**9)
    if not events:
        print("[错误] 审计日志为空")
        return 1

    out = Path(args.out) if args.out else source.parent / f"audit-export.{args.format}"
    out.parent.mkdir(parents=True, exist_ok=True)

    if args.format == "json":
        out.write_text(
            json.dumps(events, ensure_ascii=False, indent=2), encoding="utf-8"
        )
    else:
        with out.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=COLUMNS, extrasaction="ignore")
            writer.writeheader()
            for event in events:
                row = dict(event)
                row["hit_ids"] = "|".join(event.get("hit_ids") or [])
                row["citations"] = "|".join(event.get("citations") or [])
                writer.writerow(row)

    print(f"导出 {len(events)} 条 -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
