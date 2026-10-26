"""离线建索引：文档 -> 索引（命令行触发）。"""
import argparse, json, sys
from dataclasses import asdict
from src.config import get_settings
from src.ingest.pipeline import run_ingest
from src.utils.logger import setup_logging


def main() -> int:
    parser = argparse.ArgumentParser(description="构建检索索引")
    parser.add_argument("--hash-embed", action="store_true", help="用 HashEmbedder（不下载模型）")
    args = parser.parse_args()

    setup_logging()                      # 日志走 stderr
    try:
        report = run_ingest(get_settings(), force_hash=args.hash_embed)
    except Exception as exc:              # noqa: BLE001
        print(json.dumps({"error": type(exc).__name__, "message": str(exc)}, ensure_ascii=False))
        return 1
    print(json.dumps(asdict(report), ensure_ascii=False, indent=2))   # 指标走 stdout
    return 0 if report.chunks > 0 else 1


if __name__ == "__main__":
    sys.exit(main())