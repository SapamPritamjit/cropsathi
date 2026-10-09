"""CLI: verify all six artifacts and write a JSON report.

Run from the ``backend`` directory:

    python -m scripts.verify_models
"""

from __future__ import annotations

import json

from app.config import PROJECT_ROOT
from app.model_verification import build_report


def _fmt(value) -> str:
    if value is None:
        return "-"
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> int:
    report = build_report()

    header = f"{'crop':7s} {'family':10s} {'implementation':26s} " \
             f"{'decl/n':>7s} {'actual':>6s} {'output':>9s} {'status':10s}"
    print(header)
    print("-" * len(header))
    for m in report["models"]:
        print(
            f"{m['crop']:7s} {m['family']:10s} "
            f"{str(m.get('implementation', '-')):26s} "
            f"{_fmt(m.get('declared_num_classes')):>7s} "
            f"{_fmt(m.get('actual_num_classes')):>6s} "
            f"{_fmt(m.get('output_shape')):>9s} {m['status']:10s}"
        )
        if "error" in m:
            print(f"    error: {m['error']}")

    print("\nruntime:", report["runtime"])
    print("summary:", report["summary"])

    out_path = PROJECT_ROOT / "backend" / "verification_report.json"
    out_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\nreport written -> {out_path}")

    return 0 if report["summary"]["by_status"].get("verified") == report["summary"]["total"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
