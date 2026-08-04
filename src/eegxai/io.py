"""Saving + reporting helpers for exploratory / analysis outputs.

Each run writes a self-contained folder — config + metrics (CSV/NPZ) + figures + a
Markdown overview that is also rendered to PDF for communication. The same layout doubles
as the golden-fixture format for reproducibility (freeze a run folder, verify against it).

    run_dir = new_run_dir("results/exploration", "mi_explore", config=cfg)
    save_table(rows, run_dir / "qc.csv")
    save_arrays(run_dir / "erd.npz", **arrays)
    write_report(run_dir, "Title", summary_md, tables=[...], figures=[...])  # -> report.md + report.pdf
"""

from __future__ import annotations

import json
import platform
import re
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
import pandas as pd

_VERSION_PKGS = ("mne", "numpy", "scipy", "matplotlib", "pandas")


def _slug(text: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "-", text).strip("-")


def package_versions(pkgs=_VERSION_PKGS) -> dict[str, str | None]:
    """Record versions for provenance / reproducibility."""
    out: dict[str, str | None] = {"python": platform.python_version()}
    for p in pkgs:
        try:
            out[p] = version(p)
        except PackageNotFoundError:
            out[p] = None
    return out


def new_run_dir(base_dir, name: str, config: dict | None = None) -> Path:
    """Create ``<base_dir>/<name>_<utc-timestamp>/`` with a ``figures/`` subfolder and a
    ``config.json`` capturing params + package versions + timestamp."""
    ts = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    run_dir = Path(base_dir) / f"{_slug(name)}_{ts}"
    (run_dir / "figures").mkdir(parents=True, exist_ok=True)
    meta = {"name": name, "created_utc": ts, "versions": package_versions()}
    if config:
        meta["config"] = config
    (run_dir / "config.json").write_text(json.dumps(meta, indent=2))
    return run_dir


def save_table(rows, path) -> pd.DataFrame:
    """Write a list-of-dicts (or DataFrame) to CSV; return the DataFrame."""
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    df.to_csv(path, index=False)
    return df


def save_arrays(path, **arrays) -> None:
    """Bundle named numpy arrays into a single ``.npz``."""
    np.savez(path, **arrays)


def _df_to_md(df: pd.DataFrame) -> str:
    """Minimal DataFrame -> Markdown table (avoids a tabulate dependency)."""
    cols = list(df.columns)
    head = "| " + " | ".join(map(str, cols)) + " |"
    sep = "| " + " | ".join("---" for _ in cols) + " |"
    body = [
        "| " + " | ".join(f"{v}" for v in row) + " |"
        for row in df.itertuples(index=False, name=None)
    ]
    return "\n".join([head, sep, *body])


def write_report(run_dir: Path, title: str, summary_md: str,
                 tables=None, figures=None) -> tuple[Path, Path]:
    """Assemble ``report.md`` and render ``report.pdf``.

    ``tables``  : list of ``(caption, DataFrame)``.
    ``figures`` : list of ``(caption, png_path)`` — copied under ``figures/`` if not already there.
    """
    parts = [f"# {title}", "", summary_md.strip(), ""]
    for cap, df in (tables or []):
        parts += [f"## {cap}", "", _df_to_md(df), ""]
    if figures:
        parts += ["## Figures", ""]
        for cap, fig in figures:
            name = Path(fig).name
            parts += [f"**{cap}**", "", f"![{cap}](figures/{name})", ""]
    md_text = "\n".join(parts)

    md_path = run_dir / "report.md"
    md_path.write_text(md_text)
    pdf_path = run_dir / "report.pdf"
    _md_to_pdf(md_text, pdf_path, base_dir=run_dir)
    return md_path, pdf_path


def _md_to_pdf(md_text: str, pdf_path: Path, base_dir: Path) -> None:
    """Render Markdown -> HTML -> PDF (self-contained: markdown + xhtml2pdf, no system deps)."""
    import markdown
    from xhtml2pdf import pisa

    body = markdown.markdown(md_text, extensions=["tables"])
    html = (
        "<html><head><meta charset='utf-8'><style>"
        "body{font-family:Helvetica,Arial,sans-serif;font-size:11px;line-height:1.4}"
        "h1{font-size:20px} h2{font-size:15px;margin-top:14px}"
        "table{border-collapse:collapse;margin:6px 0} td,th{border:1px solid #999;padding:3px 7px}"
        "img{max-width:520px}"
        f"</style></head><body>{body}</body></html>"
    )

    def link_callback(uri, _rel):
        p = base_dir / uri
        return str(p) if p.exists() else uri

    with open(pdf_path, "wb") as f:
        pisa.CreatePDF(html, dest=f, link_callback=link_callback)
