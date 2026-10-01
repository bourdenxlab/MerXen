"""Static HTML of the annotation report (plan §9: ``report.html``).

One self-contained page: inline CSS, no scripts, no network assets; figures
are the PNGs beside it (``figures/<stem>.png``), each linked to its PDF and
its CSV. The page carries no timestamp, so identical inputs give identical
bytes.
"""

from __future__ import annotations

import html
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Final

import pandas as pd

from merxen.annotation.report_model import ReportItem
from merxen.annotation.report_panel import Banner

MAX_TABLE_ROWS: Final = 40

CSS: Final = """
:root {
  --bg: #ffffff; --fg: #1d1d1f; --muted: #5f6368; --line: #d9d9de;
  --card: #f6f6f8; --error: #b3261e; --warning: #9a6700; --info: #3980be;
  --ok: #1e7a3c;
}
@media (prefers-color-scheme: dark) {
  :root {
    --bg: #141416; --fg: #ececf0; --muted: #a0a0a8; --line: #33333a;
    --card: #1d1d21; --error: #ff8a80; --warning: #f2c14e; --info: #7fb3e6;
    --ok: #6fcf8e;
  }
}
body {
  background: var(--bg); color: var(--fg); margin: 0;
  font: 14px/1.45 system-ui, sans-serif;
}
main { max-width: 1180px; margin: 0 auto; padding: 16px; }
h1 { font-size: 22px; margin: 8px 0 4px; }
h2 {
  font-size: 18px; margin: 28px 0 6px; padding-top: 14px;
  border-top: 1px solid var(--line);
}
.muted { color: var(--muted); }
.catches { color: var(--muted); font-style: italic; }
.banner {
  border-left: 4px solid var(--info); background: var(--card);
  padding: 8px 12px; margin: 6px 0;
}
.banner.error { border-color: var(--error); }
.banner.warning { border-color: var(--warning); }
.badge {
  display: inline-block; padding: 1px 8px; border-radius: 10px;
  font-size: 12px; border: 1px solid var(--line);
}
.badge.ok { color: var(--ok); }
.badge.partial, .badge.not_available { color: var(--warning); }
.badge.failed { color: var(--error); }
table {
  border-collapse: collapse; margin: 8px 0; font-size: 12px;
  display: block; overflow-x: auto; max-width: 100%;
}
th, td {
  border: 1px solid var(--line); padding: 3px 6px; text-align: left;
  white-space: nowrap;
}
th { background: var(--card); }
figure { margin: 12px 0; }
figure img {
  max-width: 100%; height: auto; border: 1px solid var(--line);
  background: #fff;
}
figcaption { color: var(--muted); font-size: 12px; }
nav ol { columns: 2; }
a { color: var(--info); }
"""


def _escape(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.4g}"
    return html.escape(str(value))


def frame_html(frame: pd.DataFrame | None, *, max_rows: int = MAX_TABLE_ROWS) -> str:
    """Return a data frame as an HTML table (first ``max_rows`` rows)."""
    if frame is None or frame.empty:
        return ""
    shown = frame.head(max_rows)
    head = "".join(f"<th>{_escape(column)}</th>" for column in shown.columns)
    rows = "".join(
        "<tr>" + "".join(f"<td>{_escape(value)}</td>" for value in record) + "</tr>"
        for record in shown.itertuples(index=False)
    )
    more = (
        f'<p class="muted">{len(frame) - max_rows} more rows in the CSV.</p>'
        if len(frame) > max_rows
        else ""
    )
    return f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>{more}"


def banner_html(banner: Banner) -> str:
    """Return one banner."""
    who = f"<strong>{_escape(banner.sample_id)}</strong>: " if banner.sample_id else ""
    return f'<div class="banner {banner.severity}">{who}{_escape(banner.text)}</div>'


def _relative(path: Path, root: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return path.name


def item_html(item: ReportItem, root: Path) -> str:
    """Return one item's section."""
    parts = [
        f'<section id="{item.slug}"><h2>{item.number}. {_escape(item.title)} '
        f'<span class="badge {item.status}">{_escape(item.status)}</span></h2>',
        f'<p class="catches">Catches: {_escape(item.catches)}</p>',
    ]
    parts.extend(banner_html(banner) for banner in item.banners)
    if item.notes:
        parts.append(
            "<ul>"
            + "".join(f"<li>{_escape(note)}</li>" for note in item.notes)
            + "</ul>"
        )
    parts.append(frame_html(item.summary))
    for record in item.figures:
        png = _relative(record.png, root)
        pdf = _relative(record.pdf, root)
        csv = _relative(record.csv, root)
        parts.append(
            f'<figure><img src="{html.escape(png)}" alt="{_escape(record.caption)}">'
            f"<figcaption>{_escape(record.caption)} "
            f'[<a href="{html.escape(pdf)}">PDF</a>] '
            f'[<a href="{html.escape(csv)}">CSV</a>]'
            "</figcaption></figure>"
        )
    if item.tables:
        links = ", ".join(
            f'<a href="{html.escape(_relative(path, root))}">{_escape(name)}</a>'
            for name, path in sorted(item.tables.items())
        )
        parts.append(f'<p class="muted">Tables: {links}</p>')
    parts.append("</section>")
    return "\n".join(part for part in parts if part)


def _digest_table(datasets: Mapping[str, Mapping[str, Any]]) -> str:
    if not datasets:
        return ""
    frame = pd.DataFrame(
        [{"sample_id": key, **dict(value)} for key, value in sorted(datasets.items())]
    )
    return frame_html(frame)


def render_report(
    *,
    title: str,
    items: Sequence[ReportItem],
    datasets: Mapping[str, Mapping[str, Any]],
    provenance: Mapping[str, Any],
    root: Path,
) -> str:
    """Return the report page.

    Args:
        title: Page title.
        items: Report items in order.
        datasets: Per-sample digest (trust, gate).
        provenance: The provenance footer.
        root: The report directory (links are relative to it).

    Returns:
        The HTML text.
    """
    banners = [banner for item in items for banner in item.banners]
    seen: set[tuple[str | None, str]] = set()
    top: list[str] = []
    for banner in banners:
        key = (banner.sample_id, banner.code)
        if key in seen:
            continue
        seen.add(key)
        top.append(banner_html(banner))
    toc = "".join(
        f'<li><a href="#{item.slug}">{_escape(item.title)}</a> '
        f'<span class="muted">({_escape(item.status)})</span></li>'
        for item in items
    )
    footer = html.escape(json.dumps(provenance, indent=1, sort_keys=True, default=str))
    body = "\n".join(item_html(item, root) for item in items)
    return (
        '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_escape(title)}</title><style>{CSS}</style></head><body><main>"
        f"<h1>{_escape(title)}</h1>"
        '<p class="muted">MerXen annotation report (plan §9). Metrics are recorded in '
        '<a href="acceptance_metrics.json">acceptance_metrics.json</a>; verdicts '
        "against the "
        "pre-registered thresholds are made by the acceptance programme.</p>"
        + "".join(top)
        + _digest_table(datasets)
        + f"<nav><ol>{toc}</ol></nav>"
        + body
        + f'<h2 id="provenance-footer">Provenance footer</h2><pre>{footer}</pre>'
        + "</main></body></html>\n"
    )


__all__ = ["banner_html", "frame_html", "item_html", "render_report"]
