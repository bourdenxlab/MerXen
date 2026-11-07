"""Tests of the report page (``report_html``): static, escaped, no network."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from merxen.annotation.report_html import CSS, render_report
from merxen.annotation.report_model import ReportItem
from merxen.annotation.report_panel import Banner

# A URL that leaves the page's directory: a scheme (http:, https:, ftp:, data:)
# or a protocol-relative "//host".
EXTERNAL = re.compile(r"^\s*['\"]?\s*(?://|[A-Za-z][A-Za-z0-9+.-]*:)")


def _page(tmp_path: Path) -> str:
    item = ReportItem(
        number=1,
        slug="item01_x",
        title="Title <i>x</i>",
        catches="catches <script>alert(1)</script>",
        notes=["a note with <b>bold</b> & an ampersand"],
        summary=pd.DataFrame({"column <u>": ["<td>cell</td>"]}),
        banners=[Banner("error", "S_<M>", "code", "banner <em>text</em>")],
    )
    return render_report(
        title="Report <title>",
        items=[item],
        datasets={"S": {"trust": "<validated>"}},
        provenance={"key": "<value>"},
        root=tmp_path,
    )


def test_the_page_escapes_every_input_text(tmp_path: Path) -> None:
    page = _page(tmp_path)
    for raw in ("<b>bold</b>", "<i>x</i>", "<script>", "<em>text</em>", "<u>"):
        assert raw not in page
    assert "a note with &lt;b&gt;bold&lt;/b&gt; &amp; an ampersand" in page
    assert "&lt;td&gt;cell&lt;/td&gt;" in page
    assert "S_&lt;M&gt;" in page and "&lt;validated&gt;" in page


def test_the_page_loads_nothing_from_the_network(tmp_path: Path) -> None:
    page = _page(tmp_path)
    assert "@import" not in page and "@import" not in CSS
    assert "<script" not in page.lower() and "<link" not in page.lower()
    references = re.findall(r"""(?:src|href)\s*=\s*("[^"]*"|'[^']*')""", page)
    references += re.findall(r"url\(([^)]*)\)", page)
    assert not [value for value in references if EXTERNAL.match(value.strip("\"'"))]
    assert "//" not in re.sub(r"<pre>.*?</pre>", "", CSS, flags=re.S)
