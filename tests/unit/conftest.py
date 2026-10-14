"""Synthetic filings shaped like real inline-XBRL documents (Workiva style)."""
import pytest

LONG = " ".join(f"Sentence number {i} describes the business in some detail." for i in range(60))

TENK_HTML = f"""<html><head><title>x</title></head><body>
<div style="display:none"><ix:header><ix:hidden>HIDDEN XBRL CONTEXT 2024-01-28</ix:hidden></ix:header></div>
<!-- a comment that must not leak -->
<div>ACME Corporation</div>
<div>Annual Report on Form 10-K</div>
<table><tr><td>Item 1.</td><td>Business</td><td>3</td></tr>
<tr><td>Item 1A.</td><td>Risk Factors</td><td>9</td></tr>
<tr><td>Item 7.</td><td>Management Discussion</td><td>30</td></tr></table>
<div>Item 1. Business</div>
<div>Item 1A. Risk Factors</div>
<div>Item 7. Management Discussion</div>
<hr style="page-break-after:always"/>
<div>Table of Contents</div>
<div style="font-weight:bold">Part I</div>
<div><span>Item 1.</span><span> Business</span></div>
<div>{LONG}</div>
<div>12</div>
<hr style="page-break-after:always"/>
<div>Item 1A. Risk Factors</div>
<div>We face risks. See Item 8 of this report for the financial statements, which discuss risks.</div>
<div style="page-break-after:always">Risk paragraph two with detail.</div>
<div>Item 7. Management\xa0Discussion and Analysis</div>
<div>Revenue by segment is shown below.</div>
<table>
<tr><td></td><td>Jan 28, 2024</td><td></td><td>Jan 29, 2023</td></tr>
<tr><td>Data Center</td><td>$</td><td>47,525</td><td>$</td><td>15,005</td></tr>
<tr><td>Net loss</td><td>$</td><td>(1,234</td><td>)</td><td>(500</td><td>)</td></tr>
</table>
<table><tr><td>Layout cell that is just text.</td></tr></table>
<div>Item 8. Financial Statements</div>
<div>Statements follow.</div>
</body></html>"""

TENQ_HTML = """<html><body>
<div>Part I</div>
<div>Item 1. Financial Statements</div>
<div>Condensed statements follow with enough words to be a real body of text here.</div>
<div>Item 2. Management Discussion</div>
<div>Discussion of results goes here with some more words to make it substantial enough.</div>
<div>Part II</div>
<div>Item 1. Legal Proceedings</div>
<div>None.</div>
<div>Item 1A. Risk Factors</div>
<div>No material changes to the risk factors disclosed previously in the annual report.</div>
</body></html>"""


@pytest.fixture
def tenk_html():
    return TENK_HTML


@pytest.fixture
def tenq_html():
    return TENQ_HTML