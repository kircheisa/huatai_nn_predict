# -*- coding: utf-8 -*-
"""
把时点预测曲线.html 导出为 PDF（Chrome / Edge 无头模式打印，无需额外依赖）

- 版式：A4 横向（适配 15 列信号表与宽曲线图），去掉浏览器默认页眉页脚
- 打印副本中展开全部折叠内容（PDF 里无法点击展开），曲线图按页宽等比缩放
- 与网页的两处差异：信号表单元格不换行（否则 15 列会被折成每格 3-4 行）；
  快讯原文移到文末作附录（否则解读卡超过一页、被从中间劈开）
- 输出: data_new/out/时点预测曲线_{发出日}.pdf（每个发出日留档一份）

用法:
  python 43_导出PDF.py                 # 导出最新一期
  python 43_导出PDF.py 某个.html       # 导出指定页面
42 脚本注入因子后会自动调用本脚本的 export_pdf。
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "data_new" / "out"
HTML = OUT / "时点预测曲线.html"
BROWSERS = [
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe",
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]

PRINT_CSS = """<style>
@page { size: A4 landscape; margin: 8mm; }
/* 打印紧凑化：压缩卡片内外边距与字号，让前三页不留大片空白 */
body { font-size: 13px; }
h2 { margin: 0 0 10px; font-size: 20px; }
.card { padding: 10px 14px; margin-bottom: 10px; }
.card h3 { margin: 0 0 8px; font-size: 15px; }
.card p, .card li { margin: 4px 0; }
.sig-card table { font-size: 12px; }
@media print {
  * { -webkit-print-color-adjust: exact; print-color-adjust: exact; }
  body { margin: 0; background: #fff; }
  .card, #factorSection > details, #factorSection > div { break-inside: avoid; }
  #factorSection > h3 { break-after: avoid; }   /* 标题不与首张因子表分页 */
  /* 信号表：15列，单元格不换行并收紧内边距，否则每格被折成3-4行、整卡超过一页 */
  .sig-card td { white-space: nowrap; }
  .sig-card td, .sig-card th { padding: 3px 5px; font-size: 12px; }
  #factorSection table { font-size: 12px; }
  #factorSection td, #factorSection th { padding: 2px 6px; }
  #factorSection details { padding: 6px 0; }
}
</style>"""


def find_browser() -> str:
    cand = [os.environ.get("PDF_BROWSER", "")] + BROWSERS
    for b in cand:
        if b and Path(b).exists():
            return b
    raise RuntimeError("未找到 Chrome/Edge，可用环境变量 PDF_BROWSER 指定浏览器路径")


def build_print_copy(html: str) -> str:
    """由网页生成打印副本：展开折叠、曲线图可缩放、追加打印样式、快讯原文移作文末附录"""
    html = re.sub(r"<details(?![^>]*\bopen\b)([^>]*)>", r"<details open\1>", html)
    html = re.sub(r'(<svg\b[^>]*?)\swidth="(\d+)"\s+height="(\d+)"',
                  r'\1 viewBox="0 0 \2 \3" style="width:100%;height:auto" '
                  r'width="\2" height="\3"', html, count=1)
    html = html.replace("</head>", PRINT_CSS + "</head>", 1)
    html = html.replace('<div class="card"><table>', '<div class="card sig-card"><table>', 1)
    # 快讯原文移到文末作附录：留在解读卡里会把该卡撑到超过一页、被从中间劈开
    m_news = re.search(r"<details open><summary>查看搜集到的快讯原文</summary>"
                       r"(.*?)</details>", html, re.S)
    if m_news:
        html = html.replace(m_news.group(0),
                            "<p style='color:#888;font-size:12px'>"
                            "搜集到的快讯原文见文末附录。</p>", 1)
        appendix = ('<div class="card"><h3>附录：搜集到的快讯原文</h3>'
                    + m_news.group(1) + "</div>")
        html = html.replace("</body>", appendix + "\n</body>", 1)
    return html


def export_pdf(html_path: Path | None = None, out_pdf: Path | None = None) -> Path:
    html_path = Path(html_path) if html_path else HTML
    html = html_path.read_text(encoding="utf-8")
    m = re.search(r"时点预测曲线 (\d{4}-\d{2}-\d{2})", html)
    asof = m.group(1) if m else ""
    out_pdf = Path(out_pdf) if out_pdf else (
        OUT / (f"时点预测曲线_{asof}.pdf" if asof else "时点预测曲线.pdf"))

    browser = find_browser()
    with tempfile.TemporaryDirectory() as tmp:   # 临时目录用 ASCII 路径
        src, dst = Path(tmp) / "report.html", Path(tmp) / "report.pdf"
        src.write_text(build_print_copy(html), encoding="utf-8")
        subprocess.run([browser, "--headless", f"--user-data-dir={tmp}/profile",
                        "--disable-gpu", "--disable-extensions",
                        "--no-pdf-header-footer", f"--print-to-pdf={dst}",
                        src.as_uri()], timeout=180, capture_output=True)
        if not dst.exists() or dst.stat().st_size <= 1000:
            raise RuntimeError(f"PDF 生成失败（浏览器: {browser}）")
        shutil.copyfile(dst, out_pdf)
    return out_pdf


def main() -> None:
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else HTML
    pdf = export_pdf(src)
    print(f"PDF已导出: {pdf}  ({pdf.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
