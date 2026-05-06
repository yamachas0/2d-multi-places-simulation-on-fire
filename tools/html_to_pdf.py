"""HTML → PDF 変換 (playwright chromium 経由)。

@page CSS と page-break が効くため、印刷品質の PDF が出る。

Usage:
  python tools/html_to_pdf.py <input_html> [--out <pdf>]
"""
from __future__ import annotations
import argparse, sys
from pathlib import Path

from playwright.sync_api import sync_playwright


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("input", type=str, help="入力 HTML")
    ap.add_argument("--out", default=None, help="出力 PDF (default: 同名 .pdf)")
    ap.add_argument("--layout", choices=["a4", "slide16x9"], default="slide16x9",
                    help="ページレイアウト。slide16x9 (default) はプレゼン用 16:9 ワイド。a4 は従来。")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="ページ全体の等比縮小率 (0.1〜2)。default=1.0。"
                         "scale<1 にすると playwright 側で末尾コンテンツが切れる事故が起きたため、"
                         "全体縮小は @media print 内で font-size を小さくする方針に変更済み。")
    args = ap.parse_args()
    src = Path(args.input).resolve()
    if not src.exists():
        print(f"[err] not found: {src}", file=sys.stderr)
        return 1
    out = Path(args.out).resolve() if args.out else src.with_suffix(".pdf")
    file_url = src.as_uri()

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page()
        page.set_default_timeout(120000)
        page.goto(file_url, wait_until="networkidle", timeout=120000)
        # @media print を有効に
        page.emulate_media(media="print")
        # print 切替後のレイアウト再計算を待つ (末尾切れ防止)
        page.wait_for_load_state("networkidle", timeout=60000)
        page.wait_for_timeout(500)
        # ページサイズ: slide16x9 (1920x1080 比率) or A4
        if args.layout == "slide16x9":
            pdf_kwargs = {
                # 13.33in × 7.5in = 16:9 のワイドスライド (PowerPoint 標準と同じ)
                "width": "13.33in",
                "height": "7.5in",
                "margin": {"top": "10mm", "bottom": "12mm", "left": "10mm", "right": "10mm"},
            }
        else:
            pdf_kwargs = {
                "format": "A4",
                "margin": {"top": "12mm", "bottom": "14mm", "left": "10mm", "right": "10mm"},
            }
        page.pdf(
            path=str(out),
            print_background=True,
            display_header_footer=True,
            header_template='<div style="font-size:8px;color:#888;margin-left:10mm">品川 v3 統合レポート</div>',
            footer_template='<div style="font-size:8px;color:#888;width:100%;text-align:center"><span class="pageNumber"></span> / <span class="totalPages"></span></div>',
            scale=args.scale,
            **pdf_kwargs,
        )
        browser.close()
    print(f"[ok] wrote: {out} ({out.stat().st_size / 1024:.0f} KB) layout={args.layout} scale={args.scale}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
