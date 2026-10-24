#!/usr/bin/env bash
# report/report.md -> report/report.html -> report/report.pdf (headless Chrome) and report/report.docx
set -euo pipefail
cd "$(dirname "$0")/../report"
pandoc report.md -s --mathml --css style.css --resource-path=.:.. -o report.html
CHROME="/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
"$CHROME" --headless=new --disable-gpu --no-pdf-header-footer --print-to-pdf="$PWD/report.pdf" "file://$PWD/report.html" 2>/dev/null
pandoc report.html -f html --resource-path=.:.. -o report.docx
echo "pages: $(pdfinfo report.pdf 2>/dev/null | awk '/Pages/{print $2}')"
