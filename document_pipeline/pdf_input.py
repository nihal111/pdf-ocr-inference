"""Shared PDF input validation and ownership for both rendering placements."""
from contextlib import contextmanager
from pathlib import Path
import re
import stat


@contextmanager
def open_pdf(path):
    """Keep missing files distinct from invalid or incomplete PDF documents."""
    import pypdfium2 as pdfium

    source = Path(path).expanduser().resolve()
    info = source.stat()  # Preserve FileNotFoundError, including dangling symlinks.
    if not stat.S_ISREG(info.st_mode):
        raise ValueError(f"PDF input must be a regular file: {source}")
    with source.open("rb") as stream:
        header = stream.read(1024)
        stream.seek(max(0, info.st_size - 1024))
        tail = stream.read(1024)
    # PDFium can silently repair a truncated file with no EOF marker. Reject
    # incomplete inputs before invoking the native parser.
    if not re.search(rb"%PDF-\d\.\d", header) or b"%%EOF" not in tail:
        raise ValueError(f"Malformed PDF: missing PDF header or EOF marker: {source}")
    try:
        with pdfium.PdfDocument(source) as document:
            if len(document) < 1:
                raise ValueError(f"Malformed PDF: document has no pages: {source}")
            yield document
    except pdfium.PdfiumError as exc:
        raise ValueError(f"Malformed or unreadable PDF: {source}: {exc}") from exc


def render_page(document, index, dpi):
    """Return an independent RGB image and release its native page/bitmap."""
    page = document[index]
    try:
        bitmap = page.render(scale=dpi / 72)
        try:
            return bitmap.to_pil().convert("RGB")
        finally:
            bitmap.close()
    finally:
        page.close()
