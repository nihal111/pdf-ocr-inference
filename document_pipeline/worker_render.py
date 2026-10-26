"""PDF page requests for rendering inside an existing OCR worker process."""
from dataclasses import dataclass
from contextlib import contextmanager
from pathlib import Path


@dataclass(frozen=True)
class PdfPageSource:
    path: str
    dpi: int
    pages: int

    @classmethod
    def open(cls, path, dpi):
        import pypdfium2 as pdfium
        path = str(Path(path).resolve())
        with pdfium.PdfDocument(path) as document:
            return cls(path, dpi, len(document))

    def __len__(self):
        return self.pages


def page_request(source, indices, profile=None):
    request = {"pages_1_based": [index + 1 for index in indices], "profile": profile}
    if isinstance(source, PdfPageSource):
        request["pdf"] = {"path": source.path, "dpi": source.dpi}
    else:
        request["images"] = [source[index] for index in indices]
    return request


def request_images(request):
    """Return owned RGB images for PDF requests; keep the existing image path."""
    if "pdf" not in request:
        return request["images"]
    if "images" in request:
        raise ValueError("A page request must contain either PDF metadata or images")
    import pypdfium2 as pdfium
    indices = request["pages_1_based"]
    if (any(type(page) is not int or page < 1 for page in indices)
            or len(indices) != len(set(indices))):
        raise ValueError("Invalid source page indices")
    images = []
    try:
        with pdfium.PdfDocument(request["pdf"]["path"]) as document:
            if any(page > len(document) for page in indices):
                raise ValueError("Requested page exceeds the PDF page count")
            for index in indices:
                page = document[index - 1]
                try:
                    bitmap = page.render(scale=request["pdf"]["dpi"] / 72)
                    try:
                        images.append(bitmap.to_pil().convert("RGB"))
                    finally:
                        bitmap.close()
                finally:
                    page.close()
        return images
    except BaseException:
        for image in images:
            image.close()
        raise


@contextmanager
def request_image_scope(request):
    images = request_images(request)
    try:
        yield images
    finally:
        if "pdf" in request:
            for image in images:
                image.close()
