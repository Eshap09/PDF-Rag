from pathlib import Path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from app.config import get_settings
import pypdf


def load_and_chunk(file_path: str, original_filename: str) -> list[Document]:
    path = Path(file_path)
    ext  = path.suffix.lower()

    if ext == ".pdf":
        docs = _load_pdf(file_path, original_filename)
    elif ext in (".txt", ".md"):
        docs = _load_text(file_path, original_filename)
    else:
        raise ValueError(f"Unsupported file type: {ext}")

    return _split(docs)


def _load_pdf(file_path: str, filename: str) -> list[Document]:
    """
    Try normal text extraction first.
    If avg chars per page is below threshold — fall back to OCR.
    """
    reader     = pypdf.PdfReader(file_path)
    total_pages = len(reader.pages)

    # attempt normal extraction
    pages = []
    for i, page in enumerate(reader.pages):
        text = (page.extract_text() or "").strip()
        pages.append((i + 1, text))

    # check if extraction is too sparse
    avg_chars = sum(len(t) for _, t in pages) / total_pages
    print(f"[chunker] PDF '{filename}' — avg chars/page: {avg_chars:.1f}")

    if avg_chars < 150:
        print(f"[chunker] sparse text detected — switching to OCR")
        return _load_pdf_ocr(file_path, filename, total_pages)

    # normal extraction was fine — build Documents
    docs = []
    for page_num, text in pages:
        if len(text) < 20:
            continue
        docs.append(Document(
            page_content=text,
            metadata={
                "source":       filename,
                "page_number":  page_num,
                "total_pages":  total_pages,
                "extraction":   "text",
            }
        ))

    return docs


def _load_pdf_ocr(file_path: str, filename: str, total_pages: int) -> list[Document]:
    """
    Convert each PDF page to an image then run tesseract OCR on it.
    Slower but works on image-based PDFs.
    """
    from pdf2image import convert_from_path
    import pytesseract

    print(f"[chunker] running OCR on {total_pages} pages — this may take a while...")

    # convert PDF pages to PIL images
    # dpi=200 is the sweet spot — high enough for accuracy, low enough for speed
    images = convert_from_path(file_path, dpi=200)

    docs = []
    for i, image in enumerate(images):
        page_num = i + 1

        # run OCR
        text = pytesseract.image_to_string(image).strip()

        print(f"  page {page_num:2d}/{total_pages} — {len(text)} chars extracted")

        if len(text) < 20:
            continue

        docs.append(Document(
            page_content=text,
            metadata={
                "source":       filename,
                "page_number":  page_num,
                "total_pages":  total_pages,
                "extraction":   "ocr",      # tells you how this chunk was extracted
            }
        ))

    return docs


def _load_text(file_path: str, filename: str) -> list[Document]:
    text = Path(file_path).read_text(encoding="utf-8")
    return [Document(
        page_content=text,
        metadata={
            "source":      filename,
            "page_number": None,
            "extraction":  "text",
        }
    )]


def _split(docs: list[Document]) -> list[Document]:
    settings = get_settings()

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )

    chunks = splitter.split_documents(docs)

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i

    return chunks


# keep old functions as thin wrappers
def load_document(file_path: str) -> str:
    path = Path(file_path)
    ext  = path.suffix.lower()
    if ext in (".txt", ".md"):
        return path.read_text(encoding="utf-8")
    elif ext == ".pdf":
        reader = pypdf.PdfReader(file_path)
        return "\n\n".join(p.extract_text() or "" for p in reader.pages)
    raise ValueError(f"Unsupported: {ext}")


def chunk_document(text: str, source: str) -> list[Document]:
    settings = get_settings()
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
        separators=["\n\n", "\n", ". ", " "],
    )
    chunks = splitter.create_documents(
        texts=[text],
        metadatas=[{"source": source}]
    )
    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_index"] = i
    return chunks