"""
Documentation Parser for Schema Inference

Extracts text content from various documentation formats (PDF, Markdown, text)
to provide context for schema inference.
"""

import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def extract_pdf_text(pdf_path: Path) -> str:
    """
    Extract text from PDF using pymupdf.

    Args:
        pdf_path: Path to PDF file

    Returns:
        Extracted text content

    Raises:
        Exception: If PDF extraction fails (logged as warning, returns empty string)
    """
    try:
        import pymupdf  # fitz

        text_content = []
        with pymupdf.open(pdf_path) as doc:
            for page in doc:
                text_content.append(page.get_text())

        return "\n".join(text_content)

    except Exception as e:
        logger.warning(f"Failed to extract text from PDF {pdf_path}: {e}")
        return ""


def extract_markdown_text(md_path: Path) -> str:
    """
    Extract text from Markdown file.

    Args:
        md_path: Path to Markdown file

    Returns:
        Markdown content (structure preserved)
    """
    try:
        return md_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning(f"Failed to read Markdown file {md_path}: {e}")
        return ""


def extract_text(txt_path: Path) -> str:
    """
    Extract text from plain text file.

    Args:
        txt_path: Path to text file

    Returns:
        Text content
    """
    try:
        return txt_path.read_text(encoding="utf-8")
    except Exception as e:
        logger.warning(f"Failed to read text file {txt_path}: {e}")
        return ""


def extract_context_from_docs(file_paths: list[Path], max_chars: int = 50000) -> str:
    """
    Extract text content from documentation files.

    Concatenates text from multiple documentation files and truncates
    if total length exceeds max_chars.

    Args:
        file_paths: List of paths to documentation files
        max_chars: Maximum total characters (default: 50,000)

    Returns:
        Concatenated text context (truncated if needed)
    """
    if not file_paths:
        return ""

    text_parts = []

    for file_path in file_paths:
        if not file_path.exists():
            logger.warning(f"Documentation file not found: {file_path}")
            continue

        # Detect file type and extract accordingly
        suffix = file_path.suffix.lower()

        if suffix == ".pdf":
            text = extract_pdf_text(file_path)
        elif suffix == ".md":
            text = extract_markdown_text(file_path)
        elif suffix == ".txt":
            text = extract_text(file_path)
        else:
            # Unknown type, skip
            logger.warning(f"Unknown documentation file type: {file_path}")
            continue

        if text:
            text_parts.append(f"\n\n--- {file_path.name} ---\n\n{text}")

    # Concatenate all text
    full_context = "".join(text_parts)

    # Truncate if too long
    if len(full_context) > max_chars:
        logger.warning(f"Documentation context truncated from {len(full_context)} to {max_chars} chars")
        full_context = full_context[:max_chars]

    return full_context
