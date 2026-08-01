"""
documents/document_manager.py — Manages all user documents.

Responsibilities:
- Scan the configured documents directory
- Build an index of available documents by type
- Select the best document for a given application
- Track which document version was used per application

Currently supports: resume, cover_letter, certificate, transcript, portfolio
"""

from __future__ import annotations

import mimetypes
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path

from core.exceptions import DocumentNotFoundError
from core.logger import logger


class DocumentType(str, Enum):
    RESUME = "resume"
    COVER_LETTER = "cover_letter"
    CERTIFICATE = "certificate"
    TRANSCRIPT = "transcript"
    PORTFOLIO = "portfolio"
    OTHER = "other"


@dataclass
class Document:
    """Metadata about a single document file."""

    path: Path
    type: DocumentType
    filename: str
    size_bytes: int
    is_default: bool = False

    def __str__(self) -> str:
        return f"Document({self.type.value}, {self.filename}, {'default' if self.is_default else 'alt'})"


class DocumentManager:
    """
    Scans and manages documents available for job applications.

    Usage:
        manager = DocumentManager(documents_dir="/Users/himanshu/Resumes",
                                   default_resume="Himanshu_ATS_LATEX_7_JUNE.pdf")
        resume = manager.get_resume()   # → Document
    """

    # File extensions → document types
    _EXTENSION_MAP: dict[str, DocumentType] = {
        ".pdf": DocumentType.RESUME,
        ".doc": DocumentType.RESUME,
        ".docx": DocumentType.RESUME,
    }

    # Filename keywords → document types (case-insensitive)
    _KEYWORD_MAP: list[tuple[str, DocumentType]] = [
        ("resume", DocumentType.RESUME),
        ("cv", DocumentType.RESUME),
        ("cover", DocumentType.COVER_LETTER),
        ("letter", DocumentType.COVER_LETTER),
        ("cert", DocumentType.CERTIFICATE),
        ("transcript", DocumentType.TRANSCRIPT),
        ("portfolio", DocumentType.PORTFOLIO),
    ]

    def __init__(
        self,
        documents_dir: str,
        default_resume: str = "",
    ) -> None:
        self.documents_dir = Path(documents_dir)
        self.default_resume_name = default_resume
        self._index: dict[DocumentType, list[Document]] = {}
        self._refresh()

    def _refresh(self) -> None:
        """Scan the documents directory and rebuild the index."""
        self._index = {dt: [] for dt in DocumentType}

        if not self.documents_dir.exists():
            logger.warning("Documents directory not found: {}", self.documents_dir)
            return

        supported_extensions = {".pdf", ".doc", ".docx", ".txt"}
        count = 0

        for file_path in self.documents_dir.iterdir():
            if file_path.suffix.lower() not in supported_extensions:
                continue
            if file_path.name.startswith("."):
                continue

            doc_type = self._classify(file_path)
            is_default = file_path.name == self.default_resume_name
            if is_default:
                doc_type = DocumentType.RESUME

            doc = Document(
                path=file_path,
                type=doc_type,
                filename=file_path.name,
                size_bytes=file_path.stat().st_size,
                is_default=is_default,
            )
            self._index[doc_type].append(doc)
            count += 1

            # Sort so default always comes first
            self._index[doc_type].sort(key=lambda d: (not d.is_default, d.filename))

        logger.info(
            "Document index refreshed — {} documents found in {}",
            count,
            self.documents_dir,
        )
        for dt, docs in self._index.items():
            if docs:
                logger.debug("  {} — {} file(s)", dt.value, len(docs))

    def _classify(self, path: Path) -> DocumentType:
        """Determine document type from filename keywords."""
        name_lower = path.name.lower()
        for keyword, doc_type in self._KEYWORD_MAP:
            if keyword in name_lower:
                return doc_type
        return DocumentType.OTHER

    def get_resume(self, version: str | None = None) -> Document:
        """
        Return the best available resume.

        If `version` is specified, look for a file with that name.
        Otherwise, return the default resume.

        Raises DocumentNotFoundError if no resume is found.
        """
        resumes = self._index.get(DocumentType.RESUME, [])

        if not resumes:
            raise DocumentNotFoundError(
                f"No resume found in {self.documents_dir}",
                document_type="resume",
            )

        if version:
            for doc in resumes:
                if version in doc.filename:
                    return doc
            raise DocumentNotFoundError(
                f"Resume version '{version}' not found in {self.documents_dir}",
                document_type="resume",
            )

        # Return default if marked, otherwise the first one
        default = next((d for d in resumes if d.is_default), None)
        return default or resumes[0]

    def get_by_type(self, doc_type: DocumentType) -> list[Document]:
        """Return all documents of a given type."""
        return self._index.get(doc_type, [])

    def list_all(self) -> list[Document]:
        """Return all indexed documents."""
        return [doc for docs in self._index.values() for doc in docs]

    def refresh(self) -> None:
        """Re-scan the directory (call after adding new files)."""
        self._refresh()

    def summary(self) -> dict:
        """Return a summary dict for API responses."""
        return {
            dt.value: [
                {"filename": d.filename, "size_bytes": d.size_bytes, "is_default": d.is_default}
                for d in docs
            ]
            for dt, docs in self._index.items()
            if docs
        }
