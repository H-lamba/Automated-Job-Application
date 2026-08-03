"""
api/routes/documents.py — Document management endpoints.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, Depends

from api.schemas import DocumentInfo, DocumentsResponse
from core.config import Settings, settings_dep
from documents.document_manager import DocumentManager

router = APIRouter(prefix="/documents", tags=["Documents"])

SettingsDep = Annotated[Settings, Depends(settings_dep)]


@router.get("", response_model=DocumentsResponse)
async def list_documents(settings: SettingsDep):
    """List all indexed documents."""
    manager = DocumentManager(
        documents_dir=settings.storage.documents_dir,
        default_resume=settings.profile.default_resume,
    )
    summary = manager.summary()

    def _to_info(items: list[dict]) -> list[DocumentInfo]:
        return [DocumentInfo(**item) for item in items]

    return DocumentsResponse(
        resume=_to_info(summary.get("resume", [])),
        cover_letter=_to_info(summary.get("cover_letter", [])),
        certificate=_to_info(summary.get("certificate", [])),
        other=_to_info(summary.get("other", [])),
    )


@router.post("/refresh")
async def refresh_documents(settings: SettingsDep):
    """Re-scan the documents directory."""
    manager = DocumentManager(
        documents_dir=settings.storage.documents_dir,
        default_resume=settings.profile.default_resume,
    )
    manager.refresh()
    return {"message": "Document index refreshed", "count": len(manager.list_all())}
