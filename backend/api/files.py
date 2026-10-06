"""
Files Management Router
========================
Provides endpoints to list, view, create, edit, upload, and delete
files stored in the agent's filesystem workspace (deep_agent/).
"""

import os
from pathlib import Path
from typing import List
from fastapi import APIRouter, HTTPException, UploadFile, File, Form
from pydantic import BaseModel

from backend.config import DEEP_AGENT_DIR

router = APIRouter(prefix="/api/files", tags=["files"])


def _resolve_in_workspace(path: str) -> Path:
    """Resolve a user-supplied path, refusing anything outside deep_agent/."""
    target = (DEEP_AGENT_DIR / path.lstrip("/")).resolve()
    if not target.is_relative_to(DEEP_AGENT_DIR.resolve()):
        raise HTTPException(status_code=403, detail="Access denied: Path outside workspace")
    return target


class FileItem(BaseModel):
    """One file or directory under deep_agent/."""

    name: str
    path: str
    relative_path: str
    is_dir: bool
    size: int
    modified: float


class FileWriteRequest(BaseModel):
    """Body for creating or overwriting a file."""

    path: str
    content: str


@router.get("", response_model=List[FileItem])
async def list_files():
    """Recursively list all files and directories inside the agent workspace."""
    results = []
    if not DEEP_AGENT_DIR.exists():
        return results

    for root, dirs, files in os.walk(DEEP_AGENT_DIR):
        # Ignore hidden directories like .git
        dirs[:] = [d for d in dirs if not d.startswith(".")]

        for d in dirs:
            dir_path = Path(root) / d
            rel = dir_path.relative_to(DEEP_AGENT_DIR).as_posix()
            results.append(
                FileItem(
                    name=d,
                    path=str(dir_path),
                    relative_path=rel,
                    is_dir=True,
                    size=0,
                    modified=dir_path.stat().st_mtime,
                )
            )

        for f in files:
            if f.startswith("."):
                continue
            file_path = Path(root) / f
            rel = file_path.relative_to(DEEP_AGENT_DIR).as_posix()
            stat = file_path.stat()
            results.append(
                FileItem(
                    name=f,
                    path=str(file_path),
                    relative_path=rel,
                    is_dir=False,
                    size=stat.st_size,
                    modified=stat.st_mtime,
                )
            )

    return sorted(results, key=lambda x: (not x.is_dir, x.relative_path))


@router.get("/content")
async def get_file_content(path: str):
    """Retrieve text content of a specified file relative to deep_agent/."""
    clean_path = path.lstrip("/")
    target = _resolve_in_workspace(clean_path)

    if not target.exists() or not target.is_file():
        raise HTTPException(status_code=404, detail="File not found")

    try:
        content = target.read_text(encoding="utf-8")
        return {
            "path": clean_path,
            "filename": target.name,
            "content": content,
            "size": len(content),
        }
    except UnicodeDecodeError:
        return {
            "path": clean_path,
            "filename": target.name,
            "content": "[Binary file content cannot be displayed as text]",
            "size": target.stat().st_size,
        }


@router.post("")
async def create_or_update_file(req: FileWriteRequest):
    """Create or overwrite a file inside deep_agent/."""
    clean_path = req.path.lstrip("/")
    target = _resolve_in_workspace(clean_path)

    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(req.content, encoding="utf-8")
    return {"status": "saved", "path": clean_path, "bytes": len(req.content)}


@router.post("/upload")
async def upload_file(file: UploadFile = File(...), destination: str = Form("note")):
    """Upload a file to a folder inside deep_agent/."""
    dest_dir = _resolve_in_workspace(destination)
    filename = Path(file.filename or "").name
    if not filename:
        raise HTTPException(status_code=400, detail="Invalid filename")

    dest_dir.mkdir(parents=True, exist_ok=True)
    target = dest_dir / filename

    content = await file.read()
    with open(target, "wb") as f:
        f.write(content)

    return {"status": "uploaded", "path": f"{destination}/{filename}", "size": len(content)}


@router.delete("/{file_path:path}")
async def delete_file(file_path: str):
    """Delete a file from deep_agent/."""
    target = _resolve_in_workspace(file_path)
    if target == DEEP_AGENT_DIR.resolve():
        raise HTTPException(status_code=403, detail="Refusing to delete workspace root")

    if not target.exists():
        raise HTTPException(status_code=404, detail="File not found")

    if target.is_dir():
        import shutil
        shutil.rmtree(target)
    else:
        target.unlink()

    return {"status": "deleted", "path": file_path}
