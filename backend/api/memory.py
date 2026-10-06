"""
Memory Management API Router
=============================
Provides endpoints to view, add, search, and delete persistent
cross-thread memories stored in the LangGraph store (SQLite, data/store.db).
"""

from typing import List, Dict, Any
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend import resources

router = APIRouter(prefix="/api/memory", tags=["memory"])


class MemoryItem(BaseModel):
    """One entry in the cross-thread memory store."""

    namespace: List[str]
    key: str
    value: Any


class CreateMemoryRequest(BaseModel):
    """Body for adding or replacing a memory entry. A plain string value is stored as {"content": value}."""

    key: str
    value: Dict[str, Any] | str
    namespace: List[str] = ["memories"]


@router.get("", response_model=List[MemoryItem])
async def list_memories(namespace: str = "memories"):
    """List cross-thread memories in the specified namespace."""
    ns_tuple = tuple(namespace.split(","))
    try:
        items = await resources.store.asearch(ns_tuple, limit=100)
        return [
            MemoryItem(
                namespace=list(item.namespace),
                key=item.key,
                value=item.value,
            )
            for item in items
        ]
    except Exception:
        return []


@router.post("", response_model=MemoryItem)
async def create_memory(req: CreateMemoryRequest):
    """Add or update a persistent memory item."""
    ns_tuple = tuple(req.namespace)
    val = req.value if isinstance(req.value, dict) else {"content": str(req.value)}
    await resources.store.aput(ns_tuple, req.key, val)
    return MemoryItem(namespace=req.namespace, key=req.key, value=val)


@router.delete("/{key:path}")
async def delete_memory(key: str, namespace: str = "memories"):
    """Delete a memory item by key and namespace."""
    ns_tuple = tuple(namespace.split(","))
    try:
        await resources.store.adelete(ns_tuple, key)
        return {"status": "deleted", "key": key}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
