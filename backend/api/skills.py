"""
Skills API Router
=================
Exposes domain skills (aws, langgraph, python, report-writer) from
deep_agent/skills/ for the React catalog viewer and agent introspection.
"""

from pathlib import Path
from typing import List
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from backend.config import SKILLS_DIR

router = APIRouter(prefix="/api/skills", tags=["skills"])


class SkillSummary(BaseModel):
    """Catalogue entry for one skill folder."""

    name: str
    description: str
    files: List[str]
    has_instructions: bool
    has_examples: bool


def _parse_skill_description(skill_md_path: Path) -> str:
    """Extract description from SKILL.md frontmatter or first paragraph."""
    if not skill_md_path.exists():
        return "Specialized agent capability."
    text = skill_md_path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.startswith("description:"):
            return line.replace("description:", "").strip()
    return "Domain-specific guidance and runnable examples."


@router.get("", response_model=List[SkillSummary])
async def list_skills():
    """List all available skills discovered in deep_agent/skills/."""
    skills = []
    if not SKILLS_DIR.exists():
        return skills

    for skill_folder in sorted(SKILLS_DIR.iterdir()):
        if skill_folder.is_dir() and not skill_folder.name.startswith("."):
            skill_md = skill_folder / "SKILL.md"
            instr_md = skill_folder / "instructions.md"
            ex_md = skill_folder / "examples.md"

            files = [f.name for f in skill_folder.glob("*.md")]
            desc = _parse_skill_description(skill_md)

            skills.append(
                SkillSummary(
                    name=skill_folder.name,
                    description=desc,
                    files=files,
                    has_instructions=instr_md.exists(),
                    has_examples=ex_md.exists(),
                )
            )
    return skills


@router.get("/{skill_name}")
async def get_skill_details(skill_name: str):
    """Retrieve full documentation files for a specific skill."""
    target_dir = SKILLS_DIR / skill_name
    if not target_dir.exists() or not target_dir.is_dir():
        raise HTTPException(status_code=404, detail="Skill not found")

    skill_files = {}
    for f in target_dir.glob("*.md"):
        skill_files[f.name] = f.read_text(encoding="utf-8")

    return {
        "name": skill_name,
        "description": _parse_skill_description(target_dir / "SKILL.md"),
        "files": skill_files,
    }
