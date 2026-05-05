"""Shared data models for retrieval, scoring, and API responses."""

from __future__ import annotations

from typing import Optional

from pydantic import BaseModel, Field


class RetrievedJob(BaseModel):
    job_id: str
    category: str
    job_title: str
    job_description: str
    skills: list[str] = Field(default_factory=list)
    cosine_sim: float


class ScoreBreakdown(BaseModel):
    match_score: float
    semantic_pct: float
    skill_overlap_pct: float
    category_bonus: float
    matched_skills: list[str]
    missing_skills: list[str]
    matched_skill_count: int
    total_skill_count: int


class ScoredJob(BaseModel):
    job: RetrievedJob
    score: ScoreBreakdown


class RankedMatch(BaseModel):
    job_id: str
    job_title: str
    why_match: str
    evidence: list[str] = Field(default_factory=list)
    missing_skills: list[str] = Field(default_factory=list)


class SkillGap(BaseModel):
    skill: str
    evidence_job_ids: list[str] = Field(default_factory=list)
    suggestion: str


class RagGeneration(BaseModel):
    ranked_matches: list[RankedMatch] = Field(default_factory=list)
    skill_gaps: list[SkillGap] = Field(default_factory=list)
    notes: Optional[str] = None
