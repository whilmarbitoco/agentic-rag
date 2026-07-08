"""Stage 5 - Validator.

Self-checking feedback loop. The default implementation is a lightweight guard
that flags "I don't have access to your data" style hallucinations and empty
answers. Override with a stronger LLM judge by subclassing `Stage`.

Returns a `ValidationVerdict`.
"""
from __future__ import annotations

from .base import Stage, StageContext
from ..types import InterpretedQuery, RankedResult, ValidationVerdict
from ..llm.base import LLMProvider
from ..prompts import VALIDATOR_SYSTEM, VALIDATOR_USER


class ValidatorStage(Stage):
    name = "validator"

    def __init__(self, llm: LLMProvider | None = None):
        self.llm = llm

    def run(
        self,
        ctx: StageContext,
        interpreted: InterpretedQuery,
        ranked: list[RankedResult],
        candidate: str,
    ) -> ValidationVerdict:
        provider = self.llm or ctx.llm.get("validator")
        triggers = [
            "don't have access to your",
            "i don't have access",
            "no access to your data",
            "i cannot access your",
        ]
        low = (candidate or "").lower()
        if not candidate or len(candidate.strip()) < 5:
            return ValidationVerdict(valid=False, critique="Answer is empty or too short.")
        if any(t in low for t in triggers) and ranked:
            return ValidationVerdict(
                valid=False,
                critique="Answer claims no data access but tools returned data.",
            )
        try:
            out = provider.complete_json(
                [{"role": "user", "content": VALIDATOR_USER.format(answer=candidate)}],
                system_prompt=VALIDATOR_SYSTEM,
            )
            return ValidationVerdict(
                valid=bool(out.get("valid", True)),
                critique=out.get("critique", ""),
            )
        except Exception:
            return ValidationVerdict(valid=True, critique="")
