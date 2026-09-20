"""Checkpoint loading, inference serialization, and the laya-to-Jev answer mapping."""

import json
import threading
from typing import Any

from .schemas import (
    ChoiceAnswer,
    ModelCard,
    ModelList,
    NoulAnswer,
    ScoreAnswer,
    SystemOneResponse,
    Usage,
)

# The name this server reports in the response. The answer comes from laya, not from Jev,
# so the server does not claim to be a Jev version.
CANONICAL_MODEL = "laya-mlx"
# Accepted for drop-in compatibility: unmodified Jev client code sends these.
JEV_ALIASES = ("jev-latest", "jev-preview", "jev-1.13.0")
# laya-mlx 0.1.0 upload date on PyPI.
RELEASE_DATE = "2026-09-19"

MODEL_LIST = ModelList(
    models=[
        ModelCard(
            name=CANONICAL_MODEL,
            description="Laya typed decisions running natively on Apple silicon with MLX.",
            release_date=RELEASE_DATE,
        ),
        *(
            ModelCard(
                name=alias,
                description=f"Jev compatibility alias. Resolves to {CANONICAL_MODEL}.",
                release_date=RELEASE_DATE,
            )
            for alias in JEV_ALIASES
        ),
    ]
)


def _as_text(value: Any) -> str:
    """Render a criterion as text, the same way laya renders it into the prompt."""
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, separators=(", ", ": "), default=str)


def to_response(raw: dict, model: str) -> SystemOneResponse:
    """Map one laya result onto the documented Jev response.

    laya adds `action.act_probability` to every answer and `confidence` to a noul
    answer. Jev defines neither, so both are dropped here.
    """
    answers: dict[str, Any] = {}
    for qid, answer in raw["answers"].items():
        kind = answer["type"]
        if kind == "noul":
            answers[qid] = NoulAnswer(noul=answer["noul"])
        elif kind == "choice":
            answers[qid] = ChoiceAnswer(
                choice=answer["choice"],
                probabilities=answer["probabilities"],
                confidence=answer["confidence"],
            )
        else:
            answers[qid] = ScoreAnswer(
                score=answer["score"],
                legend={k: _as_text(v) for k, v in answer["legend"].items()},
                probabilities=answer["probabilities"],
                confidence=answer["confidence"],
            )
    return SystemOneResponse(
        model=model,
        answers=answers,
        usage=Usage(**raw["usage"]),
    )


class Engine:
    """One laya checkpoint, with inference serialized across requests.

    MLX holds a single device context and the Agent is not safe to call from several
    threads at once, so every call takes the lock.
    """

    def __init__(
        self,
        checkpoint: str,
        *,
        dtype: str = "float16",
        batch_size: int = 16,
        device: str | None = None,
    ) -> None:
        import laya_mlx

        self.checkpoint = checkpoint
        self._lock = threading.Lock()
        self._agent = laya_mlx.load(
            checkpoint, device=device, dtype=dtype, batch_size=batch_size
        )

    @property
    def model_name(self) -> str:
        """The name reported in the response `model` field."""
        return CANONICAL_MODEL

    @property
    def names(self) -> frozenset[str]:
        """Every model name this server answers to."""
        return frozenset({CANONICAL_MODEL, self.checkpoint, *JEV_ALIASES})

    def infer(self, state: Any, questions: dict[str, dict]) -> dict:
        with self._lock:
            return self._agent.system_one(state, questions)
