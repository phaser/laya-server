"""Request and response models for the Jev evaluation API.

Shapes, field names and limits follow the TypeSafe HTTP API reference at
https://docs.typesafe.ai/api. Nothing here is specific to laya.
"""

from typing import Annotated, Any, Literal, Union

from pydantic import BaseModel, Field, field_validator

# Jev accepts text, a JSON object, or a JSON array wherever free content is allowed.
Content = Union[str, dict[str, Any], list[Any]]
# A criterion description; null means "this option needs no extra detail".
Criterion = Union[str, dict[str, Any], list[Any], None]

MAX_CHOICE_OPTIONS = 255
MIN_SCORE_LEVELS = 2
MAX_SCORE_LEVELS = 10


class NoulCriteria(BaseModel):
    """Optional descriptions of what a yes and a no mean."""

    true: Criterion = None
    false: Criterion = None


class NoulQuestion(BaseModel):
    """A yes/no question. The answer is the probability that the answer is yes."""

    type: Literal["noul"]
    instructions: Content
    criteria: NoulCriteria | None = None


class ChoiceQuestion(BaseModel):
    """Selects one option from a set. The answer carries the full distribution."""

    type: Literal["choice"]
    instructions: Content
    criteria: dict[str, Criterion] | list[str]

    @field_validator("criteria")
    @classmethod
    def check_options(cls, value: dict | list) -> dict | list:
        if not value:
            raise ValueError("choice criteria must not be empty")
        if len(value) > MAX_CHOICE_OPTIONS:
            raise ValueError(f"choice accepts at most {MAX_CHOICE_OPTIONS} options")
        if isinstance(value, list) and len(set(value)) != len(value):
            raise ValueError("choice labels must be unique")
        return value


class ScoreQuestion(BaseModel):
    """Rates the state against ordered levels."""

    type: Literal["score"]
    instructions: Content
    criteria: list[Union[str, dict[str, Any], list[Any]]]

    @field_validator("criteria")
    @classmethod
    def check_levels(cls, value: list) -> list:
        if not MIN_SCORE_LEVELS <= len(value) <= MAX_SCORE_LEVELS:
            raise ValueError(
                f"score accepts {MIN_SCORE_LEVELS} to {MAX_SCORE_LEVELS} levels"
            )
        return value


Question = Annotated[
    Union[NoulQuestion, ChoiceQuestion, ScoreQuestion],
    Field(discriminator="type"),
]


class SystemOneRequest(BaseModel):
    """The body of POST /v1/systemone."""

    state: Content
    model: str
    questions: dict[str, Question] = Field(min_length=1)


class NoulAnswer(BaseModel):
    type: Literal["noul"] = "noul"
    noul: float


class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float


class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    legend: dict[str, str]
    probabilities: dict[str, float]
    confidence: float


Answer = Annotated[
    Union[NoulAnswer, ChoiceAnswer, ScoreAnswer],
    Field(discriminator="type"),
]


class Usage(BaseModel):
    input_tokens: int
    output_tokens: int


class SystemOneResponse(BaseModel):
    """The body of a successful POST /v1/systemone."""

    model: str
    answers: dict[str, Answer]
    usage: Usage


class ModelCard(BaseModel):
    name: str
    description: str
    release_date: str


class ModelList(BaseModel):
    """The body of GET /v1/models."""

    models: list[ModelCard]
