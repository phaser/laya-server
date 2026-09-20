"""Route and mapping tests. They use a fake engine, so no checkpoint is loaded."""

import pytest
from fastapi.testclient import TestClient

from laya_server.app import create_app
from laya_server.engine import CANONICAL_MODEL, to_response

# One result in laya's own format, as laya_mlx.Agent.system_one returns it.
LAYA_RESULT = {
    "model": "laya-rl-agent",
    "answers": {
        "department": {
            "type": "choice",
            "confidence": 0.8123,
            "action": {"act_probability": 0.7},
            "choice": "billing",
            "probabilities": {"billing": 0.88, "technical": 0.12, "sales": 0.0},
        },
        "frustration": {
            "type": "score",
            "confidence": 0.9201,
            "action": {"act_probability": 0.7},
            "score": 1.05,
            "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
            "probabilities": {"0": 0.0, "1": 0.95, "2": 0.05},
        },
        "is_urgent": {
            "type": "noul",
            "confidence": 0.95,
            "action": {"act_probability": 0.7},
            "noul": 0.95,
        },
    },
    "usage": {"input_tokens": 296, "output_tokens": 0},
}


class FakeEngine:
    model_name = CANONICAL_MODEL
    checkpoint = "fake/checkpoint"
    names = frozenset({CANONICAL_MODEL, "fake/checkpoint", "jev-latest", "jev-preview", "jev-1.13.0"})

    def __init__(self, result=None, error=None):
        self.result = result if result is not None else LAYA_RESULT
        self.error = error
        self.seen = None

    def infer(self, state, questions):
        self.seen = (state, questions)
        if self.error is not None:
            raise self.error
        return self.result


def client(engine=None, **kwargs):
    return TestClient(create_app(engine or FakeEngine(), **kwargs))


# --- mapping -------------------------------------------------------------------


def test_mapping_matches_the_documented_jev_answers():
    body = to_response(LAYA_RESULT, CANONICAL_MODEL).model_dump()
    assert body == {
        "model": "laya-mlx",
        "answers": {
            "department": {
                "type": "choice",
                "choice": "billing",
                "probabilities": {"billing": 0.88, "technical": 0.12, "sales": 0.0},
                "confidence": 0.8123,
            },
            "frustration": {
                "type": "score",
                "score": 1.05,
                "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
                "probabilities": {"0": 0.0, "1": 0.95, "2": 0.05},
                "confidence": 0.9201,
            },
            "is_urgent": {"type": "noul", "noul": 0.95},
        },
        "usage": {"input_tokens": 296, "output_tokens": 0},
    }


def test_mapping_renders_a_structured_score_level_as_text():
    raw = {
        "model": "laya-rl-agent",
        "answers": {
            "risk": {
                "type": "score",
                "confidence": 0.5,
                "action": {"act_probability": 0.1},
                "score": 0.5,
                "legend": {"0": {"label": "low"}, "1": "high"},
                "probabilities": {"0": 0.5, "1": 0.5},
            }
        },
        "usage": {"input_tokens": 10, "output_tokens": 0},
    }
    legend = to_response(raw, CANONICAL_MODEL).answers["risk"].legend
    assert legend == {"0": '{"label": "low"}', "1": "high"}


# --- POST /v1/systemone --------------------------------------------------------


def test_evaluation_returns_the_documented_response():
    response = client().post(
        "/v1/systemone",
        json={
            "state": "Help! My payouts have been failing for 3 days.",
            "model": "jev-latest",
            "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}},
        },
    )
    assert response.status_code == 200
    body = response.json()
    assert set(body) == {"model", "answers", "usage"}
    assert body["answers"]["is_urgent"] == {"type": "noul", "noul": 0.95}


def test_questions_reach_laya_unchanged():
    engine = FakeEngine()
    engine.result = {
        "model": "laya-rl-agent",
        "answers": {
            "department": LAYA_RESULT["answers"]["department"],
            "is_urgent": LAYA_RESULT["answers"]["is_urgent"],
        },
        "usage": {"input_tokens": 1, "output_tokens": 0},
    }
    questions = {
        "department": {
            "type": "choice",
            "instructions": "Which team should handle this?",
            "criteria": {"billing": "Payments", "technical": None},
        },
        "is_urgent": {
            "type": "noul",
            "instructions": "Does this convey urgency?",
            "criteria": {"true": "Time sensitive", "false": "No urgency"},
        },
    }
    response = client(engine).post(
        "/v1/systemone",
        json={"state": {"ticket": "hello"}, "model": "laya-mlx", "questions": questions},
    )
    assert response.status_code == 200
    state, seen = engine.seen
    assert state == {"ticket": "hello"}
    # A null option description survives, and so do the noul true/false keys.
    assert seen["department"]["criteria"] == {"billing": "Payments", "technical": None}
    assert seen["is_urgent"]["criteria"] == {"true": "Time sensitive", "false": "No urgency"}


def test_unknown_model_is_not_found():
    response = client().post(
        "/v1/systemone",
        json={"state": "x", "model": "gpt-4", "questions": {"q": {"type": "noul", "instructions": "y?"}}},
    )
    assert response.status_code == 404


@pytest.mark.parametrize(
    "question",
    [
        {"type": "guess", "instructions": "y?"},
        {"type": "noul"},
        {"type": "choice", "instructions": "y?"},
        {"type": "choice", "instructions": "y?", "criteria": {}},
        {"type": "choice", "instructions": "y?", "criteria": ["a", "a"]},
        {"type": "score", "instructions": "y?", "criteria": ["only one"]},
        {"type": "score", "instructions": "y?", "criteria": [str(i) for i in range(11)]},
    ],
)
def test_invalid_questions_are_rejected(question):
    response = client().post(
        "/v1/systemone", json={"state": "x", "model": "laya-mlx", "questions": {"q": question}}
    )
    assert response.status_code == 422
    assert "detail" in response.json()


@pytest.mark.parametrize(
    "body",
    [
        {"model": "laya-mlx", "questions": {"q": {"type": "noul", "instructions": "y?"}}},
        {"state": "x", "questions": {"q": {"type": "noul", "instructions": "y?"}}},
        {"state": "x", "model": "laya-mlx"},
        {"state": "x", "model": "laya-mlx", "questions": {}},
        {"state": 5, "model": "laya-mlx", "questions": {"q": {"type": "noul", "instructions": "y?"}}},
    ],
)
def test_invalid_bodies_are_rejected(body):
    assert client().post("/v1/systemone", json=body).status_code == 422


def test_question_count_is_capped():
    questions = {f"q{i}": {"type": "noul", "instructions": "y?"} for i in range(3)}
    response = client(max_questions=2).post(
        "/v1/systemone", json={"state": "x", "model": "laya-mlx", "questions": questions}
    )
    assert response.status_code == 422


def test_oversized_body_is_rejected():
    response = client(max_body_bytes=64).post(
        "/v1/systemone",
        json={
            "state": "x" * 500,
            "model": "laya-mlx",
            "questions": {"q": {"type": "noul", "instructions": "y?"}},
        },
    )
    assert response.status_code == 413


def test_laya_rejection_becomes_a_validation_error():
    engine = FakeEngine(error=ValueError("Question 'q' has too many options for the token budget"))
    response = client(engine).post(
        "/v1/systemone",
        json={
            "state": "x",
            "model": "laya-mlx",
            "questions": {"q": {"type": "choice", "instructions": "y?", "criteria": ["a", "b"]}},
        },
    )
    assert response.status_code == 422
    assert "token budget" in str(response.json()["detail"])


def test_inference_failure_hides_the_traceback():
    engine = FakeEngine(error=FloatingPointError("Non-finite model outputs"))
    response = client(engine).post(
        "/v1/systemone",
        json={"state": "x", "model": "laya-mlx", "questions": {"q": {"type": "noul", "instructions": "y?"}}},
    )
    assert response.status_code == 500
    assert response.json() == {"detail": "Inference failed."}


# --- the other routes ----------------------------------------------------------


def test_models_lists_the_jev_aliases():
    body = client().get("/v1/models").json()
    names = {model["name"] for model in body["models"]}
    assert {"laya-mlx", "jev-latest", "jev-preview", "jev-1.13.0"} <= names
    for model in body["models"]:
        assert set(model) == {"name", "description", "release_date"}


def test_health_reports_the_served_model():
    body = client().get("/healthz").json()
    assert body == {"status": "ok", "model": "laya-mlx", "checkpoint": "fake/checkpoint"}


def test_documentation_routes_are_off():
    assert client().get("/docs").status_code == 404
    assert client().get("/openapi.json").status_code == 404
