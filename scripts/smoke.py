#!/usr/bin/env python3
"""Live check against a running laya-server.

Usage:
    python scripts/smoke.py [base_url]

Part one sends every example from the TypeSafe API reference over plain HTTP and
checks the response against the documented shape. Part two runs the official
TypeSafe SDK against the same server, which proves the drop-in replacement works.
"""

import json
import os
import sys
import urllib.error
import urllib.request

BASE = (sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8123").rstrip("/")

STATE = "Help! My payouts have been failing for 3 days."
QUESTIONS = {
    "is_urgent": {
        "type": "noul",
        "instructions": "Does this convey urgency?",
        "criteria": {"true": "Explicitly time-sensitive", "false": "No urgency expressed"},
    },
    "department": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
            "billing": "Payments, invoicing, refunds",
            "technical": "Bugs, outages, integrations",
            "sales": "Pricing, upgrades, new accounts",
        },
    },
    "frustration": {
        "type": "score",
        "instructions": "How frustrated is the customer?",
        "criteria": ["Calm", "Frustrated", "Very angry"],
    },
}

ANSWER_KEYS = {
    "noul": {"type", "noul"},
    "choice": {"type", "choice", "probabilities", "confidence"},
    "score": {"type", "score", "legend", "probabilities", "confidence"},
}

failures = []


def check(name: str, condition: bool, detail: str = "") -> None:
    print(f"{'PASS' if condition else 'FAIL'}  {name}{'  ' + detail if detail else ''}")
    if not condition:
        failures.append(name)


def call(path: str, body=None, headers=None):
    """Return (status, parsed body)."""
    data = None if body is None else json.dumps(body).encode()
    request = urllib.request.Request(
        BASE + path,
        data=data,
        headers={"Content-Type": "application/json", **(headers or {})},
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as error:
        return error.code, json.loads(error.read() or b"null")


print(f"--- HTTP against {BASE} ---")

status, health = call("/healthz")
check("healthz is ready", status == 200 and health["status"] == "ok", str(health))

status, models = call("/v1/models")
names = {model["name"] for model in models.get("models", [])}
check("models lists the Jev aliases", {"jev-latest", "jev-preview", "jev-1.13.0"} <= names)

status, body = call(
    "/v1/systemone",
    {"state": STATE, "model": "jev-latest", "questions": QUESTIONS},
    {"Authorization": "Bearer ignored-by-this-server"},
)
check("evaluation succeeds", status == 200, str(body)[:200])
if status == 200:
    check("top level keys match Jev", set(body) == {"model", "answers", "usage"}, str(set(body)))
    check("usage keys match Jev", set(body["usage"]) == {"input_tokens", "output_tokens"})
    for qid, answer in body["answers"].items():
        kind = answer["type"]
        check(f"{qid} ({kind}) answer keys match Jev", set(answer) == ANSWER_KEYS[kind], str(set(answer)))
    choice = body["answers"]["department"]
    check(
        "choice probabilities sum to 1",
        abs(sum(choice["probabilities"].values()) - 1) < 0.01,
    )
    check("choice is one of the options", choice["choice"] in choice["probabilities"])
    score = body["answers"]["frustration"]
    check("score legend covers every level", set(score["legend"]) == set(score["probabilities"]))
    check("noul is a probability", 0 <= body["answers"]["is_urgent"]["noul"] <= 1)
    print("     answers:", json.dumps(body["answers"], indent=None)[:300])

status, _ = call("/v1/systemone", {"state": "x", "model": "gpt-4", "questions": {"q": QUESTIONS["is_urgent"]}})
check("unknown model returns 404", status == 404, str(status))

status, body = call("/v1/systemone", {"state": "x", "model": "jev-latest", "questions": {"q": {"type": "noul"}}})
check("missing instructions returns 422", status == 422, str(status))
check("422 body names the offending field", "detail" in (body or {}), str(body)[:120])

status, _ = call("/v1/systemone", {"state": "x", "model": "jev-latest", "questions": {}})
check("empty questions returns 422", status == 422, str(status))

print("\n--- official TypeSafe SDK against the same server ---")
try:
    from typesafe_sdk import Choice, Noul, Score, TypeSafeClient, TypeSafeNotFoundError

    os.environ.setdefault("TYPESAFE_API_KEY", "not-used-by-this-server")
    with TypeSafeClient(base_url=BASE, timeout=120.0) as client:
        result = client.system_one(
            state=STATE,
            model="jev-latest",
            questions={
                "is_urgent": Noul(instructions="Does this convey urgency?"),
                "department": Choice(
                    instructions="Which team should handle this?",
                    criteria={"billing": None, "technical": None, "sales": None},
                ),
                "frustration": Score(
                    instructions="How frustrated is the customer?",
                    criteria=["Calm", "Frustrated", "Very angry"],
                ),
            },
        )
        check("SDK parses the response", result is not None)
        check("SDK reads the noul", 0 <= result.nouls["is_urgent"].noul <= 1)
        check(
            "SDK reads the choice",
            result.choices["department"].choice in {"billing", "technical", "sales"},
            result.choices["department"].choice,
        )
        check("SDK reads the score", 0 <= result.scores["frustration"].score <= 2)
        check("SDK reads the confidence", 0 <= result.choices["department"].confidence <= 1)
        check("SDK reads usage", result.usage.input_tokens > 0)
        check("SDK lists models", any(m.name == "jev-latest" for m in client.models.list().models))
        try:
            client.system_one(state="x", model="gpt-4", questions={"q": Noul(instructions="y?")})
            check("SDK maps 404 to TypeSafeNotFoundError", False, "no error raised")
        except TypeSafeNotFoundError:
            check("SDK maps 404 to TypeSafeNotFoundError", True)
except ImportError:
    print("SKIP  typesafe-sdk is not installed (pip install typesafe-sdk)")

print()
if failures:
    print(f"{len(failures)} check(s) failed: {', '.join(failures)}")
    sys.exit(1)
print("All checks passed.")
