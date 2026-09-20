# Plan: Jev-compatible HTTP server for laya-mlx

## Goal

Serve `laya-mlx` behind the TypeSafe **Jev** wire API. A client that today calls
`https://api.typesafe.ai` must work without a code change when `TYPESAFE_BASE_URL`
points to this server.

Reference: https://docs.typesafe.ai/api

## Source of truth

| Item | Source |
|---|---|
| Jev HTTP API | https://docs.typesafe.ai/api |
| Jev models endpoint | https://docs.typesafe.ai/models |
| Jev confidence | https://docs.typesafe.ai/confidence |
| Python SDK defaults | https://docs.typesafe.ai/sdk/python/api/constants |
| laya-mlx runtime | `laya_mlx/agent.py`, `laya_mlx/common.py` (installed package) |

## Why this is a thin wrapper

`laya_mlx.Agent.system_one(state, questions)` already accepts the Jev question
schema (`type`, `instructions`, `criteria`) and already returns
`{"model", "answers", "usage"}`. The server must add HTTP, validation, model
name resolution, and a small answer clean-up. It must not re-implement inference.

## API surface

### 1. `POST /v1/systemone`

Request body:

| Field | Type | Required |
|---|---|---|
| `state` | string, object, or array | yes |
| `model` | string | yes |
| `questions` | map of id to Question | yes |

Question types:

- `noul` — `instructions`; optional `criteria` object with `true` and `false`.
- `choice` — `instructions`; `criteria` map of option to description. 255 options maximum.
- `score` — `instructions`; `criteria` ordered array. 2 to 10 levels.

Response body: `{"model": string, "answers": map, "usage": {"input_tokens", "output_tokens"}}`.

Answer shapes:

- noul: `{"type": "noul", "noul": float}`
- choice: `{"type": "choice", "choice": string, "probabilities": map, "confidence": float}`
- score: `{"type": "score", "score": float, "legend": map, "probabilities": map, "confidence": float}`

### 2. `GET /v1/models`

Returns `{"models": [{"name", "description", "release_date"}]}`.

### 3. `GET /healthz`

Not part of Jev. It reports model readiness for nginx and for start-up scripts.

## Model names

One checkpoint runs per process. The server answers to a canonical name and to the
Jev aliases, so unmodified client code works.

| Accepted `model` value | Resolves to |
|---|---|
| `laya-mlx` | the loaded checkpoint |
| `jev-latest` | the loaded checkpoint |
| `jev-preview` | the loaded checkpoint |
| `jev-1.13.0` | the loaded checkpoint |
| the configured checkpoint id | the loaded checkpoint |

Any other value returns `404`. The response `model` field reports `laya-mlx`.
The server does not claim to be `jev-1.13.0`, because a different model produced
the answer.

## Answer mapping

laya returns three fields that Jev does not define. The server removes them, so the
response matches the documented Jev schema exactly:

| laya field | Action |
|---|---|
| `action.act_probability` | remove |
| `confidence` on a noul answer | remove — Jev returns confidence for choice and score only |
| `model: "laya-rl-agent"` | replace with the resolved server model name |

`confidence` for choice and score passes through unchanged. laya computes
normalized Shannon entropy, `1 - H(p) / log(k)`. TypeSafe does not publish its
formula. Both are in the range 0 to 1 and both fall as the distribution flattens.
Numbers will differ. Tuned thresholds need a re-check.

## Errors

| Status | Cause | Body |
|---|---|---|
| `404` | unknown model name | JSON error |
| `413` | request body over the size limit | JSON error |
| `422` | body failed validation | pydantic detail list, which names the offending field |
| `500` | inference failure | JSON error, no stack trace |

`401` is not implemented. The server has no authentication, as requested.

## Known fidelity limits

State the limits in the README. Do not hide them.

1. **Context.** Jev accepts 64k tokens. The laya English checkpoint accepts 512
   tokens total, which includes the instructions and the options. laya truncates
   a longer state. The multilingual and typed-decisions checkpoints accept 1024.
2. **Confidence.** Different formula. See above.
3. **`usage.output_tokens`** is always 0. laya decodes no tokens.
4. **Large choice sets.** Jev accepts 255 options. laya must fit every option in
   the 512-token window, so a large set raises a validation error.
5. **Accuracy.** laya is a different model with different weights. Answers will
   not match Jev answer for answer.

## Security

The server has no authentication, by request. These controls stay in the code.

- Bind to `127.0.0.1` by default. An operator must opt in to `0.0.0.0`.
- Cap the request body at 1 MiB. Cap the question count at 128.
- Never log the `state` or the `questions` content. Log method, path, status,
  question count and duration only.
- Ignore the `Authorization` header. Never echo it.
- Return no stack trace in an error body.

## Files

```
PLAN.md
README.md
pyproject.toml
.gitignore
src/laya_server/__init__.py
src/laya_server/schemas.py     request and response models, validation
src/laya_server/engine.py      checkpoint load, inference lock, answer mapping
src/laya_server/app.py         FastAPI application, routes, error handlers
src/laya_server/__main__.py    argparse command line, uvicorn start
tests/test_app.py              route and mapping tests against a fake engine
scripts/smoke.py               live check: raw HTTP plus the real TypeSafe SDK
```

## Concurrency

MLX inference runs on one Metal device and one Agent instance. Route handlers are
synchronous, so FastAPI runs them in the thread pool. A `threading.Lock` serializes
`agent.system_one`. This keeps the event loop free and prevents concurrent access
to the model.

## Steps

1. **Repository skeleton** — `git init`, `.gitignore`, `pyproject.toml`, package
   directories.
   *Verify:* `pip install -e .` succeeds in the venv.
2. **Schemas** — pydantic models for the request, the three question types and the
   three answer types, with the Jev limits.
   *Verify:* unit tests reject a bad question and accept every documented example.
3. **Engine** — load the checkpoint, hold the lock, map a laya result to a Jev
   response.
   *Verify:* unit test maps a recorded laya result to the documented Jev shape.
4. **Application** — the three routes and the error handlers.
   *Verify:* `TestClient` tests with a fake engine pass. No model load needed.
5. **Command line** — argparse plus uvicorn.
   *Verify:* `--help` prints. The server starts and `/healthz` returns 200.
6. **Live check** — start the server, send the documented example requests from
   the API reference, then run the real `typesafe-sdk` and the `jev` package
   against it with `TYPESAFE_BASE_URL` set.
   *Verify:* every example returns the documented shape. The SDK parses each
   response and reads `.choice`, `.score` and `.noul`. `@jev.fn` returns a
   validated model.
7. **README** — install, run, configure, nginx example, fidelity limits.
   *Verify:* a reader can start the server from the README alone.
8. **Commit.**

## Out of scope

- Authentication, rate limiting and TLS. nginx handles these.
- Streaming. Jev does not stream.
- Batch or multi-model serving. One checkpoint per process.
- The `laya_mlx` Router, presets and email helpers.
