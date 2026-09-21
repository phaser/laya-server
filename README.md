# laya-server

A Jev-compatible HTTP server for [laya-mlx](https://github.com/mizorewww/laya-mlx).

It serves local typed decisions on Apple silicon behind the wire API of
[TypeSafe Jev](https://docs.typesafe.ai/api). A client that calls
`https://api.typesafe.ai` today works without a code change. Point
`TYPESAFE_BASE_URL` at this server.

```console
$ TYPESAFE_BASE_URL=http://127.0.0.1:8123 python my_existing_jev_app.py
```

The server has **no authentication**. Keep it on loopback, or put a reverse proxy
in front of it.

## Requirements

Apple silicon, macOS 14 or later, Python 3.11 or later.

## Install

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run

```bash
laya-server --port 8123
```

The first start downloads the checkpoint, about 600 MB. Later starts read the
Hugging Face cache. `GET /healthz` returns 200 when the model is ready.

| Option | Default | Purpose |
|---|---|---|
| `--host` | `127.0.0.1` | Bind address. |
| `--port` | `8000` | Bind port. |
| `--model` | `aac6fef/laya-mlx` | Checkpoint: a Hugging Face repository id or a local directory. See [Known models](#known-models). |
| `--dtype` | `float16` | `float16`, `float32` or `bfloat16`. |
| `--batch-size` | `16` | Questions per forward pass. |
| `--device` | MLX default | `gpu` or `cpu`. |
| `--max-questions` | `128` | Reject a request with more questions. |
| `--max-body-bytes` | `1048576` | Reject a larger request body. |

## Known models

`--model` accepts any Laya checkpoint that `laya-mlx` can load: a Hugging Face
repository id or a local directory. These are the checkpoints we know.

| Checkpoint | Encoder | Context | Languages | Note |
|---|---|---|---|---|
| [`aac6fef/laya-mlx`](https://huggingface.co/aac6fef/laya-mlx) | ModernBERT-large, 421M | 512 tokens | English | The default. MLX FP16 conversion of `convaiinnovations/laya`. |
| [`aac6fef/laya-multilingual-mlx`](https://huggingface.co/aac6fef/laya-multilingual-mlx) | mmBERT-base, 322M | 1024 tokens | 100 and more | MLX FP16 conversion of `convaiinnovations/laya-multilingual`. |
| [`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya) | ModernBERT-large, 421M | 512 tokens | English | Upstream weights. `laya-mlx` reads them directly. |
| [`convaiinnovations/laya-multilingual`](https://huggingface.co/convaiinnovations/laya-multilingual) | mmBERT-base, 322M | 1024 tokens | 100 and more | Upstream weights. |
| [`convaiinnovations/laya-typed-decisions`](https://huggingface.co/convaiinnovations/laya-typed-decisions) | ModernBERT-large, 421M | 1024 tokens | English | Fine-tuned on four synthetic workflows. Do not select it as a general default. |

```bash
laya-server --model aac6fef/laya-multilingual-mlx
laya-server --model convaiinnovations/laya-typed-decisions
laya-server --model ./my-local-checkpoint
```

Pick the multilingual checkpoint for non-English input. The English checkpoint
does not degrade gently off English. On 20-option intent classification it scores
0.100 on Hindi, against 0.050 for random choice, and it reports high confidence
while it does this.

The server loads one checkpoint per process. Run one process for each checkpoint
that you serve.

## API

### `POST /v1/systemone`

```bash
curl -s http://127.0.0.1:8123/v1/systemone \
  -H 'Content-Type: application/json' \
  -d '{
    "state": "Help! My payouts have been failing for 3 days.",
    "model": "jev-latest",
    "questions": {
      "is_urgent":   {"type": "noul",   "instructions": "Does this convey urgency?"},
      "department":  {"type": "choice", "instructions": "Which team should handle this?",
                      "criteria": {"billing": "Payments", "technical": "Bugs", "sales": "Pricing"}},
      "frustration": {"type": "score",  "instructions": "How frustrated is the customer?",
                      "criteria": ["Calm", "Frustrated", "Very angry"]}
    }
  }'
```

```json
{
  "model": "laya-mlx",
  "answers": {
    "is_urgent": {"type": "noul", "noul": 0.7786},
    "department": {"type": "choice", "choice": "billing",
                   "probabilities": {"billing": 0.8371, "technical": 0.0935, "sales": 0.0695},
                   "confidence": 0.4942},
    "frustration": {"type": "score", "score": 1.0894,
                    "legend": {"0": "Calm", "1": "Frustrated", "2": "Very angry"},
                    "probabilities": {"0": 0.0316, "1": 0.8475, "2": 0.1209},
                    "confidence": 0.5405}
  },
  "usage": {"input_tokens": 142, "output_tokens": 0}
}
```

The request and answer schemas are the ones in the
[Jev API reference](https://docs.typesafe.ai/api). The server validates the Jev
limits: 255 options for a choice, 2 to 10 levels for a score.

### `GET /v1/models`

Lists the names the `model` field accepts.

| Accepted `model` value | Meaning |
|---|---|
| `laya-mlx` | The canonical name. The response reports this name. |
| `jev-latest`, `jev-preview`, `jev-1.13.0` | Compatibility aliases. |
| the `--model` checkpoint id | The checkpoint that is loaded. |

Any other value returns 404. The server does not report itself as a Jev version,
because laya produced the answer.

### `GET /healthz`

Not part of Jev. It reports model readiness for a proxy or a start-up script.

### Errors

| Status | Cause |
|---|---|
| `404` | Unknown model name. |
| `413` | Request body over `--max-body-bytes`. |
| `422` | Body failed validation. The `detail` list names the offending field. |
| `500` | Inference failed. |

`401` is never returned. The server has no authentication and ignores the
`Authorization` header.

## Compatibility

Verified against the live server by `scripts/smoke.py`:

- The official `typesafe-sdk` Python client reads `.nouls`, `.choices`,
  `.scores`, `.confidence`, `.usage` and `models.list()`, and maps 404 to
  `TypeSafeNotFoundError`.
- The `jev` package works unchanged. Only `TYPESAFE_BASE_URL` changes:

  ```console
  $ TYPESAFE_API_KEY=unused TYPESAFE_BASE_URL=http://127.0.0.1:8123 python triage.py
  department='billing' is_urgent=True frustration=0
  ```

## Fidelity limits

The wire format matches. The model does not. Read this before you replace Jev in
production.

1. **Accuracy.** laya is a different model with different weights. Answers will
   not match Jev answer for answer.
2. **Context.** Jev accepts 64k tokens. The default English checkpoint accepts
   512 tokens in total, which includes the instructions and the options. laya
   truncates a longer state. The multilingual and typed-decisions checkpoints
   accept 1024.
3. **Confidence.** laya computes normalized Shannon entropy, `1 - H(p)/log(k)`.
   TypeSafe does not publish its formula. Both fall in the range 0 to 1 and both
   drop as the distribution flattens, but the numbers differ. Re-check every
   tuned threshold.
4. **`usage.output_tokens`** is always 0. laya decodes no tokens.
5. **Large choice sets.** Jev accepts 255 options. laya must fit every option in
   the token window, so a large set returns 422.
6. **Dropped fields.** laya returns `action.act_probability` on every answer and
   `confidence` on a noul answer. Jev defines neither, so the server removes
   them. Call `laya_mlx` directly if you want them.

## Behind nginx

```nginx
server {
    listen 443 ssl;
    server_name laya.example.com;

    ssl_certificate     /etc/ssl/certs/laya.crt;
    ssl_certificate_key /etc/ssl/private/laya.key;

    client_max_body_size 1m;

    location /v1/ {
        auth_basic           "laya";
        auth_basic_user_file /etc/nginx/laya.htpasswd;

        proxy_pass         http://127.0.0.1:8123;
        proxy_read_timeout 120s;
    }

    location = /healthz {
        proxy_pass http://127.0.0.1:8123;
    }
}
```

Keep `--host 127.0.0.1`. The proxy must terminate TLS and must authenticate every
caller, because the application does not.

## Development

```bash
pip install -e ".[dev]"
pytest                              # unit tests, no checkpoint needed
laya-server --port 8123 &           # then, against the running server:
python scripts/smoke.py http://127.0.0.1:8123
```

`scripts/smoke.py` also exercises the official SDK when `typesafe-sdk` is
installed.

## Design

| File | Role |
|---|---|
| `src/laya_server/schemas.py` | Request and response models. Jev shapes and limits. |
| `src/laya_server/engine.py` | Checkpoint load, inference lock, laya-to-Jev mapping. |
| `src/laya_server/app.py` | Routes, guards, error handlers. |
| `src/laya_server/__main__.py` | Command line, uvicorn start. |

One checkpoint runs per process. MLX holds a single device context, so a lock
serializes inference. Route handlers are synchronous, so FastAPI runs them in the
thread pool and the event loop stays free.

See `PLAN.md` for the design decisions and their sources.

## License

ISC. See [LICENSE](LICENSE).

Dependencies keep their own licenses: laya-mlx is Apache-2.0, FastAPI and
typesafe-sdk are MIT, uvicorn is BSD-3-Clause. laya-mlx is an independent MLX port of
[Convai Innovations' Laya](https://github.com/NandhaKishorM/laya), not an official
release. This project is not affiliated with TypeSafe AI.
