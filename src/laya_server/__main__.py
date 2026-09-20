"""Start the Jev-compatible laya server."""

import argparse
import logging

import uvicorn

from .app import DEFAULT_MAX_BODY_BYTES, DEFAULT_MAX_QUESTIONS, create_app
from .engine import Engine

DEFAULT_CHECKPOINT = "aac6fef/laya-mlx"

log = logging.getLogger("laya_server")


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(prog="laya-server", description=__doc__)
    parser.add_argument(
        "--host",
        default="127.0.0.1",
        help="Bind address. The server has no authentication. Keep it on loopback "
        "and put a reverse proxy in front of it.",
    )
    parser.add_argument("--port", type=int, default=8000, help="Bind port.")
    parser.add_argument(
        "--model",
        default=DEFAULT_CHECKPOINT,
        help="Checkpoint to serve: a Hugging Face repository id or a local directory.",
    )
    parser.add_argument(
        "--dtype", choices=("float16", "float32", "bfloat16"), default="float16"
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Questions per forward pass."
    )
    parser.add_argument("--device", choices=("gpu", "cpu"), default=None)
    parser.add_argument(
        "--max-questions",
        type=int,
        default=DEFAULT_MAX_QUESTIONS,
        help="Reject a request that asks more questions than this.",
    )
    parser.add_argument(
        "--max-body-bytes",
        type=int,
        default=DEFAULT_MAX_BODY_BYTES,
        help="Reject a request body larger than this.",
    )
    return parser.parse_args(argv)


def main(argv=None) -> None:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s"
    )
    if args.host not in ("127.0.0.1", "::1", "localhost"):
        log.warning(
            "Binding to %s. This server has no authentication. Restrict access at "
            "the network or the reverse proxy.",
            args.host,
        )
    log.info("Loading checkpoint %s (%s)", args.model, args.dtype)
    engine = Engine(
        args.model,
        dtype=args.dtype,
        batch_size=args.batch_size,
        device=args.device,
    )
    log.info("Checkpoint ready. Serving as %s", engine.model_name)
    app = create_app(
        engine,
        max_questions=args.max_questions,
        max_body_bytes=args.max_body_bytes,
    )
    uvicorn.run(app, host=args.host, port=args.port, log_config=None)


if __name__ == "__main__":
    main()
