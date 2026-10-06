"""Send one small Chat Completions request to check the configured judge API."""
import argparse
import json
import os
import sys

import requests


DEFAULT_JUDGE_MODEL = "gpt-4o-2024-05-13"


def ping_api(api_key, api_base, model, timeout=30):
    """Return server-reported model metadata; never return credentials or URLs."""
    try:
        response = requests.post(
            api_base.rstrip("/") + "/chat/completions",
            headers={"Authorization": "Bearer " + api_key},
            json={"model": model, "messages": [{"role": "user", "content": "Reply with exactly: pong"}],
                  "temperature": 0, "max_tokens": 8, "stream": False},
            timeout=timeout,
        )
        response.raise_for_status()
        data = response.json()
        if not isinstance(data, dict) or not data.get("choices"):
            raise ValueError("Missing completion choices")
        returned_model = data.get("model")
        return {"ok": True, "requested_model": model, "returned_model": returned_model,
                "model_matches": returned_model == model,
                "reply": data["choices"][0].get("message", {}).get("content"),
                "usage": data.get("usage")}
    except requests.RequestException as exc:
        status = getattr(getattr(exc, "response", None), "status_code", None)
        raise RuntimeError("Judge API request failed" + (f" (HTTP {status})" if status else "")
                           + "; check API credentials, model availability and connectivity") from None
    except (ValueError, KeyError, TypeError, IndexError):
        raise RuntimeError("Judge API returned an invalid Chat Completions response") from None


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--api_key", default=os.environ.get("OPENAI_API_KEY"), help="Defaults to OPENAI_API_KEY")
    parser.add_argument("--api_base", default=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"))
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", DEFAULT_JUDGE_MODEL))
    parser.add_argument("--request_timeout", type=float, default=30)
    args = parser.parse_args(argv)
    if not args.api_key:
        parser.error("Set OPENAI_API_KEY or pass --api_key")
    if args.request_timeout <= 0:
        parser.error("--request_timeout must be positive")
    try:
        result = ping_api(args.api_key, args.api_base, args.model, args.request_timeout)
    except RuntimeError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
