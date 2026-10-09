"""Server-side AI crop question answering (Increment 9P).

This package feeds a crop question (plus optional latest-prediction context) to
the configured Groq chat-completions model and returns a grounded, bounded
answer. It contains:

- ``prompt``: a pure, deterministic prompt/context builder and input validators
  (no I/O, no network, no secrets);
- ``chat``: the Groq chat client (``httpx`` only) with the project's standard
  error mapping, secret redaction, and an injectable async client for tests.

The answer is never fabricated: disabled/misconfigured state and every upstream
failure surface through the shared ``ApiError`` envelope.
"""
