"""Out-of-band signal distinguishing an *explicit* deny from a *no-decision*.

Approval callbacks return a single string (``once`` / ``session`` / ``always``
/ ``deny``). That vocabulary cannot express *why* a deny happened, so a prompt
that timed out is indistinguishable from a user who actively pressed "deny".

For dangerous **commands** the distinction does not matter: both outcomes must
refuse execution, and they do. For **memory writes** it matters a great deal.
``tools.write_approval`` maps an explicit deny to ``blocked`` — the write is
discarded — while a no-decision must fall through to *staging* so the content
survives in ``/memory pending`` and can be replayed. Without this signal a
foreground memory write silently vanished when the approval prompt timed out
(ERRATUM-11): no store write, no pending record, no error the user could act on.

Callbacks therefore call :func:`mark_no_decision` immediately before returning
``"deny"`` for a reason that is *not* a human refusal (timeout, failed
scheduling, dropped response). Consumers that can stage instead of dropping
call :func:`consume_no_decision` right after invoking the callback.

Thread-local by construction: it mirrors the per-thread approval-callback
registry in ``tools.terminal_tool`` so concurrent ACP sessions, delegation
subagents and background forks never observe each other's signal.

This module is deliberately dependency-free (stdlib ``threading`` only) so it
can be imported from ``cli.py``, ``tools.approval``, ``acp_adapter`` and
``tools.write_approval`` without creating an import cycle.
"""

from __future__ import annotations

import threading

__all__ = [
    "TIMEOUT",
    "NO_CHANNEL",
    "TRANSPORT_ERROR",
    "mark_no_decision",
    "consume_no_decision",
    "clear_no_decision",
]

# Reason codes. Values are logged and surfaced in staged-record messages, so
# keep them short, stable and human-readable.
TIMEOUT = "timeout"
NO_CHANNEL = "no_channel"
TRANSPORT_ERROR = "transport_error"

_tls = threading.local()


def mark_no_decision(reason: str = TIMEOUT) -> None:
    """Flag that the deny about to be returned is *not* a human refusal.

    Call this immediately before ``return "deny"`` in an approval callback
    whenever no human actually decided — the prompt timed out, could not be
    scheduled, or the response was dropped in transit.
    """
    _tls.reason = reason or TIMEOUT


def consume_no_decision() -> str | None:
    """Return and clear the pending no-decision reason, if any.

    Single-shot on purpose: reading the flag clears it so a stale signal can
    never leak into a subsequent, unrelated approval and silently convert a
    genuine deny into a staged write.
    """
    reason = getattr(_tls, "reason", None)
    _tls.reason = None
    return reason


def clear_no_decision() -> None:
    """Drop any pending signal before issuing a fresh approval prompt.

    Callers MUST invoke this before prompting. It is the guard that keeps the
    gate fail-closed: without it, a no-decision left over from an earlier
    prompt on the same thread could be misread as belonging to the current one
    and turn an explicit user deny into a staged (i.e. recoverable) write.
    """
    _tls.reason = None
