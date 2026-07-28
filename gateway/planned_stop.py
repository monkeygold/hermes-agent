"""Mark a systemd-managed gateway stop as intentional before SIGTERM.

The generated unit runs this module from ``ExecStop=`` with systemd's
``$MAINPID``.  The helper only writes the existing short-lived, PID-reuse-safe
planned-stop marker; systemd remains responsible for signalling and stopping
the service.
"""

from __future__ import annotations

import sys
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 1:
        return 2
    try:
        target_pid = int(args[0])
    except (TypeError, ValueError):
        return 2
    if target_pid <= 0:
        return 2

    from gateway.status import write_planned_stop_marker

    return 0 if write_planned_stop_marker(target_pid) else 1


if __name__ == "__main__":
    raise SystemExit(main())
