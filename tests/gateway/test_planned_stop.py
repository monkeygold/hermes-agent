"""Regression tests for the systemd ExecStop planned-stop marker helper."""

from __future__ import annotations


def test_main_marks_the_exact_systemd_main_pid(monkeypatch):
    from gateway import planned_stop

    marked: list[int] = []
    monkeypatch.setattr(
        "gateway.status.write_planned_stop_marker",
        lambda pid: marked.append(pid) or True,
    )

    assert planned_stop.main(["4242"]) == 0
    assert marked == [4242]


def test_main_rejects_missing_or_invalid_pid(monkeypatch):
    from gateway import planned_stop

    marked: list[int] = []
    monkeypatch.setattr(
        "gateway.status.write_planned_stop_marker",
        lambda pid: marked.append(pid) or True,
    )

    assert planned_stop.main([]) == 2
    assert planned_stop.main(["not-a-pid"]) == 2
    assert planned_stop.main(["0"]) == 2
    assert marked == []


def test_main_reports_marker_write_failure(monkeypatch):
    from gateway import planned_stop

    monkeypatch.setattr(
        "gateway.status.write_planned_stop_marker",
        lambda pid: False,
    )

    assert planned_stop.main(["4242"]) == 1
