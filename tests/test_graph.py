"""Tests for KnowledgeGraph's Neo4j connection handling.

The driver is mocked throughout, so these run without a live database.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from neo4j.exceptions import (
    AuthError,
    ClientError,
    ConfigurationError,
    CypherSyntaxError,
    ServiceUnavailable,
)

from src.storage.graph import _MAX_RETRIES, KnowledgeGraph, _execute_with_retries


URI = "bolt://localhost:7687"


class _DriverFactory:
    """Hands out a fresh mock Driver per call, recording each one.

    ``verify_connectivity`` raises whatever the matching entry of `failures`
    says; ``None`` means the attempt succeeds.
    """

    def __init__(self, failures: list[BaseException | None]):
        self._failures = list(failures)
        self.drivers: list[MagicMock] = []

    def __call__(self, uri, auth=None, **kwargs):
        driver = MagicMock(name=f"driver-{len(self.drivers)}")
        outcome = (
            self._failures[len(self.drivers)]
            if len(self.drivers) < len(self._failures)
            else None
        )
        if outcome is not None:
            driver.verify_connectivity.side_effect = outcome
        self.drivers.append(driver)
        return driver


@pytest.fixture
def no_sleep():
    """Collapse the backoff so retry tests do not actually wait 7.5s."""
    with patch("src.storage.graph.time.sleep") as sleep:
        yield sleep


def _connect(factory) -> KnowledgeGraph:
    with patch("src.storage.graph.GraphDatabase") as gdb:
        gdb.driver.side_effect = factory
        return KnowledgeGraph(URI, "neo4j", "test")


class TestConnectRetries:
    def test_transient_failure_is_retried_until_success(self, no_sleep):
        factory = _DriverFactory([ServiceUnavailable("booting")] * 2)
        kg = _connect(factory)

        assert len(factory.drivers) == 3
        assert kg._driver is factory.drivers[-1]
        assert [c.args[0] for c in no_sleep.call_args_list] == [0.5, 1.0]

    def test_failed_attempts_do_not_leak_a_driver(self, no_sleep):
        """Each discarded driver owns a pool and threads; it must be closed."""
        factory = _DriverFactory([ServiceUnavailable("booting")] * 2)
        kg = _connect(factory)

        failed, succeeded = factory.drivers[:-1], factory.drivers[-1]
        assert len(failed) == 2
        for driver in failed:
            assert driver.close.called, "a discarded driver was never closed"
        assert not succeeded.close.called
        assert kg._driver is succeeded

    def test_exhausted_retries_close_every_driver_and_raise(self, no_sleep):
        factory = _DriverFactory([ServiceUnavailable("down")] * 5)

        with pytest.raises(RuntimeError) as excinfo:
            _connect(factory)

        assert URI in str(excinfo.value)
        assert isinstance(excinfo.value.__cause__, ServiceUnavailable)
        assert len(factory.drivers) == 5
        for driver in factory.drivers:
            assert driver.close.called, "the final attempt leaked its driver"

    def test_connectivity_is_verified_not_left_unconsumed(self):
        """verify_connectivity is the check; a lazy Result would prove nothing."""
        factory = _DriverFactory([])
        kg = _connect(factory)

        kg._driver.verify_connectivity.assert_called_once_with()
        assert not kg._driver.session.called


class TestNonRetryableFailures:
    @pytest.mark.parametrize(
        "error", [AuthError("bad password"), ConfigurationError("bad scheme")]
    )
    def test_misconfiguration_propagates_immediately(self, error, no_sleep):
        factory = _DriverFactory([error] * 5)

        with pytest.raises(type(error)):
            _connect(factory)

        assert len(factory.drivers) == 1, "startup misconfiguration was retried"
        assert not no_sleep.called
        assert factory.drivers[0].close.called


class TestExecuteWithRetries:
    """Covers the retry helper `KnowledgeGraph`'s query methods share."""

    def test_transient_failure_is_retried_until_success(self, no_sleep):
        calls = {"n": 0}

        def flaky():
            calls["n"] += 1
            if calls["n"] < 3:
                raise ServiceUnavailable("booting")
            return "ok"

        result = _execute_with_retries("query", flaky)

        assert result == "ok"
        assert calls["n"] == 3
        assert [c.args[0] for c in no_sleep.call_args_list] == [0.5, 1.0]

    def test_exhausted_retries_raise_with_original_exception_chained(self, no_sleep):
        cause = ServiceUnavailable("down")

        def always_fails():
            raise cause

        with pytest.raises(RuntimeError) as excinfo:
            _execute_with_retries("query", always_fails)

        assert "query" in str(excinfo.value)
        assert excinfo.value.__cause__ is cause
        assert no_sleep.call_count == _MAX_RETRIES - 1

    @pytest.mark.parametrize(
        "error",
        [
            AuthError("bad password"),
            ConfigurationError("bad scheme"),
            CypherSyntaxError("Invalid syntax"),
        ],
    )
    def test_non_transient_errors_propagate_without_sleeping(self, error, no_sleep):
        calls = {"n": 0}

        def fails_once():
            calls["n"] += 1
            raise error

        with pytest.raises(type(error)):
            _execute_with_retries("query", fails_once)

        assert calls["n"] == 1, "a non-transient error was retried"
        assert not no_sleep.called

    def test_client_error_subclasses_are_non_transient(self, no_sleep):
        """ClientError covers CypherSyntaxError/ConstraintError/Forbidden too."""

        def fails():
            raise ClientError("generic client error")

        with pytest.raises(ClientError):
            _execute_with_retries("query", fails)

        assert not no_sleep.called


class TestDriverAttributeIsNotOptional:
    """CI runs `mypy --ignore-missing-imports src` with no mypy config, so
    `union-attr` is on: an Optional `_driver` fails the typecheck job at every
    `self._driver.session()` call site."""

    def test_mypy_is_clean_on_the_storage_module(self):
        pytest.importorskip("mypy", reason="mypy is only installed in the typecheck job")
        repo_root = Path(__file__).resolve().parents[1]
        proc = subprocess.run(
            [sys.executable, "-m", "mypy", "--ignore-missing-imports", "src"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        assert "union-attr" not in proc.stdout, proc.stdout
        assert proc.returncode == 0, proc.stdout + proc.stderr

    def test_methods_use_the_driver_without_narrowing(self):
        kg = _connect(_DriverFactory([]))
        kg.close()
        assert kg._driver.close.called
