"""Tests for KnowledgeGraph's Neo4j connection handling.

The driver is mocked throughout, so these run without a live database.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from neo4j.exceptions import AuthError, ConfigurationError, ServiceUnavailable

from src.storage.graph import KnowledgeGraph


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
