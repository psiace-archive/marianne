# tests/conftest.py

from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent


def pytest_sessionfinish(session, exitstatus):
    # if all the tests are skipped, lets not fail the entire CI run
    if exitstatus == 5:
        session.exitstatus = 0
