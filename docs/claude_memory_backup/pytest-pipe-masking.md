---
name: pytest-pipe-masking
description: never gate a commit on `pytest | tail` — the pipe masks the exit code; use a file + grep " failed"
metadata:
  type: feedback
---

`python -m pytest ... | tail -1 && git commit ...` commits even when tests fail (the pipe's exit
status is tail's). This bit twice in this project (a failing test was committed on 2026-09-02, and
again on 2026-09-03 with test_kde).

**Why:** the user's discipline is "tests green before commit/deploy"; a masked failure silently
ships broken invariants to hyde06.

**How to apply:** `python -m pytest tests/ -q > /tmp/pt.txt 2>&1; tail -1 /tmp/pt.txt; grep -q " failed"
/tmp/pt.txt && echo FAIL || (git add ... && git commit ...)` — or `set -o pipefail`.
