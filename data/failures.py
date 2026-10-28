# SPDX-License-Identifier: GPL-3.0-or-later
# Copyright (C) 2026 Tim Rahlmeier

"""Count the records whose preprocessing raised, so that none is lost silently.

The strip cache builders (data/build_twolead_strip_cache.py,
data/wearable/build_mimic_leadI_strips.py, data/wearable/build_icentia_strips.py) and some
experiment scripts process records in worker processes. A worker returns
``{"error": "<type>: <message>"}`` instead of a result when its record raised
(``as_error``); the caller passes every result to ``Failures.add`` and calls
``Failures.check`` at the end, which prints the most common messages and stops when more than
``max_frac`` of the records raised. We stop at that point because a share that large points to
a broken pipeline (a missing package, a wrong path), not to bad records. A record that is
dropped by design (a gap, a flat lead) is not an error and is counted by the caller.
"""

from __future__ import annotations

from collections import Counter


def as_error(exc: BaseException) -> dict:
    """The result a worker returns for a record that raised ``exc``."""
    return {"error": f"{type(exc).__name__}: {exc}"}


def is_error(result) -> bool:
    """Whether ``result`` is the error result of ``as_error``."""
    return isinstance(result, dict) and "error" in result


class Failures:
    """The error messages of the records of one run whose preprocessing raised."""

    def __init__(self, label: str, max_frac: float = 0.01):
        self.label, self.max_frac, self.count = label, max_frac, Counter()

    def add(self, result) -> bool:
        """Count ``result`` if it is an error; return whether it was one."""
        if is_error(result):
            self.count[result["error"]] += 1
            return True
        return False

    def check(self, n_total: int) -> int:
        """Print the error count and the three most common messages; stop (SystemExit) if more
        than ``max_frac`` of ``n_total`` records raised. Return the number of records that
        raised. With ``n_total`` zero it never stops."""
        n = sum(self.count.values())
        print(f"[{self.label}] {n} of {n_total} records raised", flush=True)
        for msg, k in self.count.most_common(3):
            print(f"[{self.label}]   {k} x {msg}", flush=True)
        if n_total and n / n_total > self.max_frac:
            raise SystemExit(f"[{self.label}] {n} of {n_total} records raised; more than "
                             f"{self.max_frac:.0%} points to a broken pipeline, not to bad records")
        return n
