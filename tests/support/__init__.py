"""Test-only orchestration and oracles for the Phase-4 vertical AD spike.

Nothing in this package is production code. In particular
:mod:`tests.support.reference_chain` is a float64 pure-Torch CPU oracle, which
CLAUDE.md permits only under ``tests/``; a production module that imported it
would be introducing a Torch numerical backend and is rejected by
``tests/test_phase4_import_boundary.py``.
"""
