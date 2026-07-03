"""Test package for evalpipe.

Being a real package lets test modules import shared helpers as
``tests.conftest`` regardless of how pytest is invoked (``pytest`` puts the
rootdir on ``sys.path`` for package-based collection; a bare directory only
works when the current directory happens to be importable).
"""
