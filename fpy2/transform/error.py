"""
The failures of a rewrite, as part of its contract.

One hierarchy for every layer that rewrites: :mod:`fpy2.transform` and
:mod:`fpy2.rewrite` raise these directly and :mod:`fpy2.strategies` re-exports
them, so one `except` covers a strategy, a raw transform and a user rewrite
alike.
"""


class TransformError(Exception):
    """Base class of the hierarchy."""


class TransformDeclined(TransformError):
    """The transform refused to rewrite at a site it was aimed at.

    The program is unchanged, and the message says why the site was
    declined.  Recoverable: this is what a try/fallback strategy catches.
    """


class TransformReferenceError(TransformError):
    """A reference -- a `where`, a cursor, a path -- that names no such place."""
