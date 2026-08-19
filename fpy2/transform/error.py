"""
The errors a transform raises as part of its contract.

One hierarchy serves both layers: the transforms raise these directly, and
:mod:`fpy2.strategies` re-exports them, so a caller catches the same
exception whether it applied a strategy or a raw transform.
"""


class TransformError(Exception):
    """Base class of the hierarchy."""


class TransformDeclined(TransformError):
    """The transform refused to rewrite at a site it was aimed at.

    The program is unchanged, and the message says why the site was
    declined.  Recoverable: this is what a try/fallback strategy catches.
    """


class TransformReferenceError(TransformError):
    """A `where` that named no candidate site."""
