"""``Location.format`` is a message fragment, so its quoting is a contract.

Two diagnostics embed it and add punctuation around it:
``CppEmitError`` prepends it to the message, and ``ReachabilityError`` wraps it
in ``at <loc>: `stmt```.  It shipped for a while opening a backtick it never
closed, which left every FPy error message with an unbalanced quote.
"""

from fpy2.utils.location import Location

LOC = Location('foo.py', 11, 13, 11, 20)


def test_format_names_the_source_and_the_start_position():
    assert LOC.format() == '`foo.py:11:13`'


def test_format_is_balanced():
    """The property the bug violated.  Both callers add their own delimiters
    around it, so an unclosed quote there is an unclosed quote in the whole
    message."""
    assert LOC.format().count('`') % 2 == 0


def test_format_reads_as_a_prefix():
    """How ``CppEmitError`` uses it -- the shape a user actually sees."""
    assert f'{LOC.format()}: unsupported' == '`foo.py:11:13`: unsupported'
