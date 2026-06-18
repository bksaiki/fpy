from sphinx_pyproject import SphinxConfig

config = SphinxConfig('../../pyproject.toml', globalns=globals())

# MathJax macros for the formal semantics (see dev/semantics).
#   \R          -- the real rounding context (identity rounding)
#   \exact{...} -- the exact real-number value of an expression, before rounding
mathjax3_config = {
    'tex': {
        'macros': {
            'R': r'\mathbb{R}',
            'exact': [r'[\![ #1 ]\!]', 1],
        },
    },
}
