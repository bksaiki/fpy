Frontend:
 - fpc2fpy:
   - clean up names
   - preconditions with statements
 - parser:
   - fix ISSUE#1
   - underscore arguments to function
   - f-string
 - language:
   - support for zip()
   - support for vector addition/multiplication
 - formatter:
   - use IndentCtx

Middle-end:
 - do we really need a separate AST / IR?
 - String IR node
 - passes:
  - copy propagation
  - eliminate unneeded phi variables
  - dead code elimination
  - variable renaming

Backend:
 - fpc:
    - properties
    - fuse let expressions
    - detect multi-variable while loops
    - detect multi-variable for loops

Misc:
 - Rational contains a Fraction
 - decnum and hexnum conversion to Fraction

Interpreters:
 - integrate FPCoreContext into other interpreters

Numbers:
  - hashing by numerical equality (except NaN!)

Documentation:
  - prune unneccesary documentation
  - integrate config with pyproject.toml
