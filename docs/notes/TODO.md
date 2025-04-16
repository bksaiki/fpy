Frontend:
 - fpc2fpy:
   - clean up names
   - preconditions with statements
 - parser:
   - fix ISSUE#1
   - use variables in the defining namespace
 - language:
   - support for zip()
   - support for vector addition/multiplication

Middle-end:
 - do we really need a separate AST / IR?

Backend:
 - fpc:
    - properties
    - fuse let expressions
    - detect multi-variable while loops
    - detect multi-variable for loops

Misc:
  - Rational contains a Fraction
  - decnum and hexnum conversion to Fraction

Numbers:
  - hashing by numerical equality (except NaN!)
