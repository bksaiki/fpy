Frontend:
 - fpc2fpy:
   - clean up names
   - preconditions
   - preconditions with statements
 - parser:
   - fix ISSUE#1
   - underscore arguments to function
   - f-string
 - language:
   - support for vector addition/multiplication
 - formatter:
   - use IndentCtx

Middle-end:
 - passes:
  - dead code elimination
  - variable renaming

Backend:
 - fpc:
    - properties
    - fuse let expressions
    - detect multi-variable while loops
    - detect multi-variable for loops

Misc:
 - fma for RealContext
 - separable mathematical core

Interpreters:
 - integrate FPCoreContext into other interpreters

Numbers:
  - hashing by numerical equality (except NaN!)

Documentation:
  - integrate config with pyproject.toml
