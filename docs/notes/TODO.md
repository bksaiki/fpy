Frontend:
 - fpc2fpy:
   - clean up names
   - preconditions
   - preconditions with statements
 - parser:
   - fix ISSUE#1
   - underscore arguments to function
   - f-string
 - float:
   - context type parameter
 - arrays:
   - enumerate
 - language:
   - support for vector addition/multiplication
   - support "enumerate"
   - support "range" with start
 - formatter:
   - use IndentCtx

Middle-end:
 - passes:
  - dead code elimination
  - variable renaming
 - tests for each pass

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
