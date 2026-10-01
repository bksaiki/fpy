Core Semantics
======================

This page documents the semantics of FPy for a *core* subset of the language.
The :doc:`derived semantics <derived-semantics>` page covers semantics for
the full language.

It describes how FPy programs *evaluate*, and in particular how the *rounding
context* governs every arithmetic operation.
The rules follow the grammar: expressions, then statements, then programs.

Syntax
------

FPy's expressions are constants, arithmetic, comparisons, lists, records,
the context constructor, and dereference. Its statements are the usual
imperative ones—assignment, sequencing, conditionals, loops, return, assertion,
and skip—together with reference allocation, update, and function
application. One is unique to FPy: the *context statement*, which sets the
rounding context for the expressions it evaluates.

In the formal syntax, :math:`n` ranges over the reals together with
:math:`\pm\infty` and NaN, :math:`x` over a countable set of identifiers
:math:`\mathit{Var}`, :math:`l` over a countable set of labels
:math:`\mathit{Label}`, and :math:`f` over a separate set of function names
:math:`\mathit{FuncName}`. A record's labels are distinct, and records are
identified up to permutation of their fields.

.. math::

   \begin{array}{rcll}
   e & ::= & \mathsf{true} \mid \mathsf{false}
       & \text{boolean constants} \\
     & \mid & n
       & \text{numerical constants} \\
     & \mid & \R
       & \text{context literal} \\
     & \mid & x
       & \text{variable} \\
     & \mid & [\, e_1, \ldots, e_m \,]
       & \text{list constructor} \\
     & \mid & e_1[e_2]
       & \text{list indexing} \\
     & \mid & \{\, l_1 = e_1, \ldots, l_m = e_m \,\}
       & \text{record constructor} \\
     & \mid & e.l
       & \text{record projection} \\
     & \mid & \mathsf{ctx}\ e
       & \text{context constructor} \\
     & \mid & \mathsf{!}\, e
       & \text{dereference} \\
     & \mid & \mathit{op}(e_1, \ldots, e_k)
       & \text{operator application} \\[1ex]
   s & ::= & x = e
       & \text{assignment} \\
     & \mid & x = \mathsf{ref}\ e
       & \text{allocation} \\
     & \mid & x := e
       & \text{update} \\
     & \mid & x = f\ e
       & \text{function application} \\
     & \mid & s_1\, \mathsf{;}\, s_2
       & \text{sequencing} \\
     & \mid & \mathsf{if}\ e\ \mathsf{then}\ s_1\ \mathsf{else}\ s_2
       & \text{conditional} \\
     & \mid & \mathsf{while}\ e\ \mathsf{do}\ s
       & \text{loop} \\
     & \mid & \mathsf{ret}\ e
       & \text{return} \\
     & \mid & \mathsf{with}\ e\ \mathsf{as}\ x\ \mathsf{in}\ s
       & \text{context statement} \\
     & \mid & \mathsf{assert}\ e
       & \text{assertion} \\
     & \mid & \mathsf{skip}
       & \text{no-op} \\[1ex]
   \mathit{op} & ::= & + \mid - \mid \times \mid \div \mid \ldots
       & \mathit{Arith} \text{ operators} \\
     & \mid & < \mid \le \mid = \mid \mathit{len} \mid \mathit{isnan}
       \mid \ldots
       & \mathit{Exact} \text{ operators}
   \end{array}

The only context literal is :math:`\R`, the *real rounding context*, whose
rounding operation is the identity. Every other context is built by the
*context constructor* :math:`\mathsf{ctx}\ e` from a single record argument,
which alone determines the context's rounding operation.

Operators :math:`\mathit{op}` fall into one of two sets, according to what the
rounding context does to the result. An :math:`\mathit{op} \in \mathit{Arith}`
returns a number that is then rounded under a rounding context :math:`C`
(**E-Arith**); an :math:`\mathit{op} \in \mathit{Exact}` returns its result
without rounding (**E-Exact**).
Operators are written in prefix form even where FPy spells them infix, so
:math:`x < y` is :math:`\mathit{op}(x, y)`. The
:doc:`derived semantics <derived-semantics>` page enumerates FPy's operators.

Values
------

Evaluating an FPy expression produces one of six kinds of value: a boolean, a
number :math:`n`, a *rounding context* :math:`C`, a list of values, a record of
values, or a *location* :math:`\ell`.

.. math::

   \begin{array}{rcl}
   v & ::= & \mathsf{true} \mid \mathsf{false} \mid n \mid C
       \mid [\, v_1, \ldots, v_m \,]
       \mid \{\, l_1 = v_1, \ldots, l_m = v_m \,\} \mid \ell \\
   C & ::= & \R \mid \mathsf{ctx}\ \{\, l_1 = v_1, \ldots, l_m = v_m \,\}
   \end{array}

.. note::

   Two records are equal when they have the same labels and equal values at
   each label. Two constructed contexts are equal when their records are
   equal, so contexts that round alike but are built from different records
   are unequal. :math:`\R` is equal only to itself.

A *location* :math:`\ell` is the value of a reference. Locations are drawn from
a countable set :math:`\mathit{Loc}` and are used only by :math:`\mathsf{!}`
and :math:`:=`.

Expressions
-----------

Evaluation requires three things: a *store* :math:`\sigma` mapping identifiers
to values, a *heap* :math:`\mu` mapping locations to the values they currently
contain, and a *rounding context* :math:`C`. Both maps are finite and partial;
the rules state the memberships they need. An expression evaluates under all
three:

.. math::

   \langle \sigma, \mu, C, e \rangle \Downarrow v

read ":math:`e` evaluates to value :math:`v`". Expressions are pure;
:math:`\mu` remains an input because :math:`\mathsf{!}` reads it.

Where a premise cannot be met—an undefined lookup, a false side condition—no
rule applies and evaluation is stuck.

Values evaluate to themselves. A location is not an expression, so **E-Val**
applies only where a value can be written in a program: the boolean and
numerical constants and the context literal.

.. math::

   \frac{}{\langle \sigma, \mu, C, v \rangle \Downarrow v}
   \tag{E-Val}

Variables evaluate to their bound value.

.. math::

   \frac{x \in \mathrm{dom}(\sigma)}
        {\langle \sigma, \mu, C, x \rangle \Downarrow \sigma(x)}
   \tag{E-Var}

A list evaluates its elements; indexing selects one.

.. math::

   \frac{\langle \sigma, \mu, C, e_i \rangle \Downarrow v_i
         \quad (1 \le i \le m)}
        {\langle \sigma, \mu, C, [\, e_1, \ldots, e_m \,] \rangle \Downarrow
         [\, v_1, \ldots, v_m \,]}
   \tag{E-List}

.. math::

   \frac{\langle \sigma, \mu, C, e_1 \rangle \Downarrow [\, v_1, \ldots, v_m \,]
         \quad
         \langle \sigma, \mu, C, e_2 \rangle \Downarrow n
         \quad
         n \in \{ 0, \ldots, m-1 \}}
        {\langle \sigma, \mu, C, e_1[e_2] \rangle \Downarrow v_{n+1}}
   \tag{E-Index}

A record evaluates its fields; projection selects one by label.

.. math::

   \frac{\langle \sigma, \mu, C, e_i \rangle \Downarrow v_i
         \quad (1 \le i \le m)}
        {\langle \sigma, \mu, C, \{\, l_1 = e_1, \ldots, l_m = e_m \,\} \rangle
         \Downarrow \{\, l_1 = v_1, \ldots, l_m = v_m \,\}}
   \tag{E-Record}

.. math::

   \frac{\langle \sigma, \mu, C, e \rangle \Downarrow
         \{\, l_1 = v_1, \ldots, l_m = v_m \,\}
         \quad
         l = l_i}
        {\langle \sigma, \mu, C, e.l \rangle \Downarrow v_i}
   \tag{E-Proj}

Contexts are interpreted by a global, partial map :math:`\rho` from records to
rounding operations. It is fixed throughout evaluation: every judgement takes
it implicitly. The context constructor evaluates its argument to a record and
builds a context from it. The record must have an interpretation under
:math:`\rho`.

.. math::

   \frac{\langle \sigma, \mu, C, e \rangle \Downarrow v
         \quad
         v \in \mathrm{dom}(\rho)}
        {\langle \sigma, \mu, C, \mathsf{ctx}\ e \rangle \Downarrow
         \mathsf{ctx}\ v}
   \tag{E-Ctx}

We write :math:`C(\cdot)` for the rounding operation of a context :math:`C`:
the identity function for :math:`\R`, and :math:`\rho(v)` for
:math:`\mathsf{ctx}\ v`.

A reference is a mutable cell. Dereferencing reads the location's current value
from the heap; allocating the cell is a statement, since it writes one (see
**E-Ref**).

.. math::

   \frac{\langle \sigma, \mu, C, e \rangle \Downarrow \ell
         \quad
         \ell \in \mathrm{dom}(\mu)}
        {\langle \sigma, \mu, C, \mathsf{!}\, e \rangle \Downarrow \mu(\ell)}
   \tag{E-Deref}

An :math:`\mathit{Arith}` operator is where rounding happens. The brackets
:math:`\exact{\cdot}` mark a value computed exactly, with no intermediate
rounding, so :math:`\exact{\mathit{op}(v_1, \ldots, v_k)}` is the true result
and :math:`C` rounds it once.

.. math::

   \frac{\mathit{op} \in \mathit{Arith}
         \quad
         \langle \sigma, \mu, C, e_i \rangle \Downarrow v_i
         \quad (1 \le i \le k)}
        {\langle \sigma, \mu, C, \mathit{op}(e_1, \ldots, e_k) \rangle
         \Downarrow C(\exact{\mathit{op}(v_1, \ldots, v_k)})}
   \tag{E-Arith}

An :math:`\mathit{Exact}` operator applies to its operands as they are; nothing
rounds, and the result may be of any kind—a boolean from a comparison, an
integer from a length. NaN is unordered: an ordering test with a NaN operand is
false.

.. math::

   \frac{\mathit{op} \in \mathit{Exact}
         \quad
         \langle \sigma, \mu, C, e_i \rangle \Downarrow v_i
         \quad (1 \le i \le k)}
        {\langle \sigma, \mu, C, \mathit{op}(e_1, \ldots, e_k) \rangle
         \Downarrow \mathit{op}(v_1, \ldots, v_k)}
   \tag{E-Exact}

Statements
----------

A statement evaluates in the same state as an expression, but it may write the
heap, so its judgement yields a heap as well as a result:

.. math::

   \langle \sigma, \mu, C, s \rangle \Downarrow_S o \,;\, \mu'

read ":math:`s` evaluates to an *outcome* :math:`o`, leaving the heap
:math:`\mu'`". A statement either completes normally with an updated store
or returns a value, so an outcome is one of:

.. math::

   o ::= \mathsf{normal}\ \sigma \mid \mathsf{return}\ v

A :math:`\mathsf{normal}` outcome carries the store threaded to the next
statement; a :math:`\mathsf{return}` outcome carries a function's result and
short-circuits the rest of the body.

Assignment, allocation, update, application, skip, and a passing assertion
complete normally; :math:`\mathsf{ret}` returns. Sequencing, conditionals, loops,
and the context statement pass along the outcome of the sub-statement they run,
so a :math:`\mathsf{return}` propagates out to the enclosing function.

Assignment evaluates its right-hand side and binds :math:`x` to the value. It
copies nothing: if :math:`v` is a location, :math:`x` becomes a second name for
the same cell.

.. math::

   \frac{\langle \sigma, \mu, C, e \rangle \Downarrow v}
        {\langle \sigma, \mu, C, x = e \rangle \Downarrow_S
         \mathsf{normal}\ \sigma[x \mapsto v] \,;\, \mu}
   \tag{E-Assign}

An allocation statement creates a mutable cell: it picks a location not already
in use, stores :math:`e`'s value there, and binds :math:`x` to the location
itself, not to the value.

.. math::

   \frac{\langle \sigma, \mu, C, e \rangle \Downarrow v
         \quad
         \ell \notin \mathrm{dom}(\mu)}
        {\langle \sigma, \mu, C, x = \mathsf{ref}\ e \rangle \Downarrow_S
         \mathsf{normal}\ \sigma[x \mapsto \ell] \,;\, \mu[\ell \mapsto v]}
   \tag{E-Ref}

An update statement replaces a reference's value. The store is unchanged:
an update mutates only the heap, so every other name for that location observes
the write.

.. math::

   \frac{\sigma(x) = \ell
         \quad
         \langle \sigma, \mu, C, e \rangle \Downarrow v
         \quad
         \ell \in \mathrm{dom}(\mu)}
        {\langle \sigma, \mu, C, x := e \rangle \Downarrow_S
         \mathsf{normal}\ \sigma \,;\, \mu[\ell \mapsto v]}
   \tag{E-Update}

Functions live in a finite *function map* :math:`\Phi` from function names to
pairs :math:`(y, s)` of a parameter and a body. It is fixed
throughout evaluation: every judgement takes it implicitly, so the rules elide
it. Only **E-App** reads it, along with program entry below.

A function application looks its callee up in :math:`\Phi`, evaluates the
argument, and runs the body to the value it returns, binding that value to
:math:`x`. The body runs in a fresh store binding only the parameter, but
under the caller's context :math:`C`. Its outcome must be
:math:`\mathsf{return}\ v'`, so a body that completes normally is stuck. The
heap is *not* fresh: the body runs in the caller's heap and its writes outlive
the call, which is how a callee mutates a reference its caller holds.

.. math::

   \frac{\Phi(f) = (y, s)
         \quad
         \langle \sigma, \mu, C, e \rangle \Downarrow v
         \quad
         \langle [\, y \mapsto v \,], \mu, C, s \rangle \Downarrow_S
         \mathsf{return}\ v' \,;\, \mu'}
        {\langle \sigma, \mu, C, x = f\ e \rangle \Downarrow_S
         \mathsf{normal}\ \sigma[x \mapsto v'] \,;\, \mu'}
   \tag{E-App}

The skip statement does nothing; :math:`\mathsf{ret}` evaluates its operand and
returns it.

.. math::

   \frac{}{\langle \sigma, \mu, C, \mathsf{skip} \rangle \Downarrow_S
           \mathsf{normal}\ \sigma \,;\, \mu}
   \tag{E-Skip}

.. math::

   \frac{\langle \sigma, \mu, C, e \rangle \Downarrow v}
        {\langle \sigma, \mu, C, \mathsf{ret}\ e \rangle \Downarrow_S
         \mathsf{return}\ v \,;\, \mu}
   \tag{E-Ret}

An assertion evaluates its test; if it holds, evaluation continues with the
store unchanged. FPy has no error handling, so a failing assertion has no
rule and evaluation is stuck.

.. math::

   \frac{\langle \sigma, \mu, C, e \rangle \Downarrow \mathsf{true}}
        {\langle \sigma, \mu, C, \mathsf{assert}\ e \rangle \Downarrow_S
         \mathsf{normal}\ \sigma \,;\, \mu}
   \tag{E-Assert}

Sequencing runs :math:`s_1` first. If it returns, the sequence returns at once;
otherwise :math:`s_2` runs under the updated store and heap to produce
the sequence's outcome.

.. math::

   \frac{\langle \sigma, \mu, C, s_1 \rangle \Downarrow_S
         \mathsf{normal}\ \sigma' \,;\, \mu'
         \quad
         \langle \sigma', \mu', C, s_2 \rangle \Downarrow_S o \,;\, \mu''}
        {\langle \sigma, \mu, C, s_1\, \mathsf{;}\, s_2 \rangle \Downarrow_S o \,;\, \mu''}
   \tag{E-Seq-Normal}

.. math::

   \frac{\langle \sigma, \mu, C, s_1 \rangle \Downarrow_S \mathsf{return}\ v \,;\, \mu'}
        {\langle \sigma, \mu, C, s_1\, \mathsf{;}\, s_2 \rangle \Downarrow_S
         \mathsf{return}\ v \,;\, \mu'}
   \tag{E-Seq-Return}

A conditional evaluates its condition to a boolean and runs the matching
branch; the branch's outcome becomes the conditional's. Only the taken branch
touches the heap.

.. math::

   \frac{\langle \sigma, \mu, C, e \rangle \Downarrow \mathsf{true}
         \quad
         \langle \sigma, \mu, C, s_1 \rangle \Downarrow_S o \,;\, \mu'}
        {\langle \sigma, \mu, C, \mathsf{if}\ e\ \mathsf{then}\ s_1\ \mathsf{else}\ s_2 \rangle
         \Downarrow_S o \,;\, \mu'}
   \tag{E-If-True}

.. math::

   \frac{\langle \sigma, \mu, C, e \rangle \Downarrow \mathsf{false}
         \quad
         \langle \sigma, \mu, C, s_2 \rangle \Downarrow_S o \,;\, \mu'}
        {\langle \sigma, \mu, C, \mathsf{if}\ e\ \mathsf{then}\ s_1\ \mathsf{else}\ s_2 \rangle
         \Downarrow_S o \,;\, \mu'}
   \tag{E-If-False}

A loop tests its condition before each iteration. If the condition is false, the
loop completes with the store unchanged; if it holds, the loop runs its
body followed by the loop again. **E-Seq-Normal** then threads the body's
store and heap into the next iteration and **E-Seq-Return** carries a
:math:`\mathsf{ret}` in the body straight out of the enclosing function.

.. math::

   \frac{\langle \sigma, \mu, C, e \rangle \Downarrow \mathsf{false}}
        {\langle \sigma, \mu, C, \mathsf{while}\ e\ \mathsf{do}\ s \rangle
         \Downarrow_S \mathsf{normal}\ \sigma \,;\, \mu}
   \tag{E-While-False}

.. math::

   \frac{\langle \sigma, \mu, C, e \rangle \Downarrow \mathsf{true}
         \quad
         \langle \sigma, \mu, C,
           s\, \mathsf{;}\, \mathsf{while}\ e\ \mathsf{do}\ s \rangle
         \Downarrow_S o \,;\, \mu'}
        {\langle \sigma, \mu, C, \mathsf{while}\ e\ \mathsf{do}\ s \rangle
         \Downarrow_S o \,;\, \mu'}
   \tag{E-While-True}

.. note::

   These rules relate a loop only to a terminating run: a loop that never exits
   has no derivation.

The context statement is the heart of FPy. The context expression :math:`e` is
evaluated under :math:`\R` to a new context :math:`C'`, and the body :math:`s`
runs under :math:`C'` with :math:`x` bound to :math:`C'`, so it can refer to its
governing context as a value. :math:`C'` governs only the body—the
surrounding context :math:`C` is unchanged and still applies after the
``with``. The body's outcome becomes the statement's outcome.
The rounding context is scoped; the store and heap are not.

.. math::

   \frac{\langle \sigma, \mu, \R, e \rangle \Downarrow C'
         \quad
         \langle \sigma[x \mapsto C'], \mu, C', s \rangle \Downarrow_S o \,;\, \mu'}
        {\langle \sigma, \mu, C, \mathsf{with}\ e\ \mathsf{as}\ x\ \mathsf{in}\ s \rangle
         \Downarrow_S o \,;\, \mu'}
   \tag{E-Context}

.. note::

   The context expression is evaluated under :math:`\R` rather than the rounding
   context :math:`C` because the constructor's record fields are usually
   precisions, bitwidths, maximum values, etc. Rounding under :math:`C`
   may inadvertently change the desired result.

Programs
--------

A program is a pair :math:`(\Phi, f_{\mathit{main}})` of a function map and an
entry point, run on an argument :math:`v` supplied by the host. Where
:math:`\Phi(f_{\mathit{main}}) = (y, s)`, the program runs its body from the
initial state:

.. math::

   \langle [\, y \mapsto v \,], \emptyset, \R, s \rangle
   \Downarrow_S \mathsf{return}\ v' \,;\, \mu'

The program's result is :math:`v'`.
