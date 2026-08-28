Core Semantics
======================

This page documents the semantics of FPy. To stay tractable, it covers only
the *core* of the language. The :doc:`derived semantics <derived-semantics>`
page covers the full language.

It describes how FPy programs *evaluate*, and in particular how the *rounding
context* governs every arithmetic operation.
The rules follow the grammar: expressions, then statements, then programs.

Syntax
------

FPy's expressions are constants, arithmetic, comparisons, lists, tuples, and
dereference. Its statements are the usual imperative ones—assignment,
sequencing, conditionals, loops, return, assertion, and skip—together with
reference allocation, update, and function application. One is unique to FPy: the
*context statement*, which sets the rounding context for the expressions
it evaluates.

In the formal syntax, :math:`n` ranges over the reals together with
:math:`\pm\infty` and NaN, :math:`x` over a countable set of identifiers
:math:`\mathit{Var}`, and :math:`f` over a separate set of function names
:math:`\mathit{FuncName}`.

Operators :math:`\mathit{op}` fall into one of two sets. An
:math:`\mathit{op} \in \mathit{Arith}` takes numbers to a number, rounded under
:math:`C` (**E-Arith**); an :math:`\mathit{op} \in \mathit{Exact}` takes values
to a value, with :math:`C` playing no part (**E-Exact**). Neither set is fixed,
so a new operator needs no new rule.

There are two forms of context constant. :math:`\R` is the *real rounding
context*, whose rounding operation is the identity.
:math:`\mathsf{ctx}\ \{ \ldots \}` is a schema standing for every other context,
its :math:`\{ \ldots \}` the parameters that fix a rounding operation; distinct
parameters give distinct constants, so a program may use several at once.

.. math::

   \begin{array}{rcll}
   e & ::= & \mathsf{true} \mid \mathsf{false}
       & \text{boolean constants} \\
     & \mid & n
       & \text{numerical constants} \\
     & \mid & \R \mid \mathsf{ctx}\ \{ \ldots \}
       & \text{context constants} \\
     & \mid & x
       & \text{variable} \\
     & \mid & [\, e_1, \ldots, e_m \,]
       & \text{list constructor} \\
     & \mid & e_1[e_2]
       & \text{list indexing} \\
     & \mid & (\, e_1, \ldots, e_m \,)
       & \text{tuple} \\
     & \mid & \mathsf{!}\, e
       & \text{dereference} \\
     & \mid & \mathit{op}(e_1, \ldots, e_k)
       & \text{operator application} \\[1ex]
   s & ::= & p = e
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
   p & ::= & x
       & \text{variable pattern} \\
     & \mid & (\, p_1, \ldots, p_m \,)
       & \text{tuple pattern} \\[1ex]
   \mathit{op} & ::= & + \mid - \mid \times \mid \div \mid \ldots
       & \mathit{Arith} \text{ operators} \\
     & \mid & < \mid \le \mid = \mid \mathit{len} \mid \mathit{max}
       \mid \ldots
       & \mathit{Exact} \text{ operators}
   \end{array}

An assignment's left-hand side is a *pattern* :math:`p`—a variable or a
tuple of (possibly nested) patterns. A tuple pattern deconstructs a tuple
position by position, the only way to take a tuple apart.

Operators are written in prefix form even where FPy spells them infix, so
:math:`x < y` is :math:`\mathit{op}(x, y)`. The
:doc:`derived semantics <derived-semantics>` gives FPy's members of each set.

Values
------

Evaluating an FPy expression produces one of six kinds of value: a boolean, a
number :math:`n`, a *rounding context* :math:`C`, a list of values, a tuple of
values, or a *location* :math:`\ell`.

.. math::

   \begin{array}{rcl}
   v & ::= & \mathsf{true} \mid \mathsf{false} \mid n \mid C
       \mid [\, v_1, \ldots, v_m \,] \mid (\, v_1, \ldots, v_m \,) \mid \ell \\
   C & ::= & \R \mid \mathsf{ctx}\ \{ \ldots \}
   \end{array}

A rounding context :math:`C` is a context constant—:math:`\R`, or one of the
family :math:`\mathsf{ctx}\ \{ \ldots \}` schematizes—and is a value in its own
right (**E-Val**). A context is opaque: the semantics uses only its rounding
operation, written :math:`C(\cdot)`, whose result need not be finite.
Full FPy provides constructors for the common rounding contexts, all of which
the core abstracts as :math:`\mathsf{ctx}\ \{ \ldots \}`.

A *location* :math:`\ell` is the value of a reference. Locations are drawn from
a countable set
:math:`\mathit{Loc}` and are used only by :math:`\mathsf{!}` and :math:`:=`.

Expressions
-----------

Evaluation requires a *store* :math:`\sigma`, a finite map from identifiers to
values, a *heap* :math:`\mu`, a finite map from locations to the values they
currently contain, and a *rounding context* :math:`C`. Both maps are partial;
the rules state the memberships they need. An expression
evaluates under all three:

.. math::

   \langle \sigma, \mu, C, e \rangle \Downarrow v

read ":math:`e` evaluates to value :math:`v`". Expressions are pure;
:math:`\mu` remains an input because :math:`\mathsf{!}` reads it.

Where a premise cannot be met—an undefined lookup, a false side condition—no
rule applies and evaluation is stuck.


Values evaluate to themselves. A location is not an expression, so **E-Val**
applies only where a value can be written in a program: the boolean, numerical,
and context constants.

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

Tuples evaluate like lists.

.. math::

   \frac{\langle \sigma, \mu, C, e_i \rangle \Downarrow v_i
         \quad (1 \le i \le m)}
        {\langle \sigma, \mu, C, (\, e_1, \ldots, e_m \,) \rangle \Downarrow
         (\, v_1, \ldots, v_m \,)}
   \tag{E-Tuple}

A reference is a mutable cell. Dereferencing reads the location's current value
from the heap; allocating the cell is a statement, since it writes one (see
**E-Ref**).

.. math::

   \frac{\langle \sigma, \mu, C, e \rangle \Downarrow \ell
         \quad
         \ell \in \mathrm{dom}(\mu)}
        {\langle \sigma, \mu, C, \mathsf{!}\, e \rangle \Downarrow \mu(\ell)}
   \tag{E-Deref}

An :math:`\mathit{Arith}` operator is where rounding happens. Its operands
evaluate to numbers, and the rounding context :math:`C` rounds the exact result
of applying it. The brackets
:math:`\exact{\cdot}` mark a value computed exactly, with no intermediate
rounding, so :math:`\exact{\mathit{op}(n_1, \ldots, n_k)}` is the true result
and :math:`C` rounds it once. Under :math:`\R`, rounding is the identity, so the
exact result is returned unchanged.

.. math::

   \frac{\mathit{op} \in \mathit{Arith}
         \quad
         \langle \sigma, \mu, C, e_i \rangle \Downarrow n_i
         \quad (1 \le i \le k)}
        {\langle \sigma, \mu, C, \mathit{op}(e_1, \ldots, e_k) \rangle
         \Downarrow C(\exact{\mathit{op}(n_1, \ldots, n_k)})}
   \tag{E-Arith}

An :math:`\mathit{Exact}` operator applies to its operands as they are; nothing
rounds, and the result may be of any kind—a boolean from a comparison, an
integer from a length, an operand itself from a selection. NaN is unordered: an
ordering test with a NaN operand is false.

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

Only statements write the heap, and only allocation and update do. A rule with
two statement premises threads it from the first into the second. The heap is the
only thing that mutates, and its domain only grows: :math:`\sigma` changes only by
binding, while :math:`\mu` is global, shared by caller and callee, and never
deallocates.

Matching uses an auxiliary judgement :math:`p \triangleright v \Rightarrow \theta`,
read "pattern :math:`p` against value :math:`v` yields bindings :math:`\theta`".
Bindings combine by disjoint union :math:`\uplus`. Matching inspects a value
without allocating, so it needs no heap.

.. math::

   \frac{}{x \triangleright v \Rightarrow [\, x \mapsto v \,]}
   \tag{M-Var}

.. math::

   \frac{p_1 \triangleright v_1 \Rightarrow \theta_1
         \quad \cdots \quad
         p_m \triangleright v_m \Rightarrow \theta_m}
        {(\, p_1, \ldots, p_m \,) \triangleright (\, v_1, \ldots, v_m \,)
         \Rightarrow \theta_1 \uplus \cdots \uplus \theta_m}
   \tag{M-Tuple}

.. note::

   Because :math:`\uplus` is defined only on disjoint domains,
   a program with a pattern such as :math:`(x, x)` has no interpretation.

Assignment evaluates its right-hand side, matches the value against the
pattern, and extends the store with the bindings (:math:`\sigma[\theta]`
is :math:`\sigma` updated with every binding in :math:`\theta`). It copies
nothing: if :math:`v` is a location, the pattern's variable becomes a second
name for the same cell.

.. math::

   \frac{\langle \sigma, \mu, C, e \rangle \Downarrow v
         \quad
         p \triangleright v \Rightarrow \theta}
        {\langle \sigma, \mu, C, p = e \rangle \Downarrow_S
         \mathsf{normal}\ \sigma[\theta] \,;\, \mu}
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
   context :math:`C` because a constructor's arguments in the full FPy language
   are usually precisions, bitwidths, maximum values, etc. Rounding under :math:`C`
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

The program's result is :math:`v'`; the final heap :math:`\mu'` is discarded.
