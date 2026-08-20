Language Semantics
======================

This page documents the semantics of FPy. To stay tractable, it covers only
the *core* of the language. The semantics for the full language can
be found in the :doc:`derived semantics <derived-semantics>` page.

It describes how FPy programs *evaluate*, and in particular how the *active
rounding context* governs every arithmetic operation.
The rules follow the grammar: expressions, then statements, then programs.

Syntax
------

FPy's expressions are constants, arithmetic, comparisons, lists, tuples, and
dereference. Its statements are the usual imperative ones—assignment,
sequencing, conditionals, loops, return, assertion, and skip—together with
reference allocation, update, and function application. One is unique to FPy: the
*context statement*, which sets the active rounding context for the expressions
it evaluates.

In the formal syntax, :math:`n` ranges over the reals together with
:math:`\pm\infty` and NaN, :math:`x` over a countable set of identifiers
:math:`\mathit{Var}`, and :math:`f` over a separate set of function names
:math:`\mathit{FuncName}`.

There are two forms of context constant. :math:`\R` is the *real rounding
context*, whose rounding operation is the identity.
:math:`\texttt{ctx}\ \{ \ldots \}` is a schema standing for every other context,
its :math:`\{ \ldots \}` the parameters that fix a rounding operation; distinct
parameters give distinct constants, so a program may use several at once.

.. math::

   \begin{array}{rcll}
   e & ::= & \texttt{true} \mid \texttt{false}
       & \text{boolean constants} \\
     & \mid & n
       & \text{numerical constants} \\
     & \mid & \R \mid \texttt{ctx}\ \{ \ldots \}
       & \text{context constants} \\
     & \mid & x
       & \text{variable} \\
     & \mid & [\, e_1, \ldots, e_m \,]
       & \text{list constructor} \\
     & \mid & e_1[e_2]
       & \text{list indexing} \\
     & \mid & (\, e_1, \ldots, e_m \,)
       & \text{tuple} \\
     & \mid & \texttt{!}\, e
       & \text{dereference} \\
     & \mid & e_1 + e_2
       & \text{arithmetic} \\
     & \mid & e_1 < e_2
       & \text{comparison} \\[1ex]
   s & ::= & p = e
       & \text{assignment} \\
     & \mid & x = \texttt{ref}\ e
       & \text{allocation} \\
     & \mid & x := e
       & \text{update} \\
     & \mid & x = f\ e
       & \text{function application} \\
     & \mid & s_1\, \texttt{;}\, s_2
       & \text{sequencing} \\
     & \mid & \texttt{if}\ e\ \texttt{then}\ s_1\ \texttt{else}\ s_2
       & \text{conditional} \\
     & \mid & \texttt{while}\ e\ \texttt{do}\ s
       & \text{loop} \\
     & \mid & \texttt{ret}\ e
       & \text{return} \\
     & \mid & \texttt{with}\ e\ \texttt{as}\ x\ \texttt{in}\ s
       & \text{context statement} \\
     & \mid & \texttt{assert}\ e
       & \text{assertion} \\
     & \mid & \texttt{skip}
       & \text{no-op} \\[1ex]
   p & ::= & x
       & \text{variable pattern} \\
     & \mid & (\, p_1, \ldots, p_m \,)
       & \text{tuple pattern}
   \end{array}

An assignment's left-hand side is a *pattern* :math:`p`—a variable or a
tuple of (possibly nested) patterns. A tuple pattern deconstructs a tuple
position by position, the only way to take a tuple apart.

``+`` and ``<`` stand in for arithmetic and comparison in general; every other
FPy operator evaluates the same way, though comparison and classification
operators yield booleans rather than rounded reals.

Values
------

Evaluating an FPy expression produces one of six kinds of value: a boolean, a
number :math:`n`, a *rounding context* :math:`C`, a list of values, a tuple of
values, or a *location* :math:`\ell`.

.. math::

   \begin{array}{rcl}
   v & ::= & \texttt{true} \mid \texttt{false} \mid n \mid C
       \mid [\, v_1, \ldots, v_m \,] \mid (\, v_1, \ldots, v_m \,) \mid \ell \\
   C & ::= & \R \mid \texttt{ctx}\ \{ \ldots \}
   \end{array}

A rounding context :math:`C` is a context constant—:math:`\R`, or one of the
family :math:`\texttt{ctx}\ \{ \ldots \}` schematizes—and is a value in its own
right (**E-Val**). A context is opaque: the semantics uses only its rounding
operation, written :math:`C(\cdot)`, whose result need not be finite.
Full FPy provides constructors for the common rounding contexts, all of which
the core abstracts as :math:`\texttt{ctx}\ \{ \ldots \}`.

A *location* :math:`\ell` is the value of a reference: the allocation statement
:math:`x = \texttt{ref}\ e` binds :math:`x` to a location whose store entry
holds :math:`e`'s value. Locations are drawn from a countable set
:math:`\mathit{Loc}` and are used only by :math:`\texttt{!}` and :math:`:=`.

Expressions
-----------

Evaluation requires an environment :math:`\sigma`, a finite map from identifiers
to values, a *store* :math:`\mu`, a finite map from locations to the values they
currently contain, and an *active rounding context* :math:`C`. An expression
evaluates under all three:

.. math::

   \langle \sigma, \mu, C, e \rangle \Downarrow v

read ":math:`e` evaluates to value :math:`v`". Expressions are pure, so the
judgement yields no store; :math:`\mu` remains an input because
:math:`\texttt{!}` reads it. The order in which a rule
evaluates its sub-expressions is therefore unobservable.

A rule that writes a lookup, such as :math:`\sigma(x)` or :math:`\mu(\ell)`,
requires it to be defined. Where it is not, no rule applies and evaluation is
stuck.

The active rounding context :math:`C` is the crux of FPy's semantics: it is
threaded through every expression and rounds the exact result of each
arithmetic operation (see **E-Add**).

Values evaluate to themselves.

.. math::

   \frac{}{\langle \sigma, \mu, C, v \rangle \Downarrow v}
   \tag{E-Val}

Variables evaluate to their bound value.

.. math::

   \frac{}{\langle \sigma, \mu, C, x \rangle \Downarrow \sigma(x)}
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

A reference is a mutable cell. Dereferencing reads the value its location
currently contains out of the store; allocating the cell is a statement, since
it writes one (see **E-Ref**).

.. math::

   \frac{\langle \sigma, \mu, C, e \rangle \Downarrow \ell}
        {\langle \sigma, \mu, C, \texttt{!}\, e \rangle \Downarrow \mu(\ell)}
   \tag{E-Deref}

Arithmetic is where rounding happens. The operands evaluate to numbers, and the
active context :math:`C` rounds their sum. The brackets :math:`\exact{\cdot}`
mark a value computed exactly, with no intermediate rounding, so
:math:`\exact{n_1 + n_2}` is the true sum and :math:`C` rounds it once. Under
:math:`\R`, rounding is the identity, so the exact result is returned unchanged;
under a finite context, an out-of-range result rounds to :math:`\pm\infty`.

.. math::

   \frac{\langle \sigma, \mu, C, e_1 \rangle \Downarrow n_1
         \quad
         \langle \sigma, \mu, C, e_2 \rangle \Downarrow n_2}
        {\langle \sigma, \mu, C, e_1 + e_2 \rangle \Downarrow
         C(\exact{n_1 + n_2})}
   \tag{E-Add}

A comparison evaluates its operands and tests them, producing a boolean;
nothing rounds. NaN is unordered: an ordering test with a NaN operand is false.
``<`` is the representative—the other comparisons behave identically.

.. math::

   \frac{\langle \sigma, \mu, C, e_1 \rangle \Downarrow n_1
         \quad
         \langle \sigma, \mu, C, e_2 \rangle \Downarrow n_2}
        {\langle \sigma, \mu, C, e_1 < e_2 \rangle \Downarrow (n_1 < n_2)}
   \tag{E-Lt}

Statements
----------

A statement evaluates in the same state as an expression, but it may write the
store, so its judgement yields a store as well as a result:

.. math::

   \langle \sigma, \mu, C, s \rangle \Downarrow_S o \,;\, \mu'

read ":math:`s` evaluates to an *outcome* :math:`o`, leaving the store
:math:`\mu'`". A statement either completes normally with an updated environment
or returns a value, so an outcome is one of:

.. math::

   o ::= \mathsf{normal}\ \sigma \mid \mathsf{return}\ v

A :math:`\mathsf{normal}` outcome carries the environment threaded to the next
statement; a :math:`\mathsf{return}` outcome carries a function's result and
short-circuits the rest of the body.

Assignment, allocation, update, application, skip, and a passing assertion
complete normally and :math:`\texttt{ret}` returns; sequencing, conditionals,
loops, and the context statement pass along the outcome of whatever
sub-statement they run,
so a :math:`\mathsf{return}` propagates out to the enclosing function.

Only statements write the store, and only allocation and update do. A rule with
two statement premises threads it from the first into the second. A prime names what a
premise leaves behind, as in :math:`\mu'` and :math:`\sigma'`, and a rule that
threads two stores in sequence names the second :math:`\mu''`. The store is the
only thing that mutates, and its domain only grows: :math:`\sigma` changes only by
binding, while :math:`\mu` is global, shared by caller and callee, and never
deallocates.

Matching uses an auxiliary judgement :math:`p \triangleright v \Rightarrow \theta`,
read "pattern :math:`p` against value :math:`v` yields bindings :math:`\theta`".
A variable matches anything and binds it; a tuple pattern matches a tuple
position by position, combining the per-component bindings by disjoint union
:math:`\uplus`. Matching inspects a value without allocating, so it needs no
store.

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
pattern, and extends the environment with the bindings (:math:`\sigma[\theta]`
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
        {\langle \sigma, \mu, C, x = \texttt{ref}\ e \rangle \Downarrow_S
         \mathsf{normal}\ \sigma[x \mapsto \ell] \,;\, \mu[\ell \mapsto v]}
   \tag{E-Ref}

An update statement replaces the value contained by a reference.
The environment is unchanged—an update mutates the store and only the store,
which is why every other name for that location observes the write.

.. math::

   \frac{\sigma(x) = \ell
         \quad
         \langle \sigma, \mu, C, e \rangle \Downarrow v
         \quad
         \ell \in \mathrm{dom}(\mu)}
        {\langle \sigma, \mu, C, x := e \rangle \Downarrow_S
         \mathsf{normal}\ \sigma \,;\, \mu[\ell \mapsto v]}
   \tag{E-Update}

Functions live in a *top-level environment* :math:`\Phi`, a finite map from
function names to pairs :math:`(y, s)` of a parameter and a body. It is fixed
throughout evaluation: every judgement is implicitly parameterized by it, so the
rules elide it. Only **E-App** reads it, along with program entry below.

A function application looks its callee up in :math:`\Phi`, evaluates the
argument, and runs the body to the value it returns, binding that value to
:math:`x`. The body runs in a fresh environment binding only the parameter, but
under the caller's context :math:`C`, and a well-formed body always returns, so
its outcome is :math:`\mathsf{return}\ v'`. The store is *not* fresh: the body
runs in the caller's store and its writes outlive the call, which is how a callee
mutates a reference its caller holds.

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

The skip statement does nothing; :math:`\texttt{ret}` evaluates its operand and
returns it.

.. math::

   \frac{}{\langle \sigma, \mu, C, \texttt{skip} \rangle \Downarrow_S
           \mathsf{normal}\ \sigma \,;\, \mu}
   \tag{E-Skip}

.. math::

   \frac{\langle \sigma, \mu, C, e \rangle \Downarrow v}
        {\langle \sigma, \mu, C, \texttt{ret}\ e \rangle \Downarrow_S
         \mathsf{return}\ v \,;\, \mu}
   \tag{E-Ret}

An assertion evaluates its test; if it holds, evaluation continues with the
environment unchanged. FPy has no error handling, so a failing assertion has no
rule and evaluation is stuck.

.. math::

   \frac{\langle \sigma, \mu, C, e \rangle \Downarrow \texttt{true}}
        {\langle \sigma, \mu, C, \texttt{assert}\ e \rangle \Downarrow_S
         \mathsf{normal}\ \sigma \,;\, \mu}
   \tag{E-Assert}

Sequencing runs :math:`s_1` first. If it returns, the sequence returns at once;
otherwise :math:`s_2` runs under the updated environment and store to produce
the sequence's outcome.

.. math::

   \frac{\langle \sigma, \mu, C, s_1 \rangle \Downarrow_S
         \mathsf{normal}\ \sigma' \,;\, \mu'
         \quad
         \langle \sigma', \mu', C, s_2 \rangle \Downarrow_S o \,;\, \mu''}
        {\langle \sigma, \mu, C, s_1\, \texttt{;}\, s_2 \rangle \Downarrow_S o \,;\, \mu''}
   \tag{E-Seq-Normal}

.. math::

   \frac{\langle \sigma, \mu, C, s_1 \rangle \Downarrow_S \mathsf{return}\ v \,;\, \mu'}
        {\langle \sigma, \mu, C, s_1\, \texttt{;}\, s_2 \rangle \Downarrow_S
         \mathsf{return}\ v \,;\, \mu'}
   \tag{E-Seq-Return}

A conditional evaluates its condition to a boolean and runs the matching
branch; the branch's outcome becomes the conditional's, so a :math:`\texttt{ret}`
in either branch returns from the enclosing function. Only the taken branch
touches the store.

.. math::

   \frac{\langle \sigma, \mu, C, e \rangle \Downarrow \texttt{true}
         \quad
         \langle \sigma, \mu, C, s_1 \rangle \Downarrow_S o \,;\, \mu'}
        {\langle \sigma, \mu, C, \texttt{if}\ e\ \texttt{then}\ s_1\ \texttt{else}\ s_2 \rangle
         \Downarrow_S o \,;\, \mu'}
   \tag{E-If-True}

.. math::

   \frac{\langle \sigma, \mu, C, e \rangle \Downarrow \texttt{false}
         \quad
         \langle \sigma, \mu, C, s_2 \rangle \Downarrow_S o \,;\, \mu'}
        {\langle \sigma, \mu, C, \texttt{if}\ e\ \texttt{then}\ s_1\ \texttt{else}\ s_2 \rangle
         \Downarrow_S o \,;\, \mu'}
   \tag{E-If-False}

A loop tests its condition before each iteration. If the condition is false the
loop completes with the environment unchanged; if it holds, the loop runs its
body followed by the loop again, so **E-Seq-Normal** threads the body's
environment and store into the next iteration and **E-Seq-Return** carries a
:math:`\texttt{ret}` in the body straight out of the enclosing function.

.. math::

   \frac{\langle \sigma, \mu, C, e \rangle \Downarrow \texttt{false}}
        {\langle \sigma, \mu, C, \texttt{while}\ e\ \texttt{do}\ s \rangle
         \Downarrow_S \mathsf{normal}\ \sigma \,;\, \mu}
   \tag{E-While-False}

.. math::

   \frac{\langle \sigma, \mu, C, e \rangle \Downarrow \texttt{true}
         \quad
         \langle \sigma, \mu, C,
           s\, \texttt{;}\, \texttt{while}\ e\ \texttt{do}\ s \rangle
         \Downarrow_S o \,;\, \mu'}
        {\langle \sigma, \mu, C, \texttt{while}\ e\ \texttt{do}\ s \rangle
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
``with``. The body's outcome becomes the statement's outcome, so a
:math:`\texttt{ret}` inside a ``with`` returns from the enclosing function.
The context is scoped; the store is not.

.. math::

   \frac{\langle \sigma, \mu, \R, e \rangle \Downarrow C'
         \quad
         \langle \sigma[x \mapsto C'], \mu, C', s \rangle \Downarrow_S o \,;\, \mu'}
        {\langle \sigma, \mu, C, \texttt{with}\ e\ \texttt{as}\ x\ \texttt{in}\ s \rangle
         \Downarrow_S o \,;\, \mu'}
   \tag{E-Context}

.. note::

   The context expression is evaluated under :math:`\R` rather than the active
   context :math:`C` because a constructor's arguments in the full FPy language
   are usually precisions, bitwidths, maximum values, etc. Rounding may inadvertently
   change the desired result.

Programs
--------

A program is a pair :math:`(\Phi, f_{\mathit{main}})` of a top-level environment
and an entry point, run on an argument :math:`v` supplied by the host. Where
:math:`\Phi(f_{\mathit{main}}) = (y, s)`, the program runs its body from the
initial state:

.. math::

   \langle [\, y \mapsto v \,], \emptyset, \R, s \rangle
   \Downarrow_S \mathsf{return}\ v' \,;\, \mu'

The only binding is the parameter, the store starts empty, and the active context
is :math:`\R`. The program's result is :math:`v'`; the final store :math:`\mu'`
is discarded.
