Language Semantics
======================

This page documents the semantics of FPy.  To stay tractable, it covers only
the *core* of the language—a minimal imperative fragment of constants,
arithmetic, function calls, and the basic statements—not the full surface
syntax; the remaining operators and statements evaluate the same way.

It describes how FPy programs *evaluate*, and in particular how the *active
rounding context* governs every arithmetic operation.  Typing is out of scope.

Syntax
------

FPy's expressions are boolean and numerical constants, arithmetic, comparisons,
function calls, and compound data—lists and tuples.  Its statements are the
usual imperative ones—assignment, sequencing, conditionals, return, assertion,
and skip—plus one unique to FPy: the *context statement*, which sets
the active rounding context for the expressions it evaluates.

In the formal syntax, :math:`n` is an arbitrary real number, :math:`x` ranges
over a countable set of identifiers, and :math:`\R` is the *real rounding
context*, whose rounding operation is the identity, so no rounding occurs.

.. math::

   \begin{array}{rcll}
   e & ::= & \texttt{true} \mid \texttt{false}
       & \text{boolean constants} \\
     & \mid & n
       & \text{numerical constants} \\
     & \mid & \R
       & \text{context constants} \\
     & \mid & x
       & \text{variable} \\
     & \mid & [\, e_1, \ldots, e_n \,]
       & \text{list constructor} \\
     & \mid & e_1[e_2]
       & \text{list indexing} \\
     & \mid & (\, e_1, \ldots, e_n \,)
       & \text{tuple} \\
     & \mid & e_1 + e_2
       & \text{arithmetic} \\
     & \mid & e_1 < e_2
       & \text{comparison} \\
     & \mid & f\ e
       & \text{function application} \\[1ex]
   s & ::= & p := e
       & \text{assignment} \\
     & \mid & s_1\, \texttt{;}\, s_2
       & \text{sequencing} \\
     & \mid & \texttt{if}\ e\ \texttt{then}\ s_1\ \texttt{else}\ s_2
       & \text{conditional} \\
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
     & \mid & p_1, \ldots, p_n
       & \text{tuple pattern}
   \end{array}

An assignment's left-hand side is a *pattern* :math:`p`—a variable or a
tuple of (possibly nested) patterns.  A tuple pattern deconstructs a tuple
position by position; this is the only way to take a tuple apart, since tuples
cannot be indexed.

``+`` and ``<`` stand in for arithmetic and comparison in general; every other
FPy operator evaluates the same way, though comparison and classification
operators yield booleans rather than rounded reals.

Values
------

Evaluating an FPy expression produces one of six kinds of value: a boolean, a
real number :math:`n`, a *rounding context* :math:`C`, a list of values, a
tuple of values, or a *function value*.  The only context constructible in this
fragment is :math:`\R`; full FPy provides constructors for the common rounding
contexts.

.. math::

   v ::= \texttt{true} \mid \texttt{false} \mid n \mid C
       \mid [\, v_1, \ldots, v_n \,] \mid (\, v_1, \ldots, v_n \,)
       \mid \langle \lambda x.\, s,\, \rho \rangle

A function value is a *closure* :math:`\langle \lambda x.\, s,\, \rho \rangle`—a parameter
:math:`x`, a body :math:`s`, and the environment :math:`\rho` captured at definition.
The fragment has no definition syntax, so closures are pre-bound in the initial environment,
one per top-level FPy function.

Evaluation
----------

Evaluation requires an environment :math:`\sigma`, mapping identifiers to
values, and an *active rounding context* :math:`C`.  The program state is the
triple :math:`\langle \sigma, C, p \rangle`, where :math:`p` is the expression
or statement under evaluation.  Two big-step judgements relate states to
results:

* :math:`\langle \sigma, C, e \rangle \Downarrow v` - expression :math:`e`
  evaluates to value :math:`v`;
* :math:`\langle \sigma, C, s \rangle \Downarrow_S o` - statement :math:`s`
  evaluates to an *outcome* :math:`o`.

A statement either completes normally with an updated environment or returns a
value, so an outcome is one of:

.. math::

   o ::= \mathsf{normal}\ \sigma \mid \mathsf{return}\ v

A :math:`\mathsf{normal}` outcome carries the environment threaded to the next
statement; a :math:`\mathsf{return}` outcome carries a function's result and
short-circuits the rest of the body.

The active rounding context :math:`C` is the crux of FPy's semantics: it is
threaded through every expression and rounds the exact result of each
arithmetic operation (see **E-Add**).

Expressions
^^^^^^^^^^^

Constants and variables evaluate to themselves and to their bound value,
respectively.  The real context :math:`\R` is itself a value.

.. math::

   \frac{}{\langle \sigma, C, \texttt{true} \rangle \Downarrow \texttt{true}}
   \tag{E-True}

.. math::

   \frac{}{\langle \sigma, C, \texttt{false} \rangle \Downarrow \texttt{false}}
   \tag{E-False}

.. math::

   \frac{}{\langle \sigma, C, n \rangle \Downarrow n}
   \tag{E-Num}

.. math::

   \frac{}{\langle \sigma, C, \R \rangle \Downarrow \R}
   \tag{E-Real}

.. math::

   \frac{}{\langle \sigma, C, x \rangle \Downarrow \sigma(x)}
   \tag{E-Var}

Lists evaluate their elements left to right; indexing selects an element.

.. math::

   \frac{\langle \sigma, C, e_1 \rangle \Downarrow v_1
         \quad \cdots \quad
         \langle \sigma, C, e_n \rangle \Downarrow v_n}
        {\langle \sigma, C, [\, e_1, \ldots, e_n \,] \rangle \Downarrow
         [\, v_1, \ldots, v_n \,]}
   \tag{E-List}

.. math::

   \frac{\langle \sigma, C, e_1 \rangle \Downarrow [\, v_1, \ldots, v_k \,]
         \quad
         \langle \sigma, C, e_2 \rangle \Downarrow n}
        {\langle \sigma, C, e_1[e_2] \rangle \Downarrow v_n}
   \tag{E-Ref}

Tuples are built like lists—elements left to right—but cannot be indexed;
a tuple is taken apart only by a tuple pattern (see **E-Assign**).

.. math::

   \frac{\langle \sigma, C, e_1 \rangle \Downarrow v_1
         \quad \cdots \quad
         \langle \sigma, C, e_n \rangle \Downarrow v_n}
        {\langle \sigma, C, (\, e_1, \ldots, e_n \,) \rangle \Downarrow
         (\, v_1, \ldots, v_n \,)}
   \tag{E-Tuple}

Arithmetic is where rounding happens.  The operands evaluate to real numbers,
and the active context :math:`C` rounds their exact sum :math:`\exact{n_1 + n_2}`
to a representable value.  Under :math:`\R`, rounding is the identity, so the
exact result is returned unchanged.

.. math::

   \frac{\langle \sigma, C, e_1 \rangle \Downarrow n_1
         \quad
         \langle \sigma, C, e_2 \rangle \Downarrow n_2}
        {\langle \sigma, C, e_1 + e_2 \rangle \Downarrow C(\exact{n_1 + n_2})}
   \tag{E-Add}

A comparison evaluates its operands and tests them as real numbers, producing a
boolean; unlike arithmetic, the result is exact and no rounding is applied.
``<`` is the representative—the other comparisons behave identically.

.. math::

   \frac{\langle \sigma, C, e_1 \rangle \Downarrow n_1
         \quad
         \langle \sigma, C, e_2 \rangle \Downarrow n_2}
        {\langle \sigma, C, e_1 < e_2 \rangle \Downarrow (n_1 < n_2)}
   \tag{E-Lt}

A function application looks up the closure bound to :math:`f`, evaluates the
argument, binds the parameter in the captured environment :math:`\rho`, and runs
the body to the value it returns. The body runs under the caller's context
:math:`C`; a well-formed body always returns, so its outcome is
:math:`\mathsf{return}\ v'`.

.. math::

   \frac{\sigma(f) = \langle \lambda x.\, s,\, \rho \rangle
         \quad
         \langle \sigma, C, e \rangle \Downarrow v
         \quad
         \langle \rho[x \mapsto v], C, s \rangle \Downarrow_S \mathsf{return}\ v'}
        {\langle \sigma, C, f\ e \rangle \Downarrow v'}
   \tag{E-App}

Statements
^^^^^^^^^^

Every statement evaluates to an outcome.  Assignment, skip, and a passing
assertion complete normally (:math:`\mathsf{normal}`) and :math:`\texttt{ret}`
returns
(:math:`\mathsf{return}`); sequencing, conditionals, and the context statement
pass along the outcome of whatever sub-statement they run, so a
:math:`\mathsf{return}` propagates out to the enclosing function.

Matching uses an auxiliary judgement :math:`p \triangleright v \Rightarrow \theta`,
read "pattern :math:`p` against value :math:`v` yields bindings :math:`\theta`".
A variable matches anything and binds it; a tuple pattern matches a tuple
position by position, combining the per-component bindings by disjoint union
:math:`\uplus` (the sub-patterns bind distinct variables).

.. math::

   \frac{}{x \triangleright v \Rightarrow [\, x \mapsto v \,]}
   \tag{M-Var}

.. math::

   \frac{p_1 \triangleright v_1 \Rightarrow \theta_1
         \quad \cdots \quad
         p_n \triangleright v_n \Rightarrow \theta_n}
        {p_1, \ldots, p_n \triangleright (\, v_1, \ldots, v_n \,)
         \Rightarrow \theta_1 \uplus \cdots \uplus \theta_n}
   \tag{M-Tuple}

Assignment evaluates its right-hand side, matches the value against the
pattern, and extends the environment with the bindings (:math:`\sigma[\theta]`
is :math:`\sigma` updated with every binding in :math:`\theta`).

.. math::

   \frac{\langle \sigma, C, e \rangle \Downarrow v
         \quad
         p \triangleright v \Rightarrow \theta}
        {\langle \sigma, C, p := e \rangle \Downarrow_S \mathsf{normal}\ \sigma[\theta]}
   \tag{E-Assign}

The skip statement does nothing; :math:`\texttt{ret}` evaluates its operand and
returns it.

.. math::

   \frac{}{\langle \sigma, C, \texttt{skip} \rangle \Downarrow_S \mathsf{normal}\ \sigma}
   \tag{E-Skip}

.. math::

   \frac{\langle \sigma, C, e \rangle \Downarrow v}
        {\langle \sigma, C, \texttt{ret}\ e \rangle \Downarrow_S \mathsf{return}\ v}
   \tag{E-Ret}

An assertion evaluates its test; if it holds, evaluation continues with the
environment unchanged. FPy has no error handling, so a failing assertion has no
rule—evaluation is simply stuck, as for any other undefined operation.

.. math::

   \frac{\langle \sigma, C, e \rangle \Downarrow \texttt{true}}
        {\langle \sigma, C, \texttt{assert}\ e \rangle \Downarrow_S \mathsf{normal}\ \sigma}
   \tag{E-Assert}

Sequencing runs :math:`s_1` first.
If :math:`s_1` returns, the sequence returns at once.
Otherwise, :math:`s_2` runs under the updated environment to produce the
sequence's outcome.

.. math::

   \frac{\langle \sigma, C, s_1 \rangle \Downarrow_S \mathsf{normal}\ \sigma'
         \quad
         \langle \sigma', C, s_2 \rangle \Downarrow_S o}
        {\langle \sigma, C, s_1\, \texttt{;}\, s_2 \rangle \Downarrow_S o}
   \tag{E-Seq-Normal}

.. math::

   \frac{\langle \sigma, C, s_1 \rangle \Downarrow_S \mathsf{return}\ v}
        {\langle \sigma, C, s_1\, \texttt{;}\, s_2 \rangle \Downarrow_S \mathsf{return}\ v}
   \tag{E-Seq-Return}

A conditional evaluates its condition to a boolean and runs the matching
branch; the branch's outcome becomes the conditional's, so a :math:`\texttt{ret}`
in either branch returns from the enclosing function.

.. math::

   \frac{\langle \sigma, C, e \rangle \Downarrow \texttt{true}
         \quad
         \langle \sigma, C, s_1 \rangle \Downarrow_S o}
        {\langle \sigma, C, \texttt{if}\ e\ \texttt{then}\ s_1\ \texttt{else}\ s_2 \rangle \Downarrow_S o}
   \tag{E-If-True}

.. math::

   \frac{\langle \sigma, C, e \rangle \Downarrow \texttt{false}
         \quad
         \langle \sigma, C, s_2 \rangle \Downarrow_S o}
        {\langle \sigma, C, \texttt{if}\ e\ \texttt{then}\ s_1\ \texttt{else}\ s_2 \rangle \Downarrow_S o}
   \tag{E-If-False}

The context statement is the heart of FPy.  The context expression :math:`e` is
evaluated under :math:`\R` to a new context :math:`C'`, and the body :math:`s`
runs under :math:`C'` with :math:`x` bound to :math:`C'`, so it can refer to its
governing context as a value.  :math:`C'` governs only the body—the
surrounding context :math:`C` is unchanged and still applies after the
``with``.  The body's outcome becomes the statement's outcome, so a
:math:`\texttt{ret}` inside a ``with`` returns from the enclosing function.

.. math::

   \frac{\langle \sigma, \R, e \rangle \Downarrow C'
         \quad
         \langle \sigma[x \mapsto C'], C', s \rangle \Downarrow_S o}
        {\langle \sigma, C, \texttt{with}\ e\ \texttt{as}\ x\ \texttt{in}\ s \rangle \Downarrow_S o}
   \tag{E-Context}
