FPy Language Semantics
======================

This page documents the semantics of FPy.
To keep the presentation tractable, it covers only the *core* constructs of
the language—a minimal imperative fragment of boolean and numerical constants,
arithmetic, function calls, and the basic statements—rather than the full surface syntax.
The remaining FPy operators and statements follow the same evaluation patterns.

It describes how FPy programs *evaluate*; in particular, how the *active
rounding context* governs the result of every arithmetic operation.  Typing is
intentionally out of scope.

Syntax
------

FPy's expression language includes boolean and numerical constants,
arithmetic expressions, and function calls.  It features the usual imperative
statements — assignment, sequencing, return, and skip — together with one
construct unique to FPy: the *context statement*, which manages the active
rounding context used to evaluate expressions.

In the formal syntax, :math:`n` is an arbitrary real number, :math:`x` ranges
over a countable set of identifiers, and :math:`\R` is the *real rounding
context*: the unique context whose rounding operation is the identity
function, so that no rounding actually occurs.

.. math::

   \begin{array}{rcll}
   e & ::= & \texttt{true} \mid \texttt{false}
       & \text{boolean constants} \\
     & \mid & n \mid \R
       & \text{numerical / context constants} \\
     & \mid & x
       & \text{variable} \\
     & \mid & [\, e_1, \ldots, e_n \,] \mid e_1[e_2]
       & \text{list literal / indexing} \\
     & \mid & e_1 + e_2
       & \text{arithmetic} \\
     & \mid & f\ e
       & \text{function application} \\[1ex]
   s & ::= & x := e
       & \text{assignment} \\
     & \mid & s_1\, \texttt{;}\, s_2
       & \text{sequencing} \\
     & \mid & \texttt{ret}\ e
       & \text{return} \\
     & \mid & \texttt{with}\ e\ \texttt{as}\ x\ \texttt{in}\ s
       & \text{context statement} \\
     & \mid & \texttt{skip}
       & \text{no-op}
   \end{array}

Only ``+`` is shown as a representative arithmetic operation; every other FPy
operator follows the same evaluation pattern.

Values
------

Evaluating an FPy expression produces one of four kinds of value: a boolean; a
real number :math:`n`; a *rounding context* :math:`C`; or a list of values.
For concision, the only constructible rounding context in this fragment is
:math:`\R`; the full language provides constructors for the common rounding
contexts.

.. math::

   v ::= \texttt{true} \mid \texttt{false} \mid n \mid C
       \mid [\, v_1, \ldots, v_n \,]

Evaluation
----------

Evaluation requires an environment :math:`\sigma`, mapping identifiers to
values, and an *active rounding context* :math:`C`.  The program state is the
triple :math:`\langle \sigma, C, p \rangle`, where :math:`p` is the expression
or statement under evaluation.  Two big-step judgements relate states to
results:

* :math:`\langle \sigma, C, e \rangle \Downarrow v` — expression :math:`e`
  evaluates to value :math:`v`;
* :math:`\langle \sigma, C, s \rangle \Downarrow_S \sigma'` — statement
  :math:`s` evaluates to an updated environment :math:`\sigma'`.

The active rounding context :math:`C` is the crux of FPy's semantics: it is
threaded through every expression and applied to the exact real-number result
of each arithmetic operation (see **E-Add**).

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

Arithmetic is where rounding happens.  The operands evaluate to real numbers,
their exact mathematical sum :math:`\exact{n_1 + n_2}` is computed, and the
active context :math:`C` rounds that exact result to a representable value.
Under the real context :math:`\R`, rounding is the identity and the exact
result is returned unchanged.

.. math::

   \frac{\langle \sigma, C, e_1 \rangle \Downarrow n_1
         \quad
         \langle \sigma, C, e_2 \rangle \Downarrow n_2}
        {\langle \sigma, C, e_1 + e_2 \rangle \Downarrow C(\exact{n_1 + n_2})}
   \tag{E-Add}

A function application evaluates its argument and applies the function bound to
the symbol in the environment.

.. math::

   \frac{\langle \sigma, C, e \rangle \Downarrow v}
        {\langle \sigma, C, f\ e \rangle \Downarrow \sigma(f)(v)}
   \tag{E-App}

Statements
^^^^^^^^^^

Assignment evaluates its right-hand side and extends the environment.
Sequencing threads the environment from one statement to the next.

.. math::

   \frac{\langle \sigma, C, e \rangle \Downarrow v}
        {\langle \sigma, C, x := e \rangle \Downarrow_S \sigma[x \mapsto v]}
   \tag{E-Assign}

.. math::

   \frac{\langle \sigma, C, s_1 \rangle \Downarrow_S \sigma'
         \quad
         \langle \sigma', C, s_2 \rangle \Downarrow_S \sigma''}
        {\langle \sigma, C, s_1\, \texttt{;}\, s_2 \rangle \Downarrow_S \sigma''}
   \tag{E-Seq}

The context statement is the heart of FPy.  The context expression :math:`e`
is evaluated under the real context :math:`\R` to obtain a new context
:math:`C'`.  The body :math:`s` is then evaluated with :math:`C'` as the active
rounding context and with :math:`x` bound to :math:`C'`, so the body can refer
to the context that governs it as an ordinary value.  The surrounding context
:math:`C` is restored once the body completes.

.. math::

   \frac{\langle \sigma, \R, e \rangle \Downarrow C'
         \quad
         \langle \sigma[x \mapsto C'], C', s \rangle \Downarrow_S \sigma'}
        {\langle \sigma, C, \texttt{with}\ e\ \texttt{as}\ x\ \texttt{in}\ s \rangle \Downarrow_S \sigma'}
   \tag{E-Context}
