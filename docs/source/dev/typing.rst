Type System
===========

FPy is *not* statically typed.  A program runs (see :doc:`semantics`) without
any type-checking, and every value carries its rounding context at run time.
Static types enter only when compiling a program to a strongly-typed target,
such as C++: the compiler infers a static type for each expression, and a
program must type-check to be compiled — even though an ill-typed program may
still run under the dynamic semantics.

This page gives the typing rules for the same core fragment as
:doc:`semantics`.  The types here are *context-free*: a real number has type
:math:`\texttt{real}`, with no rounding context attached.  To emit code for a
typed target, a separate pass refines each :math:`\texttt{real}` with the
rounding context it is produced under (so the backend can pick a concrete
machine type — ``float``, ``double``, a fixed-point format, and so on).  That
refinement is out of scope here.

Types
-----

The scalar types mirror the three scalar value kinds — booleans, real numbers,
and rounding contexts — and are joined by list, tuple, and function types.

.. math::

   T ::= \texttt{bool} \mid \texttt{real} \mid \texttt{context}
       \mid \texttt{list}\ T
       \mid T_1 \times \cdots \times T_n
       \mid T_1 \rightarrow T_2

The real rounding context :math:`\R` has type :math:`\texttt{context}`.
Function symbols are assigned a function type :math:`T_1 \rightarrow T_2`.

Typing
------

Typing is the judgement :math:`\Gamma \vdash e : T`, read "under typing context
:math:`\Gamma`, expression :math:`e` has type :math:`T`", where :math:`\Gamma`
maps each variable to its type.  Inference computes :math:`\Gamma` by
unification (so library functions may be polymorphic); the rules below present
the monomorphic case and specify when the result is well-typed.  Statement
well-formedness is written :math:`\Gamma \vdash s\ \texttt{ok}`.

Expressions
^^^^^^^^^^^

Constants have their scalar types; a variable has the type :math:`\Gamma`
assigns it.

.. math::

   \frac{}{\Gamma \vdash \texttt{true} : \texttt{bool}}
   \tag{T-True}

.. math::

   \frac{}{\Gamma \vdash \texttt{false} : \texttt{bool}}
   \tag{T-False}

.. math::

   \frac{}{\Gamma \vdash n : \texttt{real}}
   \tag{T-Num}

.. math::

   \frac{}{\Gamma \vdash \R : \texttt{context}}
   \tag{T-Real}

.. math::

   \frac{x : T \in \Gamma}{\Gamma \vdash x : T}
   \tag{T-Var}

A list is homogeneous; indexing recovers the element type.  A tuple's type
records each component.

.. math::

   \frac{\Gamma \vdash e_1 : T \quad \cdots \quad \Gamma \vdash e_n : T}
        {\Gamma \vdash [\, e_1, \ldots, e_n \,] : \texttt{list}\ T}
   \tag{T-List}

.. math::

   \frac{\Gamma \vdash e_1 : \texttt{list}\ T \quad \Gamma \vdash e_2 : \texttt{real}}
        {\Gamma \vdash e_1[e_2] : T}
   \tag{T-Ref}

.. math::

   \frac{\Gamma \vdash e_1 : T_1 \quad \cdots \quad \Gamma \vdash e_n : T_n}
        {\Gamma \vdash (\, e_1, \ldots, e_n \,) : T_1 \times \cdots \times T_n}
   \tag{T-Tuple}

As in :doc:`semantics`, ``+``, ``<``, and :math:`\wedge` are representatives:
arithmetic maps reals to a real, comparison maps reals to a boolean, and the
logical connectives map booleans to a boolean.

.. math::

   \frac{\Gamma \vdash e_1 : \texttt{real} \quad \Gamma \vdash e_2 : \texttt{real}}
        {\Gamma \vdash e_1 + e_2 : \texttt{real}}
   \tag{T-Add}

.. math::

   \frac{\Gamma \vdash e_1 : \texttt{real} \quad \Gamma \vdash e_2 : \texttt{real}}
        {\Gamma \vdash e_1 < e_2 : \texttt{bool}}
   \tag{T-Lt}

.. math::

   \frac{\Gamma \vdash e_1 : \texttt{bool} \quad \Gamma \vdash e_2 : \texttt{bool}}
        {\Gamma \vdash e_1 \wedge e_2 : \texttt{bool}}
   \tag{T-And}

.. math::

   \frac{\Gamma \vdash f : T_1 \rightarrow T_2 \quad \Gamma \vdash e : T_1}
        {\Gamma \vdash f\ e : T_2}
   \tag{T-App}

Statements
^^^^^^^^^^

Assignment binds a *pattern*, so it first types the pattern against the
right-hand side's type.  Pattern typing :math:`\Gamma \vdash p : T` checks that
the variables in :math:`p` carry the components of :math:`T`.

.. math::

   \frac{x : T \in \Gamma}{\Gamma \vdash x : T}
   \tag{TP-Var}

.. math::

   \frac{\Gamma \vdash p_1 : T_1 \quad \cdots \quad \Gamma \vdash p_n : T_n}
        {\Gamma \vdash p_1, \ldots, p_n : T_1 \times \cdots \times T_n}
   \tag{TP-Tuple}

.. math::

   \frac{\Gamma \vdash e : T \quad \Gamma \vdash p : T}
        {\Gamma \vdash p := e\ \texttt{ok}}
   \tag{T-Assign}

.. math::

   \frac{}{\Gamma \vdash \texttt{skip}\ \texttt{ok}}
   \tag{T-Skip}

The :math:`\texttt{ret}` operand may have any type; all returns in a function
share one type, which becomes the function's result type.  An assertion tests a
boolean.

.. math::

   \frac{\Gamma \vdash e : T}{\Gamma \vdash \texttt{ret}\ e\ \texttt{ok}}
   \tag{T-Ret}

.. math::

   \frac{\Gamma \vdash e : \texttt{bool}}{\Gamma \vdash \texttt{assert}\ e\ \texttt{ok}}
   \tag{T-Assert}

Sequencing and conditionals require their parts to be well-typed; a conditional
also requires a boolean guard.

.. math::

   \frac{\Gamma \vdash s_1\ \texttt{ok} \quad \Gamma \vdash s_2\ \texttt{ok}}
        {\Gamma \vdash s_1\, \texttt{;}\, s_2\ \texttt{ok}}
   \tag{T-Seq}

.. math::

   \frac{\Gamma \vdash e : \texttt{bool} \quad
         \Gamma \vdash s_1\ \texttt{ok} \quad
         \Gamma \vdash s_2\ \texttt{ok}}
        {\Gamma \vdash \texttt{if}\ e\ \texttt{then}\ s_1\ \texttt{else}\ s_2\ \texttt{ok}}
   \tag{T-If}

The context statement requires a context-typed expression — evaluating it
yields the active rounding context for the body — and binds the target to that
context.

.. math::

   \frac{\Gamma \vdash e : \texttt{context} \quad
         x : \texttt{context} \in \Gamma \quad
         \Gamma \vdash s\ \texttt{ok}}
        {\Gamma \vdash \texttt{with}\ e\ \texttt{as}\ x\ \texttt{in}\ s\ \texttt{ok}}
   \tag{T-Context}
