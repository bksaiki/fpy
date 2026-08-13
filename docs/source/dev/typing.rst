Type System
===========

FPy is *not* statically typed. A program runs (see :doc:`semantics`) without
any type-checking, and every value carries its rounding context at run time.
Static types enter only when compiling a program to a strongly-typed target,
such as C++: the compiler infers a static type for each expression, and a
program must type-check to be compiled — even though an ill-typed program may
still run under the dynamic semantics.

This page gives the typing rules for the same core fragment as
:doc:`semantics`. The types here are *context-free*: a real number has type
:math:`\texttt{real}`, with no rounding context attached. To emit code for a
typed target, a separate pass refines each :math:`\texttt{real}` with the
rounding context it is produced under (so the backend can pick a concrete
machine type — ``float``, ``double``, a fixed-point format, and so on). That
refinement is out of scope here.

Types
-----

The scalar types mirror the three scalar value kinds — booleans, real numbers,
and rounding contexts — and are joined by list, tuple, and reference types, one
per value kind.

.. math::

   T ::= \texttt{bool} \mid \texttt{real} \mid \texttt{context}
       \mid \texttt{list}\ T
       \mid T_1 \times \cdots \times T_n
       \mid \texttt{ref}\ T

Both context literals have type :math:`\texttt{context}`. A reference to a
:math:`T` has type :math:`\texttt{ref}\ T`; since a reference is both read and
written at that one type, :math:`\texttt{ref}` is invariant.

There is no function type. Functions are not values (see :doc:`semantics`), so
they are typed by a *signature* :math:`T_1 \rightarrow T_2` assigned by a
top-level environment :math:`\Phi`, mirroring the :math:`\Phi` that assigns each
function name its definition. A signature is not a :math:`T`—no type contains an
arrow, so a function can be neither stored nor passed. Like :math:`\Gamma`,
:math:`\Phi` is fixed, and only **T-App** consults it.

Typing
------

Typing is the judgement :math:`\Gamma \vdash e : T`, read "under typing context
:math:`\Gamma`, expression :math:`e` has type :math:`T`". :math:`\Gamma`
assigns each variable a single type and is the solution of whole-function type
inference — computed by unification, so library functions may be polymorphic.
The rules below present the monomorphic case and state when a program agrees
with that solution; statement well-formedness is written
:math:`\Gamma \vdash s\ \texttt{ok}`. Because :math:`\Gamma` is fixed, typing
*checks* a program against it rather than building it up statement by
statement.

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

   \frac{}{\Gamma \vdash \texttt{ctx}\ \{ \ldots \} : \texttt{context}}
   \tag{T-Ctx}

.. math::

   \frac{x : T \in \Gamma}{\Gamma \vdash x : T}
   \tag{T-Var}

A list is homogeneous; indexing recovers the element type. A tuple's type
records each component.

.. math::

   \frac{\Gamma \vdash e_1 : T \quad \cdots \quad \Gamma \vdash e_n : T}
        {\Gamma \vdash [\, e_1, \ldots, e_n \,] : \texttt{list}\ T}
   \tag{T-List}

.. math::

   \frac{\Gamma \vdash e_1 : \texttt{list}\ T \quad \Gamma \vdash e_2 : \texttt{real}}
        {\Gamma \vdash e_1[e_2] : T}
   \tag{T-Index}

.. math::

   \frac{\Gamma \vdash e_1 : T_1 \quad \cdots \quad \Gamma \vdash e_n : T_n}
        {\Gamma \vdash (\, e_1, \ldots, e_n \,) : T_1 \times \cdots \times T_n}
   \tag{T-Tuple}

Dereferencing unwraps a reference type; allocating the reference is a statement
(see **T-Ref**).

.. math::

   \frac{\Gamma \vdash e : \texttt{ref}\ T}
        {\Gamma \vdash \texttt{!}\, e : T}
   \tag{T-Deref}

As in :doc:`semantics`, ``+`` and ``<`` are representatives: arithmetic maps
reals to a real, and comparison maps reals to a boolean.

.. math::

   \frac{\Gamma \vdash e_1 : \texttt{real} \quad \Gamma \vdash e_2 : \texttt{real}}
        {\Gamma \vdash e_1 + e_2 : \texttt{real}}
   \tag{T-Add}

.. math::

   \frac{\Gamma \vdash e_1 : \texttt{real} \quad \Gamma \vdash e_2 : \texttt{real}}
        {\Gamma \vdash e_1 < e_2 : \texttt{bool}}
   \tag{T-Lt}

Statements
^^^^^^^^^^

An assignment checks that its right-hand side's type agrees with the pattern on
the left. Because :math:`\Gamma` is fixed — every variable already has its
inferred type — a pattern needs no rules of its own: it is typed by the
*expression* rules, a variable by **T-Var** and a tuple pattern
:math:`(\, x_1, \ldots, x_n \,)` like the tuple of the same shape by
**T-Tuple**.

.. math::

   \frac{\Gamma \vdash e : T \quad \Gamma \vdash p : T}
        {\Gamma \vdash p = e\ \texttt{ok}}
   \tag{T-Assign}

An allocation wraps its operand's type: the variable it binds refers to what the
right-hand side produced.

.. math::

   \frac{\Gamma \vdash e : T \quad \Gamma \vdash x : \texttt{ref}\ T}
        {\Gamma \vdash x = \texttt{ref}\ e\ \texttt{ok}}
   \tag{T-Ref}

An update writes at the type its target refers to, so the two sides agree only
up to the :math:`\texttt{ref}`.

.. math::

   \frac{\Gamma \vdash x : \texttt{ref}\ T \quad \Gamma \vdash e : T}
        {\Gamma \vdash x := e\ \texttt{ok}}
   \tag{T-Update}

An application checks the argument against the callee's signature and binds its
result at the signature's result type.

.. math::

   \frac{\Phi(f) = T_1 \rightarrow T_2 \quad
         \Gamma \vdash e : T_1 \quad
         \Gamma \vdash x : T_2}
        {\Gamma \vdash x = f\ e\ \texttt{ok}}
   \tag{T-App}

.. math::

   \frac{}{\Gamma \vdash \texttt{skip}\ \texttt{ok}}
   \tag{T-Skip}

The :math:`\texttt{ret}` operand may have any type; all returns in a function
share one type, which becomes the function's result type. An assertion tests a
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

The context statement requires a context-typed expression — evaluating it yields
the active rounding context for the body — and binds the target to that context.

.. math::

   \frac{\Gamma \vdash e : \texttt{context} \quad
         x : \texttt{context} \in \Gamma \quad
         \Gamma \vdash s\ \texttt{ok}}
        {\Gamma \vdash \texttt{with}\ e\ \texttt{as}\ x\ \texttt{in}\ s\ \texttt{ok}}
   \tag{T-Context}
