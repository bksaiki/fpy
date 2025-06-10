Type system
================

To understand the FPy type system,
we first describe the type system of :math:`\lambda_{FPy}`,
a typed lambda calculus demonstrating the core features of FPy.

:math:`\lambda_{FPy}`
^^^^^^^^^^^^^^^^^^^^^^

The :math:`\lambda_{FPy}` language consists of the usual
terms found in the simply typed lambda calculus (STLC).
The terminals of :math:`\lambda_{FPy}` consists of
boolean values, :math:`\texttt{true}` and :math:`\texttt{false}`,
real number constants, :math:`n`, and variables, :math:`x`.

.. math::

    \begin{array}{rcl}
    e & ::= & \texttt{true} \\
      & \mid & \texttt{false} \\
      & \mid & n \\
      & \mid & x \\
      & \mid & \lambda x : T. e \\
      & \mid & e_1\; e_2 \\
      & \mid & \text{with}\; R \;\text{in}\; e \\
    \end{array}

All expressions in :math:`\lambda_{FPy}` have an implcit *rounding context*
which defines how real numbers (or real-valued operations) should be rounded.
The *context expression* :math:`\text{with}\; R\; \text{in}\; e`
allows explicit control over the rounding context.

Like the STLC, types in :math:`\lambda_{FPy}` consist of
base types (boolean and real types) and function types.

.. math::

    \begin{array}{rcl}
    T & ::= & \texttt{Bool} \\
      & \mid & \texttt{Real}\; R \\
      & \mid & T_1 \overset{\small R}{\rightarrow} T_2
    \end{array}

The real number type is parameterized by the rounding context, :math:`R`.
Thus the type :math:`\texttt{Real}\; R` may be read as the type
of real numbers rounded under the rounding context :math:`R`.
Function type are also parameterized by a rounding context,
which may be viewed as a *coeffect* in the type system:
functions in :math:`\lambda_{FPy}` require a rounding context from the caller.
Unlike other coeffect systems,
a coeffect in this system consists of at most one rounding context:
rounding contexts are not additive.

.. T-True
.. math::
    \DeclareMathOperator{\Coeff}{@}
    \frac{}
         {\Gamma\Coeff\varepsilon \vdash \texttt{true} : \texttt{Bool}}
    \quad\text{(T-True)}\\

.. T-False
.. math::

    \frac{}
         {\Gamma\Coeff\varepsilon \vdash \texttt{false} : \texttt{Bool}}
    \quad\text{(T-False)}\\

.. T-Real
.. math::

    \frac{}
         {\Gamma\Coeff R \vdash n : \texttt{Real}\; R}
    \quad\text{(T-Real)}\\

.. T-Var
.. math::

    \frac{x : T \in \Gamma}
         {\Gamma\Coeff\varepsilon \vdash x : T}
    \quad \text{(T-Var)}\\

.. T-Abs
.. math::

    \frac{\Gamma, x : T_1 \Coeff R_1 \vdash e : T_2}
         {\Gamma\Coeff R_2 \vdash \lambda x : T_1. e : T_1 \overset{\small R_1}{\rightarrow} T_2}
    \quad\text{(T-Abs)}\\

.. T-App
.. math::

    \frac{\Gamma\Coeff R \vdash e_1 : T_1 \overset{\small R}{\rightarrow} T_2
         \qquad \Gamma\Coeff R \vdash e_2 : T_1}
         {\Gamma\Coeff R \vdash e_1\; e_2 : T_2}
    \quad\text{(T-App)}\\

.. T-Ctx
.. math::

    \frac{\Gamma\Coeff R_2 \vdash e : T}
         {\Gamma\Coeff R_1 \vdash \text{with}\; R_2\; \text{in}\; e : T}
    \quad\text{(T-Ctx)}\\

By convention, we extend the usual typing context :math:`\Gamma`
with a coeffect annotation, :math:`R`, denoted with :math:`\Gamma \Coeff R`.
We represent the lack of a rounding context with :math:`\varepsilon`.
At any point, the rounding context may be dropped.

Boolean constants (:math:`\text{T-True}`, :math:`\text{T-False}`)
and variables (:math:`\text{T-Var}`) are unchanged from the STLC
as they require no rounding context.
Real numbers (:math:`\text{T-Real}`) are always rounded
under the rounding context :math:`R`.
For abstraction (:math:`\text{T-Abs}`),
the abstraction body :math:`e` is typed under some
rounding context :math:`R_1` that may be different
from the rounding context :math:`R_2` outside the abstraction.
At any call site of the function, :math:`\lambda x : T_1. e`,
the coeffect is handled by the local rounding context:
for application (:math:`\text{T-App}`),
the local rounding context :math:`R`
matches the rounding context of the function type.
Context expressions (:math:`\text{T-Ctx}`) change
the rounding context of the expression :math:`e`.


.. FPy features a polymorphic Hindley-Milner type system
.. like languages such as Haskell or OCaml.

.. For brevity, the type system is described using
.. a simplified grammar of the full FPy language.
.. An FPy program consists of statements :math:`s`, expressions :math:`e`,
.. rounding contexts :math:`R`, and function symbols :math:`f`.
.. The terminals of an expression are variables :math:`x`,
.. real number constants :math:`n`, and boolean constants
.. :math:`\text{true}` and :math:`\text{false}`.
.. All functions are assumed to be unary.

.. .. math::

..     \begin{array}{rcl}
..     e & ::= & \text{true} \\
..       & \mid & \text{false} \\
..       & \mid & n \\
..       & \mid & x \\
..       & \mid & f\; e
..     \end{array}

.. .. math::

..     \begin{array}{rcl}
..     s & ::= & x = e \\
..       & \mid & s_1 ; s_2 \\
..       & \mid & \text{if}\; e\; \text{then}\; s_1\; \text{else}\; s_2 \\
..       & \mid & \text{while}\; e\; \text{then}\; s \\
..       & \mid & \text{with}\; R\; \text{do}\; s \\
..       & \mid & \text{ret}\; e\\
..     \end{array}

.. Expressions in FPy have a type, :math:`T`,
.. which is one of the following:

.. .. math::

..     \begin{array}{rcl}
..     T & ::= & \text{Unit} \\
..       & \mid & \text{Bool} \\
..       & \mid & \text{Real}\; R \\
..       & \mid & T_1 \to T_2
..     \end{array}

.. The typing judgements for the core language of FPy are below.
.. The symbol :math:`\Gamma` is a typing context and the judgement
.. :math:`\rho : T` means the return type of the current function is :math:`T`.

.. .. math::

..     \frac{}
..          {\Gamma; R \vdash \text{true} : \text{Bool}}
..     \quad\text{(T-True)}\\

.. .. math::

..     \frac{}
..          {\Gamma; R \vdash \text{false} : \text{Bool}}
..     \quad\text{(T-False)}\\

.. .. math::

..     \frac{}
..          {\Gamma; R \vdash n : \text{Real}\; R}
..     \quad\text{(T-Real)}\\

.. .. math::

..     \frac{x : T \in \Gamma}
..          {\Gamma; R \vdash x : T}
..     \quad \text{(T-Var)}

.. .. math::

..     \frac{\Gamma; R \vdash f : T \to \text{Bool}
..          \qquad \Gamma; R \vdash e : T }
..          {\Gamma; R \vdash f\; e : \text{Bool}}
..     \quad\text{(T-BoolApp)}

.. .. math::

..     \frac{\Gamma; R_1 \vdash f : T \to \text{Real}\; R_2
..          \qquad \Gamma; R_1 \vdash e : T }
..          {\Gamma; R_1 \vdash f\; e : \text{Real}\; R_2}
..     \quad\text{(T-RealApp)}

.. .. math::

..     \frac{\Gamma; R \vdash x : T
..          \qquad \Gamma; R \vdash e : T }
..          {\Gamma; R \vdash x = e : \text{Unit}}
..     \quad\text{(T-Assign)}

.. .. math::

..     \frac{\Gamma; R \vdash s_1 : \text{Unit}
..          \qquad \Gamma; R \vdash s_2 : \text{Unit} }
..          {\Gamma; R \vdash s_1 ; s_2 : \text{Unit}}
..     \quad\text{(T-Seq)}

.. .. math::

..     \frac{\Gamma; R \vdash e : \text{Bool}
..          \qquad \Gamma; R \vdash s_1 : \text{Unit}
..          \qquad \Gamma; R \vdash s_2 : \text{Unit} }
..          {\Gamma; R \vdash \text{if}\; e\; \text{then}\; s_1\; \text{else}\; s_2 : \text{Unit} }
..     \quad\text{(T-If)}

.. .. math::

..     \frac{\Gamma; R \vdash e : \text{Bool}
..          \qquad \Gamma; R \vdash s : \text{Unit} }
..          {\Gamma; R \vdash \text{while}\; e\; \text{then}\; s : \text{Unit}  }
..     \quad\text{(T-While)}

.. .. math::

..     \frac{\Gamma; R_2 \vdash s : \text{Unit}}
..          {\Gamma; R_1 \vdash \text{with}\; R_2\; \text{then}\; s : \text{Unit} }
..     \quad\text{(T-Context)}

.. .. math::

..     \frac{\Gamma; R \vdash e : T}
..          {\Gamma; R \vdash \text{ret}\; e : \text{Unit} }
..     \quad\text{(T-Ret)}

.. .. math::

..     \frac{\Gamma; R \vdash e : T}
..          {\Gamma, \rho : T; R \vdash \text{ret}\; e : \text{Unit} }
..     \quad\text{(T-Ret)}
