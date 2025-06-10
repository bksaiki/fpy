Type system
================

FPy features a polymorphic Hindley-Milner type system
like languages such as Haskell or OCaml.

For brevity, the type system is described using
a simplified grammar of the full FPy language.
Expressions :math:`e` in FPy contain
function systems :math:`f`, variables :math:`x`,
real number constants :math:`n`, and boolean constants
:math:`\text{true}` and :math:`\text{false}`.
All functions are assumed to be unary.

.. math::

    \begin{array}{rcl}
    e & ::= & \text{true} \\
      & \mid & \text{false} \\
      & \mid & n \\
      & \mid & x \\
      & \mid & f\; e
    \end{array}

Statements :math:`s` in FPy contain expressions :math:`e`
variables :math:`x`, and rounding contexts :math:`R`.

.. math::

    \begin{array}{rcl}
    s & ::= & x = e \\
      & \mid & s_1 ; s_2 \\
      & \mid & \text{if}\; e\; \text{then}\; s_1\; \text{else}\; s_2 \\
      & \mid & \text{while}\; e\; \text{then}\; s \\
      & \mid & \text{ret}\; e\\
    \end{array}

Expressions in FPy have a type, :math:`T`,
which is one of the following:

.. math::

    \begin{array}{rcl}
    T & ::= & \text{Bool} \\
      & \mid & \text{Real}\; r \\
      & \mid & T_1 \to T_2
    \end{array}

The typing judgements for the core language of FPy are
as follows.

.. math::

    \frac{}
         {\Gamma, R \vdash \text{true} : \text{Bool}}
    \quad\text{(T-True)}\\

.. math::

    \frac{}
         {\Gamma, R \vdash \text{false} : \text{Bool}}
    \quad\text{(T-False)}\\

.. math::

    \frac{}
         {\Gamma, R \vdash n : \text{Real}\; R}
    \quad\text{(T-Real)}\\

.. math::

    \frac{x : T \in \Gamma}
         {\Gamma, R \vdash x : T}
    \quad \text{(T-Var)}

.. math::

    \frac{\Gamma, R \vdash f : T \to \text{Bool}
         \qquad \Gamma, R \vdash e : T }
         {\Gamma, R \vdash f\; e : \text{Bool}}
    \quad\text{(T-BoolApp)}

.. math::

    \frac{\Gamma, R \vdash f : T \to \text{Real}\;r
         \qquad \Gamma, R \vdash e : T }
         {\Gamma, R \vdash f\; e : \text{Real}\;r}
    \quad\text{(T-RealApp)}
