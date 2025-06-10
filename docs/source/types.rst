Type system
================

FPy features a polymorphic Hindley-Milner type system
like languages such as Haskell or OCaml.

For brevity, the type system is described using
a simplified grammar of the full FPy language.
An FPy program consists of statements :math:`s`, expressions :math:`e`,
rounding contexts :math:`R`, and function symbols :math:`f`.
The terminals of an expression are variables :math:`x`,
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

.. math::

    \begin{array}{rcl}
    s & ::= & x = e \\
      & \mid & s_1 ; s_2 \\
      & \mid & \text{if}\; e\; \text{then}\; s_1\; \text{else}\; s_2 \\
      & \mid & \text{while}\; e\; \text{then}\; s \\
      & \mid & \text{with}\; R\; \text{do}\; s \\
      & \mid & \text{ret}\; e\\
    \end{array}

Expressions in FPy have a type, :math:`T`,
which is one of the following:

.. math::

    \begin{array}{rcl}
    T & ::= & \text{Unit} \\
      & \mid & \text{Bool} \\
      & \mid & \text{Real}\; R \\
      & \mid & T_1 \to T_2
    \end{array}

The typing judgements for the core language of FPy are below.
The symbol :math:`\Gamma` is a typing context and the judgement
:math:`\rho : T` means the return type of the current function is :math:`T`.

.. math::

    \frac{}
         {\Gamma; R \vdash \text{true} : \text{Bool}}
    \quad\text{(T-True)}\\

.. math::

    \frac{}
         {\Gamma; R \vdash \text{false} : \text{Bool}}
    \quad\text{(T-False)}\\

.. math::

    \frac{}
         {\Gamma; R \vdash n : \text{Real}\; R}
    \quad\text{(T-Real)}\\

.. math::

    \frac{x : T \in \Gamma}
         {\Gamma; R \vdash x : T}
    \quad \text{(T-Var)}

.. math::

    \frac{\Gamma; R \vdash f : T \to \text{Bool}
         \qquad \Gamma; R \vdash e : T }
         {\Gamma; R \vdash f\; e : \text{Bool}}
    \quad\text{(T-BoolApp)}

.. math::

    \frac{\Gamma; R_1 \vdash f : T \to \text{Real}\; R_2
         \qquad \Gamma; R_1 \vdash e : T }
         {\Gamma; R_1 \vdash f\; e : \text{Real}\; R_2}
    \quad\text{(T-RealApp)}

.. math::

    \frac{\Gamma; R \vdash x : T
         \qquad \Gamma; R \vdash e : T }
         {\Gamma; R \vdash x = e : \text{Unit}}
    \quad\text{(T-Assign)}

.. math::

    \frac{\Gamma; R \vdash s_1 : \text{Unit}
         \qquad \Gamma; R \vdash s_2 : \text{Unit} }
         {\Gamma; R \vdash s_1 ; s_2 : \text{Unit}}
    \quad\text{(T-Seq)}

.. math::

    \frac{\Gamma; R \vdash e : \text{Bool}
         \qquad \Gamma; R \vdash s_1 : \text{Unit}
         \qquad \Gamma; R \vdash s_2 : \text{Unit} }
         {\Gamma; R \vdash \text{if}\; e\; \text{then}\; s_1\; \text{else}\; s_2 : \text{Unit} }
    \quad\text{(T-If)}

.. math::

    \frac{\Gamma; R \vdash e : \text{Bool}
         \qquad \Gamma; R \vdash s : \text{Unit} }
         {\Gamma; R \vdash \text{while}\; e\; \text{then}\; s : \text{Unit}  }
    \quad\text{(T-While)}

.. math::

    \frac{\Gamma; R_2 \vdash s : \text{Unit}}
         {\Gamma; R_1 \vdash \text{with}\; R_2\; \text{then}\; s : \text{Unit} }
    \quad\text{(T-Context)}

.. math::

    \frac{\Gamma; R \vdash e : T}
         {\Gamma; R \vdash \text{ret}\; e : \text{Unit} }
    \quad\text{(T-Ret)}

.. math::

    \frac{\Gamma; R \vdash e : T}
         {\Gamma, \rho : T; R \vdash \text{ret}\; e : \text{Unit} }
    \quad\text{(T-Ret)}
