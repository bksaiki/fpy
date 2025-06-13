Type system
================

To understand the FPy type system,
we first describe the type system of :math:`\lambda_{FPy}`,
a typed lambda calculus demonstrating the core features of FPy.

:math:`\lambda_{FPy}`
^^^^^^^^^^^^^^^^^^^^^^

The :math:`\lambda_{FPy}` language extends the simply typed lambda calculus (STLC)
with *tuple expressions* :math:`(\; e_1, \ldots, e_n \;)`,
*tensor expressions* :math:`[\; e_1, \ldots, e_n \;]`,
and *context expressions* :math:`\text{with}\; R\; \text{in}\; e`.
All expressions in :math:`\lambda_{FPy}` have an implicit *rounding context*
which defines how real numbers (or real-valued operations) should be rounded.
The context expression allows explicit control over the current context.
The terminals of :math:`\lambda_{FPy}` include
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
      & \mid & \text{let}\; x = e_1 \; \text{in} \; e_2 \\
      & \mid & (\; e_1, \ldots, e_n \;) \\
      & \mid & [\; e_1, \ldots, e_n \;] \\
      & \mid & \text{with}\; R \;\text{in}\; e \\
    \end{array}

The types of :math:`\lambda_{FPy}` include two base types and three compound types.
The base types include a boolean type (:math:`\texttt{Bool}`) and
a real number type (:math:`\texttt{Real}\; R`).
The compound types include function types (:math:`T_1 \overset{\small R}{\rightarrow} T_2`),
product types (:math:`T_1 \times \cdots \times T_n`),
and tensor types (:math:`\texttt{Tensor}\; d\; T`),
where :math:`d` is a non-negative integer.

.. math::

    \begin{array}{rcl}
    T & ::= & \texttt{Bool} \\
      & \mid & \texttt{Real}\; R \\
      & \mid & T_1 \overset{\small R}{\rightarrow} T_2 \\
      & \mid & T_1 \times \cdots \times T_n \\
      & \mid & \texttt{Tensor}\; d\; T \\
    \end{array}

The real number type is parameterized by the rounding context, :math:`R`.
Thus, the type :math:`\texttt{Real}\; R` may be read as the type
of real numbers rounded under the rounding context :math:`R`.
Function type are also parameterized by a rounding context,
which may be viewed as a *coeffect* in the type system:
functions in :math:`\lambda_{FPy}` require a rounding context from the caller.
Unlike other coeffect systems,
a coeffect in this system consists of at most one rounding context:
rounding contexts are not additive.
A product type represents the usual heterogeneous tuple type,
while a tensor type represents a homogeneous tuple type of some size :math:`d`.

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

.. T-Let
.. math::

    \frac{\Gamma\Coeff R \vdash e_1 : T_1
         \qquad \Gamma, x : T_1 \Coeff R \vdash e_2 : T_2}
         {\Gamma\Coeff R \vdash \text{let}\; x = e_1 \; \text{in} \; e_2 : T_2}
    \quad\text{(T-Let)}\\

.. T-Tuple
.. math::

    \frac{\Gamma\Coeff R \vdash e_1 : T_1 \qquad \ldots \qquad \Gamma\Coeff R \vdash e_n : T_n}
         {\Gamma\Coeff R \vdash (\; e_1, \ldots, e_n \;) : T_1 \times \cdots \times T_n}
    \quad\text{(T-Tuple)}\\

.. T-Tensor
.. math::

    \frac{\Gamma\Coeff R \vdash e_1 : T \qquad \ldots \qquad \Gamma\Coeff R \vdash e_d : T}
         {\Gamma\Coeff R \vdash [\; e_1, \ldots, e_d \;] : \texttt{Tensor}\; d\; T}
    \quad\text{(T-Tensor)}\\

.. T-Ctx
.. math::

    \frac{\Gamma\Coeff R_2 \vdash e : T}
         {\Gamma\Coeff R_1 \vdash \text{with}\; R_2\; \text{in}\; e : T}
    \quad\text{(T-Ctx)}\\

By convention, we extend the usual typing context :math:`\Gamma`
with a coeffect annotation, :math:`R`, denoted with :math:`\Gamma \Coeff R`.
We represent the lack of a rounding context with :math:`\varepsilon`.
At any point, the rounding context may be dropped.

Despite the presence of rounding contexts,
most of the typing rules are similar to those of the STLC.
Boolean constants (:math:`\text{T-True}`, :math:`\text{T-False}`)
are not real-valued so rounding does not apply.
Similarly, variables (:math:`\text{T-Var}`) require no rounding context
since during evaluation, the value of the variable is sustituted as-is, without rounding.
Let expressions (:math:`\text{T-Let}`) are typed
under the same rounding context as their value and body.
Similarly, tuple expressions (:math:`\text{T-Tuple}`) and tensor expressions (:math:`\text{T-Tensor}`)
are also typed under the same rounding context as their elements;
the only difference between the tuple and tensor expressions
is that elements of a tuple may have different types,
while the tensor expressions must have the same type.

The typing rules that involve rounding contexts
are real number constants (:math:`\text{T-Real}`),
function abstractions (:math:`\text{T-Abs}`),
and function applications (:math:`\text{T-App}`).
For a real number constant,
the constant is always rounded under the local rounding context :math:`R`.
For abstraction,
the abstraction body :math:`e` is typed under
a fresh rounding context :math:`R_1` that may be different
from the rounding context :math:`R_2` outside the abstraction.
For a function :math:`f`,
the coeffect :math:`R_1` is handled at each application site:
:math:`f` is instantiated using the local rounding context :math:`R`,
and the return type is :math:`[R_1 \rightarrow R]\; T_2`.
Context expressions (:math:`\text{T-Ctx}`) change
the rounding context: if :math:`R_2` is the rounding context
of the expression, then :math:`R_1` is the new rounding context
for the body expression :math:`e`.

Tuples may only be destructured by pattern matching;
let expressions in FPy are extended to
:math:`\text{let}\; p = e_1 \;\text{in}\; e_2`
where :math:`p` is the usual pattern matching syntax.
The typing rules for let expressions must be altered accordingly.

.. patterns
.. math::

    \begin{array}{rcl}
    p & ::= & x \\
      & \mid & (p_1, \ldots, p_n) \\
    \end{array}

On the other hand, tensor expressions have no syntax for
destructuring since tensor are parameterized by a size :math:`d`.
Rather, :math:`\lambda_{\text{FPy}}` provides some
primitive tensor operations:

.. math::

    \begin{array}{rcl}
    \texttt{dim}  &:& \texttt{Tensor}\; d\; T \overset{\small R}{\rightarrow} \texttt{Real}\; R \\
    \texttt{ref}  &:& \texttt{Tensor}\; d\; T \overset{\small \varepsilon}{\rightarrow} \texttt{Real}\; R_2 \overset{\small \varepsilon}{\rightarrow} T \\
    \texttt{size} &:& \texttt{Tensor}\; d\; T \overset{\small \varepsilon}{\rightarrow} \texttt{Real}\; R_2 \overset{\small R_1}{\rightarrow} \texttt{Real}\; R_1 \\
    \end{array}

The :math:`\texttt{dim}` operation returns the number of dimensions of the tensor.
The :math:`\texttt{ref}` operation returns the value at a specific index in the tensor.
The :math:`\texttt{size}` operation returns the size of the tensor at a particular dimension.
