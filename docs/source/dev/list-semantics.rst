List Semantics
==============

FPy lists are *mutable*, which the :doc:`core semantics <semantics>` cannot
express: it has no element assignment, and its list values are structures rather
than references.  This page adds both, and supersedes the core's **E-List** and
**E-Ref**.

The split it describes is numpy's and torch's.  **Construction copies** —
building a list materialises its elements, as ``np.stack([a, b])`` does — while
**projection and assignment are references**, as ``arr[0]`` is a view.  So a list
owns what it holds, and the only way to reach into someone else's list is to be
handed a reference to it.

Two syntactic forms carry the difference:

.. math::

   \begin{array}{rcll}
   e & ::= & \cdots \mid e_1[e_2 \mathbin{:} e_3]
       & \text{slice} \\
   s & ::= & \cdots \mid x[e_1] := e_2
       & \text{element assignment}
   \end{array}

Locations and the store
-----------------------

A list value is a *location* :math:`\ell`, and a **store** :math:`\mu` maps
locations to sequences of values.  Tuples stay structures, holding values — and
so possibly locations — directly.

.. math::

   v ::= \cdots \mid \ell
   \qquad
   \mu \in \mathit{Loc} \rightharpoonup v^{*}

Both judgements gain the store, threaded left to right:

* :math:`\langle \sigma, \mu, C, e \rangle \Downarrow v\,;\, \mu'`
* :math:`\langle \sigma, \mu, C, s \rangle \Downarrow_S o\,;\, \mu'`

Every core rule carries :math:`\mu` unchanged — none of them mutates — so this
page states only the rules in which the store does something.

Copying
-------

One auxiliary judgement underlies the rest.
:math:`\mathsf{copy}_\mu(v) = (v'\,;\, \mu')` reads "copying :math:`v` yields
:math:`v'`, allocating in :math:`\mu`".  It **descends through lists** and
**stops at tuples**: a list is an owned container, a tuple a transparent
grouping.

.. math::

   \frac{\mu(\ell) = v_1, \ldots, v_n
         \quad
         \mu_0 = \mu
         \quad
         \mathsf{copy}_{\mu_{i-1}}(v_i) = (v_i'\,;\, \mu_i)
         \quad
         \ell' \notin \mathrm{dom}(\mu_n)}
        {\mathsf{copy}_\mu(\ell) =
         (\ell'\,;\, \mu_n[\ell' \mapsto v_1', \ldots, v_n'])}
   \tag{L-Copy-List}

.. math::

   \frac{v \ne \ell \text{ for every } \ell}
        {\mathsf{copy}_\mu(v) = (v\,;\, \mu)}
   \tag{L-Copy-Other}

**L-Copy-Other** covers tuples, so copying a tuple leaves its fields — including
any locations among them — untouched.  Hence :math:`(\, \ell, n \,)` copied still
holds :math:`\ell`, while :math:`[\, \ell \,]` copied holds a fresh location.

Why the copy stops at tuples
^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Because it makes the rule cut along a joint, and three things follow for free.

*Builtins stay cheap with no exemption.*  ``zip(A, B)`` is a list whose elements
are *tuples* of locations, so copying them costs nothing.  Every list-producing
builtin has tuple or scalar elements, never list ones, so none needs a special
case.

*No provenance discrepancy.*  A tuple built directly and the same tuple copied
into a list share alike, so :math:`\mathsf{fst}(t)` and
:math:`\mathsf{fst}([\, t \,][1])` agree.  Descending would make them differ.

*Multiple return stays constant-time.*  ``ret (xs, n)`` is FPy's main use of
tuples; copying at tuple construction would put a silent linear cost on it.

The price is a weaker guarantee — *a list's own spine is yours*, not *lists are
values*.  :math:`[\, t \,]` does not isolate you from mutations reaching lists
through :math:`t`.

Construction copies
-------------------

A list constructor allocates and copies each element, so the result shares
nothing with its operands.

.. math::

   \frac{\langle \sigma, \mu_{i-1}', C, e_i \rangle \Downarrow v_i\,;\, \mu_i'
         \quad
         \mu_0' = \mu
         \quad
         \mathsf{copy}_{\mu_i'}(v_i) = (v_i'\,;\, \mu_i)
         \quad
         \ell \notin \mathrm{dom}(\mu_n)}
        {\langle \sigma, \mu, C, [\, e_1, \ldots, e_n \,] \rangle \Downarrow
         \ell\,;\, \mu_n[\ell \mapsto v_1', \ldots, v_n']}
   \tag{L-List}

A slice is construction over a sub-range, so it copies too — it is not a view.

.. math::

   \frac{\begin{array}{c}
         \langle \sigma, \mu, C, e_1 \rangle \Downarrow \ell\,;\, \mu_1
         \quad
         \langle \sigma, \mu_1, C, e_2 \rangle \Downarrow j\,;\, \mu_2
         \quad
         \langle \sigma, \mu_2, C, e_3 \rangle \Downarrow k\,;\, \mu_3
         \\
         \mu_3(\ell) = v_1, \ldots, v_m
         \quad
         1 \le j \le k \le m
         \quad
         \mathsf{copy}_{\mu_{i-1}''}(v_i) = (v_i'\,;\, \mu_i'')
         \quad
         \mu_{j-1}'' = \mu_3
         \quad
         \ell' \notin \mathrm{dom}(\mu_k'')
         \end{array}}
        {\langle \sigma, \mu, C, e_1[e_2 \mathbin{:} e_3] \rangle \Downarrow
         \ell'\,;\, \mu_k''[\ell' \mapsto v_j', \ldots, v_k']}
   \tag{L-Slice}

Projection is a reference
-------------------------

Indexing hands back the element *as it stands*.  When that element is itself a
list the result is its location, so the projection is a reference and writing
through it is visible in the container.

.. math::

   \frac{\langle \sigma, \mu, C, e_1 \rangle \Downarrow \ell\,;\, \mu_1
         \quad
         \langle \sigma, \mu_1, C, e_2 \rangle \Downarrow n\,;\, \mu_2
         \quad
         \mu_2(\ell) = v_1, \ldots, v_m
         \quad
         1 \le n \le m}
        {\langle \sigma, \mu, C, e_1[e_2] \rangle \Downarrow v_n\,;\, \mu_2}
   \tag{L-Index}

Nothing copies on binding either.  The core's **E-Assign** carries a location as
it would any other value, so :math:`y := x` gives :math:`x`'s list a second name,
and **E-Ret** hands a location out to the caller.

Element assignment overwrites
-----------------------------

:math:`x[e_1] := e_2` writes *into* the slot rather than replacing it.  Where the
slot holds a list, the copy's contents are moved into the location already there,
which keeps that location's identity — so a projection taken earlier sees the
write.

.. math::

   \frac{\begin{array}{c}
         \sigma(x) = \ell
         \quad
         \langle \sigma, \mu, C, e_1 \rangle \Downarrow n\,;\, \mu_1
         \quad
         \langle \sigma, \mu_1, C, e_2 \rangle \Downarrow v\,;\, \mu_2
         \\
         \mu_2(\ell) = v_1, \ldots, v_m
         \quad
         1 \le n \le m
         \quad
         v_n = \ell_{\mathrm{slot}}
         \quad
         \mathsf{copy}_{\mu_2}(v) = (\ell_v\,;\, \mu_3)
         \end{array}}
        {\langle \sigma, \mu, C, x[e_1] := e_2 \rangle \Downarrow_S
         \mathsf{normal}\ \sigma\,;\,
         \mu_3[\ell_{\mathrm{slot}} \mapsto \mu_3(\ell_v)]}
   \tag{L-Store-List}

Where the slot holds anything else the distinction is unobservable, since no
other value has identity, and the slot is simply updated.

.. math::

   \frac{\begin{array}{c}
         \sigma(x) = \ell
         \quad
         \langle \sigma, \mu, C, e_1 \rangle \Downarrow n\,;\, \mu_1
         \quad
         \langle \sigma, \mu_1, C, e_2 \rangle \Downarrow v\,;\, \mu_2
         \\
         \mu_2(\ell) = v_1, \ldots, v_m
         \quad
         1 \le n \le m
         \quad
         v_n \ne \ell' \text{ for every } \ell'
         \end{array}}
        {\langle \sigma, \mu, C, x[e_1] := e_2 \rangle \Downarrow_S
         \mathsf{normal}\ \sigma\,;\,
         \mu_2[\ell \mapsto v_1, \ldots, v_{n-1}, v, v_{n+1}, \ldots, v_m]}
   \tag{L-Store-Scalar}

:math:`\sigma` is unchanged in both: an element assignment mutates the store and
never the environment, which is why every other name for that list observes it.

Overwriting says where the values land, not that they are shared — the store
copies, so mutating :math:`e_2`'s list afterwards does not reach the container.

The rules in brief
------------------

* :math:`[\, e_1, \ldots, e_n \,]` and :math:`e_1[e_2 \mathbin{:} e_3]` —
  **copy** each element.
* :math:`x[e] := e'` — **copy** :math:`e'`, overwriting the slot.
* :math:`e_1[e_2]` — reference; mutating it mutates the container.
* :math:`y := x` and :math:`\texttt{ret}\ x` — reference.
* :math:`(\, e_1, \ldots, e_n \,)` — reference; a tuple groups, it does not own.

Not modelled here
-----------------

**Comprehensions** are construction, so they copy exactly as **L-List** does;
they desugar per :doc:`derived-semantics`.

**Overwriting is one level deep.**  **L-Store-List** replaces the slot's own
elements with fresh copies rather than overwriting them in place, so a reference
to something *deeper* than the slot does not survive the store.

**Stuck states.**  An out-of-range index, and a length mismatch between
:math:`\mu_3(\ell_v)` and the slot in **L-Store-List**, have no rule — evaluation
is stuck, as with a failing assertion.  FPy has no error handling, so this is
where an implementation is free to choose; the reference interpreter resizes.
