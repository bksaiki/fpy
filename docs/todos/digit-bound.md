# Digit-bound inference

Relational bounds on the precision of a value rounded at a run-time position
(`fpy2/analysis/digit_bound/`), read by format inference as a reduced product.
This document is the contract: each rule states its **obligation**, what must
hold of every execution for the constraint it adds to be valid.  A rule the
code applies with no obligation here is a bug in this document.

## 1. Problem

A rounding under a context whose position is computed at run time,
`with MPFixedContext(n): y = round(x)` with `n = max(es) - F - 1`, has a format
no non-relational domain can bound: `n` ranges over hundreds of values and so
does `x`'s exponent, but they move *together*.  The analysis bounds the
**precision** of such a value -- the digits between its leading digit and its
last -- by tracking linear relations between integer exponents.

## 2. Concrete semantics

**Numbers.**  For a real `v`: `E(v) = floor(log2 |v|)` (`logb`) for finite
non-zero `v`, `E(0) = -inf`, `E(v) = +inf` for an infinity or NaN; `G(v)`, its
last digit, the largest `g` with `v` a multiple of `2^g`, for finite non-zero
`v`, else `+inf`; `P(v) = E(v) - G(v) + 1`.

**Rounding.**  Every operation rounds under the context active where it runs:
`y = rnd_C(op(args))`, integer arithmetic and `logb` included.  `sum(xs)` is a
left fold: `n - 1` roundings for `n` elements, none for one.  A context may
overflow (to an infinity, wrapping, saturating, raising), underflow, round
stochastically, or take parameters known only at run time.

**Executions.**  A definition nested in binders -- `for` and `while` loops,
comprehension clauses -- takes one value per iteration vector that ran.  An
instance **exists** only where its statement ran.

**Lists are references.**  A list value is a reference to a heap object.
`ys = xs` copies the reference; `ys[k] = v` changes the object, and so what
every reference to it reads; a callee receives the caller's objects; `for x in
xs` reads the live object.  `alias.py` computes **regions**, which references
may share an object (over-approximate).

## 3. The abstract domain

**Variables.**  A real `v` has `msb(v)`, exactly `E(v)`, and `lsb(v)`, a lower
bound on `G(v)`; an integer `n` has `val(n)`, exactly `n`.  A list's terms are
its innermost element's.  `msb` is exact because `logb` reads it as a value.
An infinite `E` or `val` is an arbitrarily low or high integer, and a bound in
the false direction is guarded (Literals).

**Dimensions.**  What an entity varies over: each binder around it, and for a
list, one element index per nesting level.  **Correspondence** equates two
dimensions where the program makes their indices equal:

| construct | equation |
|---|---|
| `for x in xs`; `xs[i]`, `i` binder `b`'s target | `b` = `xs`'s element |
| a builder's store `out[k] = v` (Lists) | `b` = `out`'s element |
| a comprehension of one clause | the clause = the result's element |
| `zip(xs, ys)` | its element = each operand's |
| a use, a branch phi of lists | the element = the definition's, each arm's |
| a row `xss[i]`, `for row in xss`, a stored row | the row's element = `xss`'s inner one |
| a literal or comprehension of lists | the result's inner element = each row's |
| a parameter, a call's result | the argument's, each returned list's |

The classes are the **index spaces**.  Invariant: **no class holds two
dimensions one nested in the other** -- a binder in those around it, a list's
element in the binders it varies with, an inner element in its outer one.  An
equation the invariant refuses is not stated, and the read it came from is a
copy.  A variable's **signature** is the classes it varies over.

**Pointwise and summary.**  A constraint over variables sharing a class holds
at every index of it, at that index in each (Halbwachs and Peron, PLDI 2008).
Over variables of different classes it holds for every pair, one from each:
the variables are summaries (Gopan et al., TACAS 2004).

**Literals.**  Boolean variables with signatures, standing for a fact at each
point.  Each variable `t` has `low(t)`, "`t` is not `-inf`", and `high(t)`,
"`t` is not `+inf`" (for `msb`, the value is finite).  A literal is false where
its entity does not exist.

**States and concretization.**  A state is constraints `t1 <= t2`, `t1 <=
max(ts)`, `t1 >= min(ts)` over affine terms, each optionally guarded by a
conjunction of literals, and implications between literals.  An execution is in
its concretization iff some assignment -- one integer per variable per point of
its signature, one boolean per literal -- is **exact** where an instance exists
(`msb` its `E`, `val` its value, `lsb` at most its `G`, literals their facts'
truth), **virtual** where it does not (what the operation would compute from
its operands' assignments there, or no value: `msb` low, `lsb` high, literals
false), and satisfies every constraint at every point.  The virtual clause
lets an operator's theorem hold without a guard; a fact true only on a path is
guarded by that path's literal.

## 4. Transfer rules

### Pointwise facts

Exact operations, each a theorem about the exact result `z`:

| construct | constraint |
|---|---|
| `a * b` | `E(z) <= E(a) + E(b) + 1` (`+ 0` with a power of two); `G(z) >= G(a) + G(b)` |
| `a + b`, `a - b` | `E(z) <= max(E(a), E(b)) + 1`; `G(z) >= min(G(a), G(b))` |
| `logb(x)` | `z = E(x)` |
| `2 ** n` | `E(z) = n`, `G(z) = n` |
| `max(a, b)` of integers | `z >= a`, `z >= b` |

Rounding `y = rnd_C(z)`; `c = 0` for deterministic `RTZ`, else `1`:

| context | `msb(y)` | `lsb(y)` |
|---|---|---|
| `REAL` | `= E(z)` | `>= G(z)` |
| `MPFixedContext(n)` | `<= max(E(z) + c, n + 1)` | `>= max(G(z), n + 1)` |
| bounded fixed, raising or to an infinity | as `MPFixedContext`, under `high(y)` | as `MPFixedContext` |
| bounded fixed, wrapping or saturating | none | `>=` its position plus one |
| float, precision `p`, least exponent `expmin` | `<= max(E(z) + c, expmin)`, under `high(y)` where it can overflow; to nearest, `E(z) + c` | `>= max(G(z), msb(y) - p + 1)` |
| unknown class or parameters | none | none |

`val(y) = z` only where `C` represents every integer in `z`'s range.

**`sum(xs)`**: one element is `xs[0]`.  For `n >= 2`, the exact sum's `E <=
max_i E(xs[i]) + ceil(log2 n)` on a witness (Aggregates); under `RTZ` that
bounds `y`; under a carrying mode each of the `n - 1` roundings adds a quantum
(fixed at `q`: `msb(y) <= max(X + 1, q + 1 + ceil(log2 n))`) or a relative
unit (float: `X + 1` while `n - 1 <= 2^(p - 2)`, `max(X + 2, expmin + 1 +
ceil(log2 n))` with subnormals, else none).  `lsb(y) >= min(lsb(xs))`, and the
context's floor where `n >= 2`.

**Seeds** from format inference -- `msb(v) <= hi`, `lsb(v) >= lo`, `val`'s range
-- bound the *finite* values the format admits, where the site runs: an upper
bound under the term's `high` where the format admits a non-finite value, a
lower one under its `low` where it may be `-inf`, and both under the literals
of the arms around the site.  One function states every seed.

Obligation: each row is a theorem of the operation for every exact or virtual
assignment; the guards cover exactly where it is not.

### Finiteness

Implications between literals at one point.  **Backward**, through strict
operations (`+`, `-`, `*`, `round`, `sqrt`, `neg`, `abs`, `trunc` and kin, `2
** k` in `k`, a dividend, `sum`'s elements): `high(y)` implies the operand's.
**Forward**, through total operations under a context that does not overflow,
and through any value that is one of its sources (a phi, a join, a merged
return, a selection): the operands' `high` implies the result's.  A `with`
establishes what constructing its context requires (`MPFixedContext(e - 10)`
needs `e` finite); an `if` arm runs under a literal of its own.  A branch phi
is the taken arm's value, so `high(phi)` implies whatever both arms imply: what
each arm's `high` implies, or a value its test proves finite (`abs(x) < t`
proves `x` finite).  An exact rule
needs `high` of its result only where finite operands can give a non-finite
one.

### Aggregates

**Envelope**: `m = max(xs)` gives `val(m) >= val(xs)`, true of every element.
**Upper bound**: `E(sum xs) <= max_i E(xs[i]) + ceil(log2 n)` is a fact about
the worst element, stated on a copy of the element at a witness of its own.

### Lists

A list is an **index function** where every store into its region writes
`out[k] = v` at the `k`-th iteration of one loop, the innermost around it, for
every `k` (the loop's own index, or a count of its iterations), into the
region's one object, an `empty` that entered that loop; and no other loop
carries it, that is, reads its phi: a list made anew in each iteration is
carried by none.  A list holding rows that are not index functions is not one.
Comprehensions, literals, parameters the function does not write, call
results, their rows, and the loops normalization lowers comprehensions to are
index functions, wherever they are defined.

- **An index function** relates element for element: its uses share its
  terms, the equations of section 3 hold, and reads follow Element reads.  A
  built list's element `k` is what iteration `k` stored, wherever a read sees
  it -- an element not yet written is uninitialized -- so the definitions
  outside the builder's body share one set of terms, its builder's index is
  its element, and as the loop ends the element is stated to be what the body
  leaves (inside it, each path's definition is the value stored, merged as
  branches merge): `msb` equal, `val` equal, `lsb` at least.  Where the
  invariant refuses the builder's index the list's element, it is what some
  iteration left, bounded above only.  A read inside the builder anywhere but
  at its own index is seeded alone: it is stated before the builder's terms
  are.
- **Any other list** -- written in place, written by a callee, built in pieces,
  carried by a loop, aliased and written -- is a **region summary**.  Each
  read of it is bounded by every value entering its region: each allocation's elements (a
  parameter's, a comprehension's, a literal's, a call's) and every value
  stored, each a copy at a witness of the read's own over every dimension the
  value varies with -- but the loops every object in the region is made in,
  where no phi carries one past an iteration: there the read sees that
  iteration's objects alone.  A region a callee may write has no bound but its
  seed.

Obligation: an index function's definition denotes its object at every use
-- its stores build it, and an element read before it is written is
uninitialized, which no bound has to cover.  A region summary's read is one of
the values that entered its region, at some point and index: every element of
every object in a region was put there by an allocation or a store into it.

### Element reads

An index function `xs[e]` reads its representative where `e` is the target of a
binder the invariant lets correspond with `xs`'s element, and a **copy** at one
index anywhere else: a constant, `a + s * k`, a part `range(a, b, s)` (one copy
per index set as one binder enumerates it), or a binder's target the invariant
refuses (a copy shared by every list read there).  Two reads at one point
share a copy.  Obligation: the copy's point is the index read (section 6).

### Control

- **Branches** join their arms: `phi <= max(arms)`, `phi >= min(arms)`.  Where
  the test proves a value zero (`x == 0`, `a or b`, `all(...)` over a
  comprehension whose element has one zero path), the merge states the
  non-zero arm's bound below; the else arm proves a single-atom test's value
  non-zero.  An arm reached only where some `d` is not finite is dropped under
  `high(d)`.  An arm with no finite non-zero value (`x * 0`) has no digits and
  is dropped, under `high(phi)` where it may not be finite.
- **A `for` loop** is one generic iteration.  A carried scalar's phi joins the
  pre-loop value with the body's at a copy over the loop, the iteration
  before; past the loop it is the pre-loop value or a copy at the last
  iteration.  A list it carries
  is a summary (Lists).
- **A `while` loop** is a binder no copy moves: its phis are fresh, seeded.

### Calls and returns

The callee is instantiated at the call site: its variables vary with the
site's binders plus its own; its parameters share the arguments' terms and
element spaces.  A call's result is the value of the return that ran: each
return's terms copied at a witness over the callee's binders, joined.  A return
that cannot produce a finite value is dropped under the result's `high`.  A
callee analyzed once for several sites holds at another only where that
site's arguments satisfy what it assumed.

## 5. Queries and soundness

    prec(y) = max { sigma(msb(y)) - sigma(lsb(y)) + 1 : sigma satisfies Enc(Phi), A(y) }

`A(y)`: `low(msb y)` and `high(msb y)` (the instance is finite and non-zero),
what each `with` around `y` establishes, the literals of the arms around it,
the definitions value-class inference proves finite there, what its call
sites assume, and what follows by the implications.

**Theorem.**  For any execution on inputs in their declared formats and any
instance `y[i]` of a rounding with a finite non-zero value, if format
inference's bounds contain every finite value, `P(y[i]) <= prec(y)`.

## 6. The encoding

A summary is one set of variables, its **representative**; a copy (an element
read, a witness) is another, renaming the variables -- and literals -- whose
signature meets a moved class, with every constraint over them replayed.
A literal of a variable that moves renames with it; the variable is copied
where the literal is met first.

**Lemma (copies).**  For any execution in the concretization and each copy at
the point its key denotes, the exact or virtual assignment there satisfies
`Enc(Phi)`; where the key denotes no point, the copy takes the no-value
assignment.  A copy renames exactly the variables varying with the moved
classes; two copies sharing a key share a point; by the invariant, moving a
class moves one index.  Renaming fewer is unsound; renaming more is safe only
for a copy no other read shares.

## Status

Checked by `tests/unit/analysis/digit_bound_oracle.py`, which runs each program
on adversarial inputs and holds every value against the store, and by its
generator.  Known failures are pinned in `PENDING` in
`test_digit_bound_oracle.py`.
