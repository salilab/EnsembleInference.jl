# irreducible unitary representations of the groups SO(3) and SE(3) and representations of
# their Lie algebras.
# For SO(3), see
# Varshalovich, D. A., Moskalev, A. N. & Khersonskii, V. K. Quantum Theory of Angular
# Momentum. 1988. World Scientific, Singapore. ISBN: 9971-5-0107-4.
# For SO(3) and SE(3), see §12.9 and 12.12 of
# Chirikjian, G. S. Stochastic Models, Information Theory, and Lie Groups, Volume 2. 2012.
# ISBN: 978-0-8176-4943-2. doi: 10.1007/978-0-8176-4944-9.

cln(ℓ, n) = ℓ ≥ n ? sqrt((ℓ - n) * (ℓ + n + 1)) : zero(float(Base.promote_typeof(ℓ, n)))

# TODO: refactor to be in terms of diagonals so we can avoid kronecker delta entirely

# i.e. infinitesimal rotation elements of Varshalovich §4.18.3.
function representation_element(::typeof(so3), ::typeof(E1), ℓ, m, n)
    return im * -(cln(ℓ, -n) * kron_delta(m + 1, n) + cln(ℓ, n) * kron_delta(m - 1, n)) / 2
end
function representation_element(::typeof(so3), ::typeof(E2), ℓ, m, n)
    return (cln(ℓ, -n) * kron_delta(m + 1, n) - cln(ℓ, n) * kron_delta(m - 1, n)) / 2
end
function representation_element(::typeof(so3), ::typeof(E3), ℓ, m, n)
    return im * float(-n * kron_delta(m, n))
end

"""
    representation_block(::typeof(so3), X, ℓ) -> Tridiagonal{Complex}

Compute the size `(2ℓ+1, 2ℓ+1)` block ``ℓ`` of the representation of the element ``X`` of
the element of the Lie algebra ``𝔰𝔬(3)``.

The representation is defined as

```math
u^ℓ(X) = \\lim_{t → 0} \\frac{d}{dt} U^ℓ(\\exp(t X)),
```

where ``U^ℓ(R)`` is the representation of the rotation ``R``, and ``\\exp(R)`` is the group
exponential map on ``\\mathrm{SO}(3)``.
"""
representation_block(::typeof(so3), X, ℓ)

function representation_block(::typeof(so3), Ei::BasisVector{B,I}, ℓ) where {B,I}
    T = I == 2 ? float(eltype(ℓ)) : complex(float(eltype(ℓ)))
    ℓ == 0 && return Tridiagonal(zeros(T, 1, 1))
    dl = representation_element.(Ref(so3), Ref(Ei), ℓ, (1 - ℓ):ℓ, (-ℓ):(ℓ - 1))
    d = representation_element.(Ref(so3), Ref(Ei), ℓ, (-ℓ):ℓ, (-ℓ):ℓ)
    du = .-conj.(dl) # representation is skew-Hermitian
    return Tridiagonal(dl, d, du)
end
function representation_block(::typeof(so3), Xⁱ::AbstractVector{<:Real}, ℓ)
    # because representation_element returns a result with real eltype Float64, we use `T`
    # to avoid promoting if the eltype of Xⁱ is lower precision
    T = complex(float(eltype(Xⁱ)))
    ℓ == 0 && return Tridiagonal(zeros(T, 1, 1))
    m_dl = (1 - ℓ):ℓ
    n_dl = (-ℓ):(ℓ - 1)
    dl = (
        Xⁱ[1] * T.(representation_element.(Ref(so3), Ref(E1), ℓ, m_dl, n_dl)) .+
        Xⁱ[2] * T.(representation_element.(Ref(so3), Ref(E2), ℓ, m_dl, n_dl))
    )
    m_d = n_d = (-ℓ):ℓ
    d = Xⁱ[3] .* T.(representation_element.(Ref(so3), Ref(E3), ℓ, m_d, n_d))
    du = .-conj.(dl) # representation is skew-Hermitian
    return Tridiagonal(dl, d, du)
end
function representation_block(::typeof(so3), X::AbstractMatrix, ℓ)
    # represent the vector X as an actual Vector using the basis
    Xⁱ = Manifolds.vee(SO3, Manifolds.Identity(SO3, one(X)), X)
    return representation_block(so3, Xⁱ, ℓ)
end

"""
    representation_block(::typeof(SO3), p, ℓ) -> Matrix{Complex}

Compute the size `(2ℓ+1, 2ℓ+1)` block ``ℓ`` of the representation of the rotation ``p``.

This block is identical to the ``D^ℓ`` block of the Wigner ``D``-matrix using the
conventions of [^Varshalovich1988].

[^Varshalovich1988]: Varshalovich, D. A., Moskalev, A. N. & Khersonskii, V. K. Quantum Theory of Angular
    Momentum. 1988. World Scientific, Singapore. ISBN: 9971-5-0107-4.
# Examples

To create the "little-``d``" block ``d^ℓ(\\frac{π}{4}) = D^ℓ(0, \\frac{π}{4}, 0)``,

```julia-repl
julia> using EnsembleInference, Rotations

julia> EnsembleInference.representation_block(SO3, RotZXZ(0, π/2, 0), 1)
3×3 Matrix{ComplexF64}:
  0.5+0.0im              0.0-0.707107im  -0.5+0.0im
  0.0-0.707107im  1.2175e-16+0.0im        0.0-0.707107im
 -0.5+0.0im              0.0-0.707107im   0.5+0.0im

julia> EnsembleInference.representation_block(SO3, RotZXZ(0, π/2, 0), 2)
5×5 Matrix{ComplexF64}:
      0.25+0.0im   0.0-0.5im          -0.612372+0.0im           0.0+0.5im               0.25+0.0im
       0.0-0.5im  -0.5-0.0im                0.0-3.24998e-16im  -0.5-0.0im                0.0+0.5im
 -0.612372-0.0im   0.0-2.41689e-16im       -0.5-0.0im           0.0-3.62534e-16im  -0.612372-0.0im
       0.0+0.5im  -0.5+0.0im                0.0-2.03875e-16im  -0.5+0.0im                0.0-0.5im
      0.25+0.0im   0.0+0.5im          -0.612372+0.0im           0.0-0.5im               0.25+0.0im
```
"""
function representation_block(::typeof(SO3), p, ℓ)
    X = Manifolds.group_log(SO3, p)
    u = representation_block(so3, X, ℓ)
    return exp(Matrix(u))
end
