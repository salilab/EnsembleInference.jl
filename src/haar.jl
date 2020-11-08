"""
    Haar{T,TM<:Manifolds.AbstractGroupManifold} <: Manifolds.MPointDistribution{TM}

Haar (invariant, i.e. uniform) measure on group manifold ``M``.

# Constructor

    Haar(M::AbstractGroupManifold, p)

Constructs a Haar distribution on the group manifold ``M`` with points represented like the
provided point ``p``, used for random sampling when implemented for compact groups.
"""
struct Haar{P,M<:Manifolds.AbstractGroupManifold} <: Manifolds.MPointDistribution{M}
    manifold::M
    point::P # point stored for random sampling
end

function Distributions.logpdf(d::Haar, p)
    T = real(Base.promote_eltype(
        Manifolds.number_eltype(p), Manifolds.number_eltype(d.point)
    ))
    return zero(T)
end

Base.eltype(::Type{<:Haar{T}}) where {T} = T

function Distributions.convolve(
    d1::D1, d2::D2
) where {P1,P2,G<:Manifolds.AbstractGroupManifold,D1<:Haar{P1,G},D2<:Haar{P2,G}}
    M = d1.manifold
    # compose to promote points
    p = Manifolds.compose(M, d1.point, d2.point)
    return Haar(M, p)
end

Distributions.insupport(d::Haar, p) = Manifolds.is_manifold_point(d.manifold, p)

inversion(d::Haar) = d

## random rotation matrices

function Distributions._rand!(
    rng::AbstractRNG, ::Haar{P,M}, q::AbstractMatrix
) where {P,N,M<:SpecialOrthogonal{N}}
    n = size(q, 1)
    @assert n == N
    randn!(rng, q)
    F = qr(q)
    q .= F.Q .* sign.(diag(F.R)')
    if det(q) < 0 # q is orthogonal but not special; swap first two columns make it special
        for i in 1:N
            @inbounds q[i, 1], q[i, 2] = q[i, 2], q[i, 1]
        end
    end
    return q
end

function Base.rand(
    rng::AbstractRNG, s::Random.SamplerTrivial{<:Haar{P,M}}
) where {P,M<:SpecialOrthogonal}
    d = s.self
    q = Manifolds.allocate_result(d.manifold, rand, d.point)
    return Random.rand!(rng, d, q)
end
