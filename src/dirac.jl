"""
    DiracDelta{T,TM<:Manifolds.Manifold} <: Manifolds.MPointDistribution{TM}

Dirac distribution around a point ``p``, that is, a distribution whose support is a single
point ``p`` on a manifold ``M``.

# Constructor

    DiracDelta(M::Manifold, p)
"""
struct DiracDelta{P,M<:Manifolds.Manifold} <: Manifolds.MPointDistribution{M}
    manifold::M
    point::P
end

Base.eltype(::Type{<:DiracDelta{T}}) where {T} = T

function Base.rand(::AbstractRNG, s::Random.SamplerTrivial{<:DiracDelta})
    d = s.self
    return copy(d.point)
end

Distributions._rand!(::AbstractRNG, d::DiracDelta, q) = copyto!(q, d.point)

function Distributions.convolve(
    d1::D1, d2::D1
) where {P1,P2,G<:Manifolds.AbstractGroupManifold,D1<:DiracDelta{P1,G},D2<:DiracDelta{P2,G}}
    M = d1.manifold
    p = Manifolds.compose(M, d1.point, d2.point)
    return DiracDelta(M, p)
end

function Distributions.insupport(d::DiracDelta, p)
    M = d.manifold
    return Manifolds.is_manifold_point(M, p) && isapprox(M, p, d.point)
end

Statistics.mean(d::DiracDelta) = d.point

Statistics.median(d::DiracDelta) = Statistics.mean(d)

Statistics.std(d::DiracDelta) = zero(real(Manifolds.number_eltype(d.point)))

Statistics.var(d::DiracDelta) = zero(real(Manifolds.number_eltype(d.point)))

Distributions.mode(d::DiracDelta) = Statistics.mean(d)

inversion(d::DiracDelta) = d
