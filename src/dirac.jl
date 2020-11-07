"""
    Dirac{T,TM<:Manifolds.Manifold} <: Manifolds.MPointDistribution{TM}

Dirac distribution around a point ``p``, that is, a distribution whose support is a single
point ``p`` on a manifold ``M``.

# Constructor

    Dirac(M::Manifold, p)
"""
struct Dirac{P,M<:Manifolds.Manifold} <: Manifolds.MPointDistribution{M}
    manifold::M
    point::P
end

Base.eltype(::Type{<:Dirac{T}}) where {T} = T

function Base.rand(::AbstractRNG, s::Random.SamplerTrivial{<:Dirac})
    d = s.self
    return copy(d.point)
end

Distributions._rand!(::AbstractRNG, d::Dirac, q) = copyto!(q, d.point)

function Distributions.convolve(
    d1::D1,
    d2::D1,
) where {P1,P2,G<:Manifolds.AbstractGroupManifold,D1<:Dirac{P1,G},D2<:Dirac{P2,G}}
    M = d1.manifold
    p = Manifolds.compose(M, d1.point, d2.point)
    return Dirac(M, p)
end

function Distributions.insupport(d::Dirac, p)
    M = d.manifold
    return Manifolds.is_manifold_point(M, p) && isapprox(M, p, d.point)
end

Statistics.mean(d::Dirac) = d.point

Statistics.median(d::Dirac) = Statistics.mean(d)

Statistics.std(d::Dirac) = zero(real(Manifolds.number_eltype(d.point)))

Statistics.var(d::Dirac) = zero(real(Manifolds.number_eltype(d.point)))

Distributions.mode(d::Dirac) = Statistics.mean(d)
