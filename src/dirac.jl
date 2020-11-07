"""
    DiracDistribution{T,TM<:Manifolds.Manifold} <: Manifolds.MPointDistribution{TM}

Dirac distribution around a point ``p``, that is, a distribution whose support is a single
point ``p`` on a manifold ``M``.

# Constructor

    DiracDistribution(M::Manifold, p)
"""
struct DiracDistribution{P,M<:Manifolds.Manifold} <: Manifolds.MPointDistribution{M}
    manifold::M
    point::P
end

function Distributions.logpdf(d::DiracDistribution, p)
    T = real(Base.promote_eltype(
        Manifolds.number_eltype(p),
        Manifolds.number_eltype(d.point),
    ))
    return Distributions.insupport(d.manifold, p) ? zero(T) : T(-Inf)
end

Base.eltype(::Type{<:DiracDistribution{T}}) where {T} = T

function Base.rand(::AbstractRNG, s::Random.SamplerTrivial{<:DiracDistribution})
    d = s.self
    return copy(d.point)
end

function Distributions.convolve(
    d1::D1,
    d2::D1,
) where {
    P1,
    P2,
    G<:Manifolds.AbstractGroupManifold,
    D1<:DiracDistribution{P1,G},
    D2<:DiracDistribution{P2,G},
}
    M = d1.manifold
    p = Manifolds.compose(M, d1.point, d2.point)
    return DiracDistribution(M, p)
end

function Distributions.insupport(d::DiracDistribution, p)
    M = d.manifold
    return Manifolds.is_manifold_point(M, p) && isapprox(M, p, d.point)
end

Statistics.mean(d::DiracDistribution) = d.point

Statistics.median(d::DiracDistribution) = mean(d)

Statistics.var(d::DiracDistribution) = zero(real(Manifolds.number_eltype(d.point)))

Distributions.mode(d::DiracDistribution) = mean(d)
