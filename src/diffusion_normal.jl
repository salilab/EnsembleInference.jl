"""
    DiffusionNormal{Tμ,TΣ,Te,TB,TD,TM<:Manifolds.AbstractGroupManifold} <:
    Manifolds.MPointDistribution{TM}

# Constructor

    DiffusionNormal(
        M::Manifolds.AbstractGroupManifold,
        μ,
        Σ,
        e = Manifolds.Identity(M),
        basis = Manifolds.DefaultOrthogonalBasis(),
        direction = Manifolds.LeftAction()
    )

Construct a `DiffusionNormal` distribution on a group manifold ``M``, with points
represented with the same type as the identity element ``e``, which may be user-specified.
The mode of the distribution is at ``μ``, and ``Σ`` is a diffusion matrix applied to the
coordinates of the Lie algebra ``𝔪 = Tₑ M`` using the orthogonal `basis`.
"""
struct DiffusionNormal{Tμ,TΣ,Te,TB,TD,TM<:Manifolds.AbstractGroupManifold} <:
       Manifolds.MPointDistribution{TM}
    manifold::TM
    μ::Tμ
    Σ::TΣ
    e::Te
    basis::TB
    direction::TD

    function DiffusionNormal(M, μ, Σ, e, basis, direction)
        pdΣ = _pdmat(M, Σ)
        return new{
            typeof(μ),typeof(pdΣ),typeof(e),typeof(basis),typeof(direction),typeof(M)
        }(
            M, μ, pdΣ, e, basis, direction
        )
    end
end

function DiffusionNormal(
    M,
    μ,
    Σ;
    e=Manifolds.Identity(M),
    basis=Manifolds.DefaultOrthogonalBasis(),
    direction=Manifolds.LeftAction(),
)
    return DiffusionNormal(M, μ, Σ, e, basis, direction)
end

const SO3DiffusionNormal{Tμ,TΣ,Te,TB,TD} = DiffusionNormal{
    Tμ,TΣ,Te,TB,TD,SpecialOrthogonal{3}
}
const SE3DiffusionNormal{Tμ,TΣ,Te,TB,TD} = DiffusionNormal{
    Tμ,TΣ,Te,TB,TD,SpecialEuclidean{3}
}

Base.eltype(::Type{<:DiffusionNormal{Tμ}}) where {Tμ} = Tμ

Distributions.insupport(d::DiffusionNormal, p) = Manifolds.is_point(d.manifold, p)

Distributions.mode(d::DiffusionNormal) = d.μ

# TODO: overload convolve for Dirac and DiffusionNormal

function inversion(d::DiffusionNormal)
    M = d.manifold
    μ = inv(M, d.μ)
    direction = Manifolds.switch_direction(d.direction)
    return DiffusionNormal(M, μ, d.Σ, d.e, d.basis, direction)
end

function Distributions._rand!(rng::AbstractRNG, d::DiffusionNormal, q)
    # sample using Euler-Maruyama scheme
    # TODO: use logdetjac of exp_lie to tune n so that
    n = 100
    M = d.manifold
    B = d.basis
    e = d.e
    dliealg = Distributions.MvNormal(d.Σ / n)
    Xᵛ = rand(rng, dliealg)
    X = Manifolds.get_vector_lie(M, Xᵛ, B)
    p = Manifolds.exp_lie!(M, q, X)
    for i in 1:(n - 1)
        rand!(rng, dliealg, Xᵛ)
        Manifolds.get_vector_lie!(M, X, Xᵛ, B)
        Manifolds.exp_lie!(M, p, X)
        Manifolds.compose!(M, q, q, p)
    end
    Manifolds.translate!(M, q, d.μ, q, d.direction)
    return q
end
function Random.rand!(rng::AbstractRNG, qs::AbstractVector, d::DiffusionNormal)
    if all(i -> isassigned(qs, i), eachindex(qs))
        for i in eachindex(qs)
            rand!(rng, d, qs[i])
        end
    else
        for i in eachindex(qs)
            qs[i] = rand(rng, d)
        end
    end
    return qs
end

function Base.rand(rng::AbstractRNG, d::DiffusionNormal)
    q = Manifolds.allocate_result(d.manifold, rand, d.μ)
    return Random.rand!(rng, d, q)
end
