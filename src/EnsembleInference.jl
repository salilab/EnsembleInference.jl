module EnsembleInference

using Distributions: Distributions
using LinearAlgebra
using Manifolds: Manifolds, SpecialEuclidean, SpecialOrthogonal, TranslationGroup
using PDMats: PDMats
using Random
using Statistics: Statistics
using StatsBase: StatsBase

const SO3 = SpecialOrthogonal(3)
const SE3 = SpecialEuclidean(3)

# this type is defined to avoid Manifolds's complicated type vector space type hierarchy and
# to avoid choosing a point representation
struct LieAlgebra{G}
    group::G
end

const so3 = LieAlgebra(SO3)
const se3 = LieAlgebra(SE3)

const SHAPESPEC_SE3 = Manifolds.ShapeSpecification(
    Manifolds.StaticReshaper(), Manifolds.base_manifold(SpecialEuclidean(3)).manifolds...
)

"""
    inversion(d::Distribution)

Given a distribution ``d`` of elements ``g ∈ G``, where ``G`` is a group, the inversion of
``d`` is the corresponding distribution of ``g^{-1}``.
"""
inversion

se3_array(x) = Manifolds.ProductArray(SHAPESPEC_SE3, x)

include("utility.jl")
include("dirac.jl")
include("haar.jl")
include("diffusion_normal.jl")

export SO3, so3, SE3, se3
export DiffusionNormal, DiracDelta, Haar
export inversion, se3_array

end # module
