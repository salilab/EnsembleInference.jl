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

struct BasisVector{B,I}
    basis::B
end
BasisVector(basis, i::Integer) = BasisVector{typeof(basis),i}(basis)

# basis vectors needed for SO(3) (E1-E3) and SE(3) (E1-E6)
const E1 = BasisVector(Manifolds.DefaultOrthogonalBasis(),1)
const E2 = BasisVector(Manifolds.DefaultOrthogonalBasis(),2)
const E3 = BasisVector(Manifolds.DefaultOrthogonalBasis(),3)
const E4 = BasisVector(Manifolds.DefaultOrthogonalBasis(),4)
const E5 = BasisVector(Manifolds.DefaultOrthogonalBasis(),5)
const E6 = BasisVector(Manifolds.DefaultOrthogonalBasis(),6)

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
include("representation.jl")

export SO3, so3, SE3, se3
export DiffusionNormal, DiracDelta, Haar
export inversion, se3_array

end # module
