module EnsembleInference

using Distributions: Distributions
using LinearAlgebra
using Manifolds: Manifolds, SpecialEuclidean, SpecialOrthogonal, TranslationGroup
using Random
using Statistics: Statistics
using StatsBase: StatsBase

const SHAPESPEC_SE3 = Manifolds.ShapeSpecification(
    Manifolds.StaticReshaper(),
    Manifolds.base_manifold(SpecialEuclidean(3)).manifolds...,
)

se3_array(x) = Manifolds.ProductArray(SHAPESPEC_SE3, x)

include("dirac.jl")
include("haar.jl")

export Dirac, Haar
export se3_array

end # module
