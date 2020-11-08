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

"""
    inversion(d::Distribution)

Given a distribution ``d`` of elements ``g ∈ G``, where ``G`` is a group, the inversion of
``d`` is the corresponding distribution of ``g^{-1}``.
"""
inversion

se3_array(x) = Manifolds.ProductArray(SHAPESPEC_SE3, x)

include("dirac.jl")
include("haar.jl")

export Dirac, Haar
export inversion, se3_array

end # module
