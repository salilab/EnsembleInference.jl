using Distributions, LinearAlgebra, Manifolds, Random, Statistics, StatsBase, Test

@testset "DiracDelta" begin
    M = Euclidean(3)
    @testset "constructor" begin
        p = randn(3)
        d = DiracDelta(M, p)
        @test d isa DiracDelta
        @test d isa Manifolds.MPointDistribution{typeof(M)}
        @test d.point == p
        @test d.manifold === M
    end

    @testset "eltype" begin
        p = randn(3)
        d = DiracDelta(M, p)
        @test eltype(d) === typeof(p)
    end

    @testset "rand!" begin
        p = randn(3)
        q = Vector{Float64}(undef, 3)
        d = DiracDelta(M, p)
        @test rand!(d, q) === q
        @test q == p
    end

    @testset "rand" begin
        p = randn(3)
        d = DiracDelta(M, p)
        @test rand(d) == p

        ps = rand(d, 2)
        @test typeof(ps) == Vector{typeof(p)}
        @test length(ps) == 2
        @test ps[1] == p
        @test ps[2] == p
    end

    @testset "insupport" begin
        p = randn(3)
        d = DiracDelta(M, p)
        @test insupport(d, p)
        @test !insupport(d, randn(3))
    end

    @testset "$f" for f in (mean, median, mode)
        p = randn(3)
        d = DiracDelta(M, p)
        @test f(d) == p
    end

    @testset "$f" for f in (var, std)
        p = randn(3)
        d = DiracDelta(M, p)
        @inferred f(d)
        @test isreal(f(d))
        @test iszero(f(d))

        M2 = Euclidean(3; field=ℂ)
        p2 = randn(ComplexF64, 3)
        d2 = DiracDelta(M2, p2)
        @inferred f(d2)
        @test isreal(f(d2))
        @test iszero(f(d2))
    end

    @testset "convolve" begin
        G = SpecialOrthogonal(3)
        p1, p2 = ntuple(_ -> exp_lie(G, hat(G, Matrix(Diagonal(ones(3))), randn(3))), 2)
        d1 = DiracDelta(G, p1)
        d2 = DiracDelta(G, p2)
        d12 = convolve(d1, d2)
        @test d12 isa typeof(d1)
        @test isapprox(G, d12.point, compose(G, p1, p2))
    end

    @testset "inversion" begin
        p = randn(3)
        d = DiracDelta(M, p)
        @test inversion(d) === d
    end
end
