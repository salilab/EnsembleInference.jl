using Distributions, LinearAlgebra, Manifolds, Random, Statistics, StatsBase, Test

@testset "Haar" begin
    M = TranslationGroup(3)
    @testset "constructor" begin
        p = randn(3)
        d = Haar(M, p)
        @test d isa Haar
        @test d isa Manifolds.MPointDistribution{typeof(M)}
        @test d.point == p
        @test d.manifold === M
    end

    @testset "eltype" begin
        p = randn(3)
        d = Haar(M, p)
        @test eltype(d) === typeof(p)
    end

    @testset "logpdf" begin
        p = randn(3)
        d = Haar(M, p)
        p2 = Haar(M, p)
        @inferred logpdf(d, p2)
        @test isreal(logpdf(d, p2))
        @test iszero(logpdf(d, p2))
    end

    @testset "insupport" begin
        p = randn(3)
        d = Haar(M, p)
        @test insupport(d, p)
        @test !insupport(d, randn(4))
    end

    @testset "convolve" begin
        G = TranslationGroup(3)
        p1, p2 = randn(3), randn(3)
        d1 = Haar(G, p1)
        d2 = Haar(G, p2)
        d12 = convolve(d1, d2)
        @test d12 isa typeof(d1)
        @test isapprox(G, d12.point, compose(G, p1, p2))
    end

    @testset "inversion" begin
        p = randn(3)
        d = Haar(M, p)
        @test inversion(d) === d
    end

    @testset "rand! SpecialOrthogonal" begin
        M = SpecialOrthogonal(3)
        p = exp_lie(M, hat(M, Matrix(Diagonal(ones(3))), randn(3)))
        q = Matrix{Float64}(undef, 3, 3)
        d = Haar(M, p)
        @test rand!(d, q) === q
        @test is_point(M, q, true)
    end

    @testset "rand SpecialOrthogonal" begin
        p = exp_lie(M, hat(M, Matrix(Diagonal(ones(3))), randn(3)))
        d = Haar(M, p)
        @inferred rand(d)
        q = rand(d)
        @test is_point(M, q, true)

        ps = rand(d, 2)
        @test typeof(ps) == Vector{typeof(p)}
        @test length(ps) == 2
        @test is_point(M, ps[1], true)
        @test is_point(M, ps[2], true)
    end
end
