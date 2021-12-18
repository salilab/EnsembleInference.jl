using Distributions, LinearAlgebra, Manifolds, PDMats, Random, Statistics, StatsBase, Test

@testset "DiffusionNormal" begin
    M = SpecialOrthogonal(3)
    @testset "constructors" begin
        μ = exp_lie(M, hat(M, Matrix(Diagonal(ones(3))), randn(3)))
        Σ = exp(Symmetric(randn(3, 3)))
        @inferred DiffusionNormal(M, μ, Σ)
        d = DiffusionNormal(M, μ, Σ)
        @test d isa DiffusionNormal
        @test d.Σ isa PDMats.PDMat
        @test d.μ == μ
        @test d.e === Identity(M)
        @test d.direction === Manifolds.LeftAction()

        Σ = rand() * I
        @inferred DiffusionNormal(M, μ, Σ)
        d = DiffusionNormal(M, μ, Σ; direction=Manifolds.RightAction())
        @test d.Σ isa PDMats.ScalMat
        @test d.direction === Manifolds.RightAction()

        Σ = Diagonal(rand(3))
        @inferred DiffusionNormal(M, μ, Σ)
        d = DiffusionNormal(M, μ, Σ; e=Matrix{Float32}(Matrix(Diagonal(ones(3)))))
        @test d.Σ isa PDMats.PDiagMat
        @test d.e isa Matrix{Float32}
    end

    @testset "eltype" begin
        μ = exp_lie(M, hat(M, Matrix(Diagonal(ones(3))), randn(3)))
        Σ = exp(Symmetric(randn(3, 3)))
        d = DiffusionNormal(M, μ, Σ)
        @test eltype(d) === typeof(μ)
    end

    @testset "inversion" begin
        μ = exp_lie(M, hat(M, Matrix(Diagonal(ones(3))), randn(3)))
        Σ = exp(Symmetric(randn(3, 3)))
        d = DiffusionNormal(M, μ, Σ)
        dinv = inversion(d)
        @test dinv.μ == inv(M, μ)
        @test dinv.Σ == d.Σ
        @test dinv.basis === d.basis
        @test dinv.e == d.e
        @test dinv.direction === Manifolds.RightAction()
    end

    @testset "rand! SpecialOrthogonal" begin
        M = SpecialOrthogonal(3)
        μ = exp_lie(M, hat(M, Matrix(Diagonal(ones(3))), randn(3)))
        Σ = exp(Symmetric(randn(3, 3)))
        d = DiffusionNormal(M, μ, Σ)
        q = Matrix{Float64}(undef, 3, 3)
        d = DiffusionNormal(M, μ, Σ)
        @test rand!(d, q) === q
        @test is_point(M, q, true)
    end

    @testset "rand SpecialOrthogonal" begin
        M = SpecialOrthogonal(3)
        μ = exp_lie(M, hat(M, Matrix(Diagonal(ones(3))), randn(3)))
        Σ = exp(Symmetric(randn(3, 3)))
        d = DiffusionNormal(M, μ, Σ)
        @inferred rand(d)
        q = rand(d)
        @test is_point(M, q, true)

        ps = rand(d, 2)
        @test typeof(ps) == Vector{typeof(μ)}
        @test length(ps) == 2
        @test is_point(M, ps[1], true)
        @test is_point(M, ps[2], true)
    end

    @testset "insupport" begin
        μ = exp_lie(M, hat(M, Matrix(Diagonal(ones(3))), randn(3)))
        Σ = exp(Symmetric(randn(3, 3)))
        d = DiffusionNormal(M, μ, Σ)
        p = exp_lie(M, hat(M, Matrix(Diagonal(ones(3))), randn(3)))
        @test insupport(d, p)
        @test !insupport(d, randn(3))
        @test !insupport(d, randn(4, 4))
        @test !insupport(d, zeros(3, 3))
        @test !insupport(d, Matrix(Diagonal([-1.0, 1.0, 1.0])))
    end

    @testset "mode" begin
        μ = exp_lie(M, hat(M, Matrix(Diagonal(ones(3))), randn(3)))
        Σ = exp(Symmetric(randn(3, 3)))
        d = DiffusionNormal(M, μ, Σ)
        @test mode(d) == μ
    end
end
