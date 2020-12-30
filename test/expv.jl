using LinearAlgebra, ChainRulesCore, ChainRulesTestUtils
using ChainRulesTestUtils: rand_tangent

@testset "action of the matrix exponential" begin
    n = 10

    @testset "expv" begin
        @testset "$(TA{T}), n=$n" for TA in (Matrix, Diagonal), T in (Float64, ComplexF64)
            t, A, b = rand(T), TA(randn(T, n, n)), randn(T, n)
            w = @inferred EnsembleInference.expv(t, A, b)
            @test w ≈ exp(t * A) * b
        end
    end

    @testset "expv rrule" begin
        @testset "$(TA{T}), n=$n" for TA in (Matrix, Diagonal), T in (Float64, ComplexF64)
            t, A, b = rand(T), TA(randn(T, n, n)), randn(T, n)
            ∂t, ∂A, ∂b = rand_tangent(t), rand_tangent(A), rand_tangent(b)
            w = EnsembleInference.expv(t, A, b)
            Δw = rand_tangent(w)
            rrule_test(EnsembleInference.expv, Δw, (t, ∂t), (A, ∂A), (b, ∂b))
            if TA <: Complex
                rrule_test(
                    EnsembleInference.expv, Δw, (real(t), real(∂t)), (A, ∂A), (b, ∂b)
                )
            end
            # check type-stable
            _, back = @inferred rrule(EnsembleInference.expv, t, A, b)
            @test_broken @inferred back(Δw)
            _, ∂t, ∂A, ∂b = back(Δw)
            @inferred unthunk(∂t)
            @inferred unthunk(∂A)
            @inferred unthunk(∂b)
        end
    end
end
