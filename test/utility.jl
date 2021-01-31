using EnsembleInference

@testset "Utility functions" begin
    @testset "kron_delta" begin
        @test EnsembleInference.kron_delta(0, 0)
        @test EnsembleInference.kron_delta(1, 1)
        @test EnsembleInference.kron_delta(2, 2)
        @test !EnsembleInference.kron_delta(0, 1)
        @test !EnsembleInference.kron_delta(1, 0)
        @test !EnsembleInference.kron_delta(0, 2)
        @test !EnsembleInference.kron_delta(2, 0)
        @test !EnsembleInference.kron_delta(1, 2)
        @test !EnsembleInference.kron_delta(2, 1)
    end
end
