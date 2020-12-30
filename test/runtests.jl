using EnsembleInference
using Test

@testset "EnsembleInference.jl" begin
    include("expv.jl")
    include("dirac.jl")
    include("haar.jl")
    include("diffusion_normal.jl")
end
