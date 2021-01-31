using EnsembleInference
using EnsembleInference: E1, E2, E3, representation_block
using LinearAlgebra
using Manifolds: Manifolds
using Rotations

isskewhermitian(A) = ishermitian(im * A)

isunitary(A) = A * A' ≈ I && abs(det(A)) ≈ 1

# explicit little-d matrix entries, from Tables 4.3-4.6 of
# Varshalovich. Quantum Theory of Angular Momentum. 1998.
# NOTE: the tables order the indices in decreasing order, where we order them in increasing
# order.
function dmat(β, ℓ)
    sinβ, cosβ = sincos(β)
    α = β / 2
    sinα, cosα = sincos(α)
    return dℓ = if ℓ == 0
        [one(cosβ)]
    elseif ℓ == 1//2
        [cosα sinα; -sinα cosα]
    elseif ℓ == 1
        [
            (1 + cosβ)/2 sinβ/√2 (1 - cosβ)/2
            -sinβ/√2 cosβ sinβ/√2
            (1 - cosβ)/2 -sinβ/√2 (1 + cosβ)/2
        ]
    elseif ℓ == 3//2
        [
            cosα^3 √3*sinα*cosα^2 √3*sinα^2*cosα sinα^3
            -√3*sinα*cosα^2 cosα*(3cosα^2 - 2) -sinα*(3sinα^2 - 2) √3*sinα^2*cosα
            √3*sinα^2*cosα sinα*(3sinα^2 - 2) cosα*(3cosα^2 - 2) √3*sinα*cosα^2
            -sinα^3 √3*sinα^2*cosα -√3*sinα*cosα^2 cosα^3
        ]
    else
        throw(DomainError(ℓ, "ℓ must be less than or equal to 3/2"))
    end
end

@testset "Representations" begin
    @testset "so(3) representations" begin
        e = Matrix{Float64}(I, 3, 3)
        basis = Manifolds.DefaultOrthogonalBasis()
        @testset "basis vector E$i" for (i, Ei) in enumerate((E1, E2, E3))
            Xⁱ = Float64.(i .== (1:3))
            Eimat = Manifolds.get_vector(SO3, e, Xⁱ, basis)
            @test @inferred(representation_block(so3, Ei, 0)) ≈ zeros(1, 1)
            @testset "ℓ=$ℓ" for ℓ in 1:0.5:100
                u_Ei = @inferred representation_block(so3, Ei, ℓ)
                u_Eimat = @inferred representation_block(so3, Eimat, ℓ)
                @test u_Ei isa Tridiagonal
                @test u_Eimat isa Tridiagonal
                @test isskewhermitian(u_Ei)
                @test isskewhermitian(u_Eimat)
                @test representation_block(so3, Ei, ℓ) ≈ representation_block(so3, Eimat, ℓ)
            end
        end
        @testset "non-basis vector" begin
            @testset "ℓ=$ℓ" for ℓ in 1:0.5:100
                Xⁱ = randn(3)
                X = Manifolds.get_vector(SO3, e, Xⁱ, basis)
                u1 = @inferred representation_block(so3, Xⁱ, ℓ)
                u2 = @inferred representation_block(so3, X, ℓ)
                u3 = (
                    Xⁱ[1] * representation_block(so3, E1, ℓ) +
                    Xⁱ[2] * representation_block(so3, E2, ℓ) +
                    Xⁱ[3] * representation_block(so3, E3, ℓ)
                )
                @test isskewhermitian(u3)
                @test u1 ≈ u3
                @test u2 ≈ u3
            end
        end
    end

    @testset "SO(3) representations" begin
        # R = Rotations.UnitQuaternion(normalize(randn(4)))
        @testset "basic group properties" begin
            @testset for "ℓ=$ℓ" for ℓ in 0:0.5:20
                R1 = Rotations.UnitQuaternion(normalize(randn(4)))
                R2 = Rotations.UnitQuaternion(normalize(randn(4)))
                R12 = R1 * R2
                # identity
                Ue = @inferred representation_block(SO3, Matrix{Float64}(I, 3, 3), ℓ)
                @test Ue ≈ I
                # composition
                U1 = @inferred representation_block(SO3, R1, ℓ)
                @test isunitary(U1)
                U2 = @inferred representation_block(SO3, R2, ℓ)
                @test isunitary(U2)
                U12 = @inferred representation_block(SO3, R12, ℓ)
                @test isunitary(U12)
                @test U1 * U2 ≈ U12
                # invertible
                @test inv(U1) ≈ representation_block(SO3, inv(R1), ℓ)
                # don't check associativity, because implied by matrix
            end
        end

        @testset "little d matrix" begin
            @testset "ℓ=$ℓ" for ℓ in 0:0.5:1.5
                βs = randn(20)
                for β in βs
                    # Varshalovich uses the ZYZ Euler angle convention, see §1.4.1
                    R = Matrix(Rotations.RotY(β))
                    dℓ = representation_block(SO3, R, ℓ)
                    @test isunitary(dℓ)
                    @test dℓ ≈ dmat(β, ℓ)
                end
            end
        end

        # TODO: test to ensure conventions used are consistent with Varshalovich
        # TODO: use FiniteDifferences to test connection of algebra and group representations
    end
end
