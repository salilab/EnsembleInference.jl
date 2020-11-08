_pdmat(M, Σ) = PDMats.PDMat(Σ)
_pdmat(M, Σ::PDMats.AbstractPDMat) = Σ
_pdmat(M, Σ::Diagonal) = PDMats.PDiagMat(diag(Σ))
_pdmat(M, Σ::UniformScaling) = PDMats.ScalMat(Manifolds.manifold_dimension(M), Σ.λ)
