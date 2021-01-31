_pdmat(M, Σ) = PDMats.PDMat(Σ)
_pdmat(M, Σ::PDMats.AbstractPDMat) = Σ
_pdmat(M, Σ::Diagonal) = PDMats.PDiagMat(diag(Σ))
_pdmat(M, Σ::UniformScaling) = PDMats.ScalMat(Manifolds.manifold_dimension(M), Σ.λ)

"""
    kron_delta(i, j) -> Bool

The Kronecker delta function

```math
δ_{i,j} = \\begin{cases}
    0 & \\text{if } i ≠ j\\\\
    1 & \\text{if } i = j
\\end{cases}
```
"""
kron_delta(i, j) = i == j
