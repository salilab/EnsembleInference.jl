
"""
    expv(t::Number, A, b::AbstractVecOrMat; solver::ODESolver = Tsit5(), kwargs...)

Compute the action of the matrix exponential, `u(t) = exp(t * A) * b`, by solving the
differential equation ``u̇ = A u`` with initial condition ``u(0)=b``.

The solution is computed using `OrdinaryDiffEq.solve` with the provided `solver`.
Remaining `kwargs` are passed to `OrdinaryDiffEq.solve`. The same solver and keywords
are used to compute the pullback when using a ChainRules-compatible automatic
differentiation package.
"""
function expv(
    t,
    A,
    b;
    solver = OrdinaryDiffEq.Tsit5(),
    solve_kwargs::NamedTuple = (; abstol=1e-8, reltol=1e-8),
)
    A isa AbstractMatrix && isdiag(A) && return expv(t, Diagonal(A), b)
    # include time as parameter, which allows us to handle t::Complex
    Tt = real(typeof(t))
    tspan = (zero(Tt), one(Tt))
    params = (A, t)
    problem = OrdinaryDiffEq.ODEProblem(f_expv!, b, tspan, params)
    solve_kwargs = merge(solve_kwargs, (; save_everystep=false))
    w = similar(b, Base.promote_eltype(A, b)) # make return value type-inferrable
    copyto!(w, last(OrdinaryDiffEq.solve(problem, solver; solve_kwargs...)))
    return w
end
expv(t, A::Diagonal, b; kwargs...) = exp.(t .* A.diag) .* b

function ChainRulesCore.rrule(
    ::typeof(expv),
    t,
    A,
    b;
    kwargs...,
)
    w = expv(t, A, b; kwargs...)
    function expv_pullback(Δw)
        ∂t = @thunk expv_rev_t(t, A, w, Δw)
        ∂A = @thunk expv_rev_A(t, A, w, Δw, kwargs...)
        # NOTE: ∂b is computed as part of ∂A, but because ∂A is much more expensive, we
        # recompute here in case the user doesn't need ∂A
        ∂b = @thunk expv_rev_b(t, A, w, Δw; kwargs...)
        return NO_FIELDS, ∂t, ∂A, ∂b
    end
    return w, expv_pullback
end

expv_rev_t(t, A, w, Δw) = conj(dot(Δw, A, w))
expv_rev_t(::Real, A, w, Δw) = real(dot(Δw, A, w))

expv_rev_b(t, A, w, Δw; kwargs...) = expv(conj(t), A', Δw; kwargs...)

function expv_rev_A(
    t,
    A,
    w,
    Δw;
    solver = OrdinaryDiffEq.Tsit5(),
    solve_kwargs = (; abstol=1e-8, reltol=1e-8),
)
    ∂A = similar(A)
    if isdiag(A)
        copyto!(∂A, expv_rev_A(t, Diagonal(A), w, Δw))
        return ∂A
    end
    # solve system backwards, augmented to evolve adjoints to parameters
    # based on Algorithm 1 of https://arxiv.org/abs/1806.07366, though the approach
    # is older
    n = length(w)
    u0 = Matrix{Base.promote_eltypeof(w, Δw, A)}(undef, n, n + 2)
    u0[:, 1] .= w
    u0[:, 2] .= Δw
    fill!(@view(u0[:, 3:n+2]), false)
    solve_kwargs = merge(solve_kwargs, (; save_everystep=false))
    # include time as parameter, which allows us to handle t::Complex
    Tt = real(typeof(t))
    tspan = (one(Tt), zero(Tt)) # reverse time
    params = (A, n, t)
    problem = OrdinaryDiffEq.ODEProblem(f_expv_rev_A!, u0, tspan, params)
    u1 = last(OrdinaryDiffEq.solve(problem, solver; solve_kwargs...))
    ∂A .= conj(t) .* @view(u1[:, 3:n+2])
    return ∂A
end
expv_rev_A(t, A::Diagonal, w, Δw; kwargs...) = outer_sparse!(similar(A), Δw, w, conj(t), false)

f_expv!(du, u, (A, c), t) = mul!(du, A, u, c, false)

function f_expv_rev_A!(du, u, (A, n, c), t)
    z, a = @views u[:, 1], u[:, 2]
    dz, da, dμ = @views du[:, 1], du[:, 2], du[:, 3:n+2]
    mul!(dz, A, z, c, false)
    mul!(da, A', a, -conj(c), false)
    outer_sparse!(dμ, a, z, -1, false)
end

# in-place outer product Z = x α y' + Z β, for vectors x,y and scalars α,β,
# for sparse `Z`, this should be overloaded to respect sparsity pattern and reduce computation
outer_sparse!(Z, x, y, α, β) = mul!(Z, x, y', α, β)
function outer_sparse!(Z::Diagonal, x, y, α, β)
    Z.diag .= Z.diag .* β .+ x .* α .* conj.(y)
    return Z
end
