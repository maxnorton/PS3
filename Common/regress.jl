module Regress

export ols, iv, nls

using Match, Parameters
using Optim, ForwardDiff, Distributions

@with_kw mutable struct Params
    VarType::Symbol = :Default
    Stats::Bool = true
    IncludeConst::Bool = true
    Alpha::Float64 = 0.05
    OptimOpts::Optim.Options = Optim.Options()
end

struct Stats
    AVar
    SE
    CI
    Resid
end

# Standard normal distribution
snd = Normal();

function add_const(n, x...)
    o = ones(n); map(y -> [o y], x)
end

"""
OLS regression.

# Arguments
- `y::Vector`: dependent variable.
- `x::Matrix`: regressors.
- `params::Params`: various parameters.
"""
function ols(y::Vector, x::Matrix, params::Params)
    n = size(x, 1)

    if n != length(y)
        error("Incompatible data.")
    end

    # Get estimates
    xa = !params.IncludeConst ? x : add_const(n, x)[1]
    xx = xa' * xa
    b_hat = xx \ (xa' * y)

    if !params.Stats
        b_hat, nothing
    else
        # Get residuals
        r_hat = y - xa * b_hat

        # Get asyvar.
        avar = @match params.VarType begin
            :Default => (r_hat' * r_hat) .* inv(xx) / n
            :HC      => (xr = xa .* r_hat; xx \ (xr' * xr) / xx)
            _ => error("Invalid VarType parameter.")
        end

        se = sqrt.(diag(avar));
        ci = norm_ci(b_hat, se, params.Alpha)
        stats = Stats(avar, se, ci, r_hat)
        b_hat, stats
    end
end

"""
IV regression.

# Arguments
- `y::Vector`: dependent variable.
- `x::Matrix`: regressors.
- `z::Matrix`: instruments.
- `params::Params`: various parameters.
"""
function iv(y::Vector, x::Matrix, z::Matrix, params::Params)
    n = size(x, 1)

    if n != length(y) || any(size(x) != size(z))
        error("Incompatible data.")
    end

    # Get estimates
    xa, za = !params.IncludeConst ? (x, z) : add_const(n, x, z)
    zx = za' * xa;
    b_hat = zx \ (za' * y);

    if !params.Stats
        b_hat, nothing
    else
        # Get residuals
        r_hat = y - xa * b_hat;

        # Get asyvar.
        avar = @match params.VarType begin
            :Default => (r_hat' * r_hat) * (zx \ (za' * za) / zx') / n
            :HC      => (zr = za .* r_hat; zx \ (zr' * zr) /zx')
            _ => error("Invalid VarType parameter.")
        end

        se = sqrt.(diag(avar));
        ci = norm_ci(b_hat, se, params.Alpha)
        stats = Stats(avar, se, ci, r_hat)
        b_hat, stats
    end
end

"""
NLS regression.
...
# Arguments
- `f`: anonymous function representing m(X, b).
- `y::Vector`: dependent variable.
- `x::Matrix`: regressors.
- `beta0::Vector`: initial guess.
- `params::Params`: various parameters.
"""
function nls(f, y::Vector, x::Matrix, beta0::Vector, params::Params)
    n = size(x, 1)

    if n != length(y)
        error("Incompatible data.")
    end

    # Objective function
    obj = b -> sum((y - f(x, b)) .^ 2);

    # Get estimates
    td = TwiceDifferentiable(obj, beta0; autodiff = :forward)
    o = optimize(td, beta0, Newton(), params.OptimOpts)

    if !Optim.converged(o)
        error("Minimization failed.")
    end

    b_hat = Optim.minimizer(o)

    if !params.Stats
        b_hat, nothing
    else
        # Get residuals
        r_hat = y - f(x, b_hat)

        # Get asyvar. (using the exact gradient)
        v = map(i -> ForwardDiff.gradient(z -> f(x[i, :]', z), b_hat), 1:n)
        md = vcat(v'...)

        me = md .* r_hat; mmd = md' * md
        avar = mmd \ (me' * me) / mmd

        se = sqrt.(diag(avar));
        ci = norm_ci(b_hat, se, params.Alpha)
        stats = Stats(avar, se, ci, r_hat)
        b_hat, stats
    end
end

function norm_ci(b::Vector, se::Vector, α::Float64)
    if α <= 0 || α >= 0.5
        error("Invalid α parameter")
    end

    (quantile(snd, α / 2) .* se + b),
    (quantile(snd, 1 - α / 2) .* se + b)
end

end