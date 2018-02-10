include("Common/sim.jl");
include("Common/regress.jl");

using Distributions, DataFrames
using Regress, Simulation

# set parameters
ρ = .9;
β = .15;
α = [0.1; 0.05; 0.01];
errV = [1 ρ; ρ 1];
nostats = Regress.Params(Stats=false, IncludeConst=false);
stats = Regress.Params(Stats=true, IncludeConst=false, VarType=:HC);

function realize(n)
    # generate data
    #n=100;
    w = rand(n,1);
    z = -.5.*(w.<.2) - .1.*(.2.<=w.<.4) + .1*(.4.<=w.<.6) + (w.>=.6);
    errorDist = MvNormal([0; 0], errV);
    e = rand(errorDist,n)';
    u = (1+z).*e[:,1];
    x = 4*z.^2 + e[:, 2];
    y = β*x + u;

    # compute estimators
    #inefficient IV
    b_2sls, stats_2sls = Regress.iv(y[:], x, z, stats);

    #infeasible efficient IV
    gzi = 4*z.^2 ./ ( ( ρ*(1+z) ).^2 )
    b_iev, stats_iev = Regress.iv(y[:], x, gzi, stats);

    #feasible/estimated efficient IV
    d = [1*(z'.==-.5); 1*(z'.==-.1); 1*(z'.==.1); 1*(z'.==1)]';
    pihat = Regress.ols(x[:], d, nostats);
    u_2sls = stats_2sls.Resid;
    usqhat = Regress.ols(u_2sls.^2, d, nostats);
    gzihat = d * (pihat[1] ./ usqhat[1]);
    b_fev, stats_fev = Regress.iv(y[:], x, gzihat[:,:], stats);

    #compute confidence intervals
    ci_2sls_10 = Regress.norm_ci(b_2sls, stats_2sls.SE, α[1])
    ci_2sls_05 = Regress.norm_ci(b_2sls, stats_2sls.SE, α[2])
    ci_2sls_01 = Regress.norm_ci(b_2sls, stats_2sls.SE, α[3])
    ci_iev_10 = Regress.norm_ci(b_iev, stats_iev.SE, α[1])
    ci_iev_05 = Regress.norm_ci(b_iev, stats_iev.SE, α[2])
    ci_iev_01 = Regress.norm_ci(b_iev, stats_iev.SE, α[3])
    ci_fev_10 = Regress.norm_ci(b_fev, stats_fev.SE, α[1])
    ci_fev_05 = Regress.norm_ci(b_fev, stats_fev.SE, α[2])
    ci_fev_01 = Regress.norm_ci(b_fev, stats_fev.SE, α[3])

    b_2sls, b_fev, b_iev, ci_2sls_10[1], ci_2sls_10[2], ci_iev_10[1], ci_iev_10[2], ci_fev_10[1], ci_fev_10[2], ci_2sls_05[1], ci_2sls_05[2], ci_iev_05[1], ci_iev_05[2], ci_fev_05[1], ci_fev_05[2], ci_2sls_01[1], ci_2sls_01[2], ci_iev_01[1], ci_iev_01[2], ci_fev_01[1], ci_fev_01[2]

end

b2, bf, bi, ct10l, ct10u, ci10l, ci10u, cf10l, cf10u, ct5l, ct5u, ci5l, ci5u, cf5l, cf5u, ct1l, ct1u, ci1l, ci1u, cf1l, cf1u = Simulation.run_avg(_ -> realize(100), 10000, seed = 576);
b24, bf4, bi4, ct10l4, ct10u4, ci10l4, ci10u4, cf10l4, cf10u4, ct5l4, ct5u4, ci5l4, ci5u4, cf5l4, cf5u4, ct1l4, ct1u4, ci1l4, ci1u4, cf1l4, cf1u4 = Simulation.run_avg(_ -> realize(400), 10000, seed = 576);
