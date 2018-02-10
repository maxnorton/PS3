module Simulation

    using ProgressMeter

    run_avg(stepf, niter; seed = nothing) = begin
        if seed != nothing; srand(seed) end
        p = Progress(niter)
        f = i -> (ProgressMeter.next!(p); stepf(i))
        mapreduce(f, (a, b) -> a .+ b, 1:niter) ./ niter
    end

end