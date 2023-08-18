module benchmark3

using ClosedLoopReachability

### Model Dynamics
@taylorize function benchmark3!(dx, x, p, t)
    dx[1] = -x[1] * (0.1 + (x[1] + x[2])^2)
    dx[2] = (x[3] + x[1]) * (0.1 + (x[1] + x[2])^2)
    dx[3] = zero(x[3])
    return dx
end

controller = read_nnet_polar(@modelpath("Benchmarks", "nn_3_relu_sigmoid"));

X₀ = Hyperrectangle(low=[0.8, 0.4],
                    high=[0.9, 0.5])
U₀ = ZeroSet(1);

vars_idx = Dict(:states=>1:2, :controls=>3)
ivp = @ivp(x' = benchmark3!(x), dim: 3, x(0) ∈ X₀ × U₀)
period = 0.1;  # control period

prob = ControlledPlant(ivp, controller, vars_idx, period);

## Safety specification
T = 6  # time horizon
target_states = cartesian_product(
    Hyperrectangle(low=[0.2, -0.3],
                   high=[0.3,   -0.05]),
    Universe(1))
# predicate = X -> isdisjoint(overapproximate(X, Hyperrectangle), unsafe_states);
predicate_R_tend = R -> overapproximate(R, Zonotope, tend(R)) ⊆ target_states
predicate_R_all = R -> R ⊆ target_states
predicate_sol_suff = sol -> predicate_R_all(sol[end]);

# ## Results

# To integrate the ODE, we use the Taylor-model-based algorithm:
alg = TMJets(abstol=1e-6, orderT=6, orderQ=1);

# To propagate sets through the neural network, we use the `DeepZ` algorithm:
alg_nn = DeepZ()

function benchmark(; silent::Bool=false)
    ## We solve the controlled system:
    silent || println("flowpipe construction")
    res_sol = @timed solve(prob, T=T, alg_nn=alg_nn, alg=alg)
    sol = res_sol.value
    silent || print_timed(res_sol)

    # ## Next we check the property for an overapproximated flowpipe:
    silent || println("property checking")
    res_pred = @timed predicate_sol_suff(sol)
    silent || print_timed(res_pred)
    if res_pred.value
        silent || println("The property is satisfied.")
    else
        silent || println("The property may be violated.")
    end
    return sol
end

benchmark(silent=false)  # warm-up
res = @timed benchmark()  # benchmark
sol = res.value
println("total analysis time")
print_timed(res);
io = isdefined(Main, :io) ? Main.io : stdout
print(io, "JuliaReach, 2, -, verified, $(res.time)\n")

# We also compute some simulations:

import DifferentialEquations

println("simulation")
res = @timed simulate(prob, T=T, trajectories=10, include_vertices=false)
sim = res.value
print_timed(res);

# Finally we plot the results:

using Plots

function plot_helper(fig, vars; show_simulation::Bool=true)
    plot!(fig, sol, vars=vars, color=:yellow, lab="")
    target_states_projected = project(target_states, vars)
    plot!(fig, target_states_projected, color=:blue, alpha=:0.2,
          lab="target states", leg=:topleft)
    if show_simulation
        plot_simulation!(fig, sim; vars=vars, color=:black, lab="")
    end
end

vars = (1, 2)
fig = plot(xlab="x₀", ylab="x₁")
plot_helper(fig, vars)

savefig("benchmark3.png")

#-

end  #jl
nothing  #jl

