module benchmark6

using ClosedLoopReachability

### Model Dynamics
@taylorize function benchmark6!(dx, x, p, t)
    dx[1] = x[2]
    dx[2] = -x[1] + 0.1*sin(x[3])
    dx[3] = x[4]
    dx[4] = x[5]
    dx[5] = zero(x[5])
    return dx
end

controller = read_nnet_polar(@modelpath("Benchmarks", "nn_6_relu_tanh"));

X₀ = Hyperrectangle(low=[-0.77, -0.45, 0.51, -0.3],
                    high=[-0.75, -0.43, 0.53, -0.28])
U₀ = ZeroSet(1);

vars_idx = Dict(:states=>1:4, :controls=>5)
ivp = @ivp(x' = benchmark6!(x), dim: 5, x(0) ∈ X₀ × U₀)
period = 0.5;  # control period

prob = ControlledPlant(ivp, controller, vars_idx, period);

## Safety specification
T = 5.0  # time horizon
target_states = cartesian_product(
    Hyperrectangle(low=[-0.1, -0.9, -1000, -1000],
                   high=[0.2, -0.6, 1000, 1000]),
    Universe(1))
# predicate = X -> isdisjoint(overapproximate(X, Hyperrectangle), target_states);
# predicate_R_tend = R -> overapproximate(R, Zonotope, tend(R)) ⊆ target_states
predicate_R_all = R -> R ⊆ target_states
predicate_sol_suff = sol -> predicate_R_all(sol[end]);

goal_states_x1x2 = Hyperrectangle(low=[-0.1, -0.9], high=[0.2, -0.6])
predicate_reachability = sol -> project(sol[end][end], [1, 2]) ⊆ goal_states_x1x2;


# ## Results

# To integrate the ODE, we use the Taylor-model-based algorithm:
alg = TMJets(abstol=1e-10, orderT=8, orderQ=1);

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
    solz = overapproximate(sol, Zonotope)
    res_pred = @timed predicate_reachability(solz)
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

savefig("benchmark6.png")

#-

end  #jl
nothing  #jl

