module docking

using ClosedLoopReachability

### Model Dynamics
@taylorize function docking!(dx, x, p, t)
    dx[1] = x[3] * 0.5 / 1000.0
    dx[2] = x[4] * 0.5 / 1000.0
    dx[3] = (2.0 * 0.001027 * x[4] * 0.5 + 3 * 0.001027 * 0.001027 * x[1] * 1000.0 + x[7]/12.0) / 0.5
    dx[4] = (-2.0 * 0.001027 * x[3] * 0.5 + x[9] / 12.0) / 0.5
    dx[5] = ((2.0 * 0.001027 * x[4] * 0.5  + 3 * 0.001027 * 0.001027 * x[1] * 1000.0 +  x[7] / 12.) * x[3] * 0.5  + (-2.0 * 0.001027 * x[3] * 0.5  + x[9] / 12.) * x[4] * 0.5 ) / x[5]
    dx[6] = 2.0 * 0.001027 * (x[1] * 1000.0 * x[3] * 0.5 + x[2] * 1000.0 * x[4] * 0.5) / sqrt(x[1] * 1000.0 * x[1] * 1000.0 + x[2] * 1000.0 * x[2] * 1000.0)
	dx[7] = zero(x[7])
    dx[8] = zero(x[8])
    dx[9] = zero(x[9])
    dx[10] = zero(x[10])
    return dx
end

controller = read_nnet_polar(@modelpath("docking", "docking_tanh64x64_tanh"));

X₀ = Hyperrectangle(low=[24/1000.0, 24/1000.0, -0.13776233054248638/0.5, -0.13776233054248638/0.5, 0.1948253562373095, 0.2697150717707441],
                    high=[26/1000.0, 26/1000.0, -0.13776233054248638/0.5, -0.13776233054248638/0.5, 0.1948253562373095, 0.27552466108497276])
U₀ = ZeroSet(4);

vars_idx = Dict(:states=>1:6, :controls=>7:10)
ivp = @ivp(x' = docking!(x), dim: 10, x(0) ∈ X₀ × U₀)
period = 1;  # control period

prob = ControlledPlant(ivp, controller, vars_idx, period);

## Safety specification
T = 120 # time horizon
#target_states = cartesian_product(
#    Hyperrectangle(low=[24, 24, -0.13776233054248638, -0.13776233054248638, 0.1948253562373095, 0.2697150717707441],
#                   high=[26, 26, -0.13776233054248638, -0.13776233054248638, 0.1948253562373095, 0.27552466108497276]),
#    Universe(4))
# predicate = X -> isdisjoint(overapproximate(X, Hyperrectangle), target_states);
# predicate_R_tend = R -> overapproximate(R, Zonotope, tend(R)) ⊆ target_states
#predicate_R_all = R -> R ⊆ target_states
#predicate_sol_suff = sol -> predicate_R_all(sol[end]);

#goal_states_x5x6 = Hyperrectangle(low=[0.24, 0.15], high=[0.25, 0.2])
#predicate_reachability = sol -> project(sol[end][end], [5, 6]) ⊆ goal_states_x5x6;


# ## Results

# To integrate the ODE, we use the Taylor-model-based algorithm:
alg = TMJets(abstol=1e-10, orderT=4, orderQ=1);

# To propagate sets through the neural network, we use the `DeepZ` algorithm:
alg_nn = DeepZ()

function benchmark(; silent::Bool=false)
    ## We solve the controlled system:
    silent || println("flowpipe construction")
    res_sol = @timed solve(prob, T=T, alg_nn=alg_nn, alg=alg)
    sol = res_sol.value
    silent || print_timed(res_sol)

    # ## Next we check the property for an overapproximated flowpipe:
    #silent || println("property checking")
    #solz = overapproximate(sol, Zonotope)
    #res_pred = @timed predicate_reachability(solz)
    #silent || print_timed(res_pred)
    #if res_pred.value
    #    silent || println("The property is satisfied.")
    #else
    #    silent || println("The property may be violated.")
    #end
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
    #target_states_projected = project(target_states, vars)
    #plot!(fig, target_states_projected, color=:blue, alpha=:0.2,
    #      lab="target states", leg=:topleft)
    if show_simulation
        plot_simulation!(fig, sim; vars=vars, color=:black, lab="")
    end
end

vars = (5, 6)
fig = plot(xlab="v", ylab="vₛ")
plot_helper(fig, vars)

savefig("docking.png")

#-

end  #jl
nothing  #jl

