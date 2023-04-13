import os
import sys

# CROWN_DIR = "/home/jiameng/packages/CROWN_FLOWSTAR/alpha-beta-CROWN/complete_verifier/"
# CROWN_DIR = "/home/zhilu/layR/ITNE_CROWN/alpha-beta-CROWN/complete_verifier/"
CROWN_DIR = "/mnt/d/TCAD/POLAR_Tool/alpha-beta-CROWN/complete_verifier/"
if not os.path.isdir(CROWN_DIR):
    raise Exception("Please set your own CROWN directory.")
sys.path.append(CROWN_DIR)

import subprocess
import socket
import random
import time
import gc

import numpy as np
import pandas as pd

import torch
import arguments

import matplotlib.pyplot as plt

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from bab_verification_general import mip, incomplete_verifier, bab

from crown_model_def import AttitudeController

def config_args():

    h = ["general"]
    arguments.Config.add_argument('--complete_verifier', choices=["bab", "mip", "bab-refine", "skip"], default="bab",
            help='Complete verification verifier. "bab": branch and bound with beta-CROWN; "mip": mixed integer programming (MIP) formulation; "bab-refine": branch and bound with intermediate layer bounds computed by MIP.', hierarchy=h + ["complete_verifier"])
    arguments.Config.add_argument('--no_incomplete', action='store_false', dest='incomplete',
            help='Enable/Disable initial alpha-CROWN incomplete verification (this can save GPU memory when disabled).', hierarchy=h + ["enable_incomplete_verification"])
    arguments.Config.add_argument('--network_name', type=str, default="tanh20x20",
            help='Name of the neural network controller.', hierarchy=h + ["network_name"])
    h = ["debug"]
    arguments.Config.add_argument("--lp_test", type=str, default=None,
        choices=["MIP", "LP", "LP_intermediate_refine", "MIP_intermediate_refine", None],
        help='Debugging option, do not use.', hierarchy=h + ['lp_test'])
    arguments.Config.parse_config()

    arguments.Config["bab"]["decision_thresh"] = np.inf

    arguments.Config["specification"]["norm"] = np.inf
    arguments.Config["data"]["num_outputs"] = 1
    # arguments.Config["bab"]["get_upper_bound"] = True
    # arguments.Config["solver"]['beta-crown']["lr_alpha"] = 10000
    # arguments.Config["solver"]['beta-crown']["lr_beta"] = 10000
    # arguments.Config["solver"]['beta-crown']["lr_decay"] = 0.99
    # arguments.Config["solver"]['beta-crown']["iteration"] = 1000
    # arguments.Config["solver"]['alpha-crown']["iteration"] = 1000

def load_model(eval_ctrl_ub = False):
    nn_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), arguments.Config["general"]["network_name"])
    sign = -1 if eval_ctrl_ub else 1
    model_ori = AttitudeController(nn_path, sign).to(arguments.Config["general"]["device"])
    return model_ori


def ctrl_input_bound(model_ori, input_lb, input_ub, alpha_only=True):
    data_max = None
    data_min = None
    # x_ub = torch.tensor(data_max).flatten().unsqueeze(0).to(arguments.Config["general"]["device"])
    # x_lb = torch.tensor(data_min).flatten().unsqueeze(0).to(arguments.Config["general"]["device"])
    input_lb = torch.tensor(input_lb)
    input_ub = torch.tensor(input_ub)
    x = (input_lb + input_ub) / 2
    perturb_eps = x - input_lb
    x = x.unsqueeze(0).to(dtype=torch.get_default_dtype(), device=arguments.Config["general"]["device"])
    perturb_eps = perturb_eps.unsqueeze(0).to(dtype=torch.get_default_dtype(), device=arguments.Config["general"]["device"])
    start_incomplete = time.time()
    data = x

    if data_max is None:
        data_max = data_ub = data + perturb_eps  # perturb_eps is already normalized
        data_min = data_lb = data - perturb_eps
    else:
        data_ub = torch.min(data + perturb_eps, data_max)
        data_lb = torch.max(data - perturb_eps, data_min)

    # with torch.no_grad():
    #     print(model_ori(data_ub).to('cpu').numpy()-4, model_ori(data_lb).to('cpu').numpy()-4)

    lb_record = []
    init_global_lb = saved_bounds = saved_slopes = None
    y = None
    verified_status, init_global_lb, saved_bounds, saved_slopes = incomplete_verifier(model_ori, x, norm = arguments.Config["specification"]["norm"], 
                y=y, data_ub=data_ub, data_lb=data_lb, eps=perturb_eps)
    lower_bounds, upper_bounds = saved_bounds[1], saved_bounds[2]
    arguments.Config["bab"]["timeout"] -= (time.time()-start_incomplete)

    if alpha_only:
        return lower_bounds[-1][0].cpu().numpy()

    for pidx in range(model_ori.output_size):

        start_inner = time.time()
        if arguments.Config["general"]["enable_incomplete_verification"]:
            # Reuse results from incomplete results, or from refined MIPs.
            # skip the prop that already verified
            rlb, rub = list(lower_bounds), list(upper_bounds)
            rlb[-1] = rlb[-1][0, pidx]
            rub[-1] = rub[-1][0, pidx]
            # print(init_global_lb[0].min().item(), init_global_lb[0].min().item() - arguments.Config["bab"]["decision_thresh"] <= -100.)
            # if init_global_lb[0].min().item() - arguments.Config["bab"]["decision_thresh"] <= -100.:
            #     print(f"Initial alpha-CROWN with worst bound {init_global_lb[0].min().item()}. We will run branch and bound.")
            #     l, u, nodes, glb_record = rlb[-1].item(), rub[-1].item(), 0, []
            # elif init_global_lb[0, pidx] >= arguments.Config["bab"]["decision_thresh"]:
            #     print(f"Initial alpha-CROWN verified for label {pidx} with bound {init_global_lb[0, pidx]}")
            #     l, u, nodes, glb_record = rlb[-1].item(), rub[-1].item(), 0, []
            # else:
            if arguments.Config["bab"]["timeout"] < 0:
                print(f"verification failure (running out of time budget).")
                l, u, nodes, glb_record = rlb[-1].item(), float('inf'), 0, []
            else:
                # feed initialed bounds to save time
                l, u, nodes, glb_record = bab(model_ori, x, pidx, y=y, eps=perturb_eps, data_ub=data_max, data_lb=data_min,
                                lower_bounds=lower_bounds, upper_bounds=upper_bounds, reference_slopes=saved_slopes)
        else:
            assert arguments.Config["general"]["complete_verifier"] == "bab"  # for MIP and BaB-Refine.
            # Main function to run verification
            l, u, nodes, glb_record = bab(model_ori, x, pidx, y=y, eps=perturb_eps,
                                            data_ub=data_max, data_lb=data_min)
        time_cost = time.time() - start_inner
        # print('label {} verification end, final lower bound {}, upper bound {}, time: {}'.format(pidx, l, u, time_cost))
        arguments.Config["bab"]["timeout"] -= time_cost
        lb_record.append([glb_record])

def post_processing(u_min, u_max):
    res = []
    for i in range(8):
        possible_max = True
        for j in range(8):
            if i == j:
                continue
            if u_max[i] < u_min[j]:
                possible_max = False
                break
        if possible_max:
            res.append(i)
    return res

def main(steps = 5, flowstar_stepsize = 0.05):
    model_lb = load_model()
    model_ub = load_model(eval_ctrl_ub=True)
    xs_min = [-0.05, -0.025, 0.0, 0.0, 0.0, 0.0]
    xs_max = [-0.025, 0.0, 0.0, 0.0, 0.0, 0.0]
    Xs_min = [xs_min]
    Xs_max = [xs_max]
    pre_process = np.array([0.2,0.2,0.2,0.1,0.1,0.1])
    flowpipes_this_step = [(xs_min, xs_max)]
    out_of_bound = False

    t0 = time.time()

    for step in range(steps):
        flowpipes_next_step = []
        in_step_id = 0
        print("******* step", step, "total flowpipes:", len(flowpipes_this_step))
        for fp_id in range(len(flowpipes_this_step)):
            xs_min, xs_max = flowpipes_this_step[fp_id]
            xs_min = np.array(xs_min)
            xs_max = np.array(xs_max)
            us_min = ctrl_input_bound(model_lb, xs_min*pre_process, xs_max*pre_process)
            us_max = -ctrl_input_bound(model_ub, xs_min*pre_process, xs_max*pre_process)
            us_min = (us_min - model_lb.output_offset) * model_lb.output_scale
            us_max = (us_max - model_ub.output_offset) * model_ub.output_scale
            print("******CTRL BOUND*********")
            print(us_min)
            print(us_max)
            possible_ctrl_idxs = post_processing(us_min, us_max)
            print("possible_ctrl_idxs:", possible_ctrl_idxs)
            for ctrl_idx in possible_ctrl_idxs:
                command = ['./flowstar_1step_v1', '4']
                command += [str(i) for pair in zip(xs_min, xs_max) for i in pair]
                command += [str(ctrl_idx)]
                command += [str(flowstar_stepsize), str(step), str(in_step_id)]
                try:
                    flowstar_res = subprocess.check_output(" ".join(command), shell=True, text=True)
                except subprocess.CalledProcessError as e:
                    print("Flowstar exploded at step", step)
                    out_of_bound = True

                lines = flowstar_res.split('\n')
                xs_min_new = []
                xs_max_new = []
                print(lines[0])
                for i in range(6):
                    # print(lines[i])
                    x_range = [float(s) for s in lines[i+1].split(' ')]
                    xs_min_new.append(x_range[0])
                    xs_max_new.append(x_range[1])
                    if i<3 and (x_range[0] < -0.32 or x_range[1] > 0.32):
                        print("Out of bound at step", step)
                        out_of_bound = True
                Xs_min.append(xs_min_new)
                Xs_max.append(xs_max_new)
                flowpipes_next_step.append((xs_min_new, xs_max_new))
                in_step_id += 1
        flowpipes_this_step = flowpipes_next_step
        if out_of_bound:
            print("CROWN-FLOWSTAR terminated at step", step)
            break

    runtime = int(time.time() - t0 + 1)     # Ceiling to an int
    print(f"Total time = {runtime} seconds.")

    with open('./outputs/quadrotor_crown_flowstar/quadrotor_crown_result.txt', 'w') as f:
        res = "No" if out_of_bound else "Yes"
        f.write(f"Verification result: {res} ({step})\n")
        f.write(f"Running time: {runtime} seconds\n")

    # fig, ax = plt.subplots()
    # for xs_min, xs_max in zip(Xs_min, Xs_max):
    #     x0_min, x0_max, x1_min, x1_max = xs_min[0], xs_max[0], xs_min[1], xs_max[1]
    #     ax.plot([x0_min]*2, [x1_min, x1_max], 'b')
    #     ax.plot([x0_max]*2, [x1_min, x1_max], 'b')
    #     ax.plot([x0_min, x0_max], [x1_min]*2, 'b')
    #     ax.plot([x0_min, x0_max], [x1_max]*2, 'b')

    # # ax.set_xlim([-0.35, 0.35])
    # # ax.set_ylim([-0.35, 0.35])
    # plt.show()



if __name__ == "__main__":
    config_args()
    main(30)        # Flowstar error at step 27 because of too large input range

