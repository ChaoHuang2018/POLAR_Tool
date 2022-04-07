import os
import sys

CROWN_DIR = "/home/jiameng/packages/CROWN_FLOWSTAR/alpha-beta-CROWN/complete_verifier/"
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
    arguments.Config.add_argument('--network_name', type=str, default="nn_4_relu",
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
    verified_status, init_global_lb, saved_bounds, saved_slopes = incomplete_verifier(model_ori, x,
                y, data_ub=data_ub, data_lb=data_lb, eps=perturb_eps)
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

def main(steps = 5):
    model_lb = load_model()
    model_ub = load_model(eval_ctrl_ub=True)
    x0_min = 0.25
    x0_max = 0.27
    x1_min = 0.08
    x1_max = 0.1
    x2_min = 0.25
    x2_max = 0.27
    Xs = np.zeros((steps+1, 6))
    Xs[0] = [x0_min, x0_max, x1_min, x1_max, x2_min, x2_max]

    for step in range(steps):
        us_min = ctrl_input_bound(model_lb, [x0_min, x1_min, x2_min], [x0_max, x1_max, x2_max])
        us_max = -ctrl_input_bound(model_ub, [x0_min, x1_min, x2_min], [x0_max, x1_max, x2_max])
        u_min = (us_min[0] - model_lb.output_offset) * model_lb.output_scale
        u_max = (us_max[0] - model_ub.output_offset) * model_ub.output_scale
        print("******CTRL BOUND*********", u_min, u_max)
        command = ['./flowstar_1step', '6']
        command += [str(i) for i in [x0_min, x0_max, x1_min, x1_max, x2_min, x2_max, u_min, u_max]]
        command += [str(step)]
        command += [arguments.Config["general"]["network_name"]]
        try:
            flowstar_res = subprocess.check_output(" ".join(command), shell=True, text=True)
        except subprocess.CalledProcessError as e:
            print("Flowstar exploded at step", step)
            Xs = Xs[:step+1]
            break

        lines = flowstar_res.split('\n')
        print(lines[0])
        print(lines[1])
        print(lines[2])
        x0_range = [float(i) for i in lines[0].split(' ')]
        x1_range = [float(i) for i in lines[1].split(' ')]
        x2_range = [float(i) for i in lines[2].split(' ')]
        x0_min = x0_range[0]
        x0_max = x0_range[1]
        x1_min = x1_range[0]
        x1_max = x1_range[1]
        x2_min = x2_range[0]
        x2_max = x2_range[1]
        Xs[step+1] = [x0_min, x0_max, x1_min, x1_max, x2_min, x2_max]

    fig, ax = plt.subplots()
    for bound in Xs:
        print(bound)
        x0_min, x0_max, x1_min, x1_max, x2_min, x2_max = bound
        ax.plot([x0_min]*2, [x1_min, x1_max], 'b')
        ax.plot([x0_max]*2, [x1_min, x1_max], 'b')
        ax.plot([x0_min, x0_max], [x1_min]*2, 'b')
        ax.plot([x0_min, x0_max], [x1_max]*2, 'b')

    # ax.set_xlim([-0.5, 1])
    # ax.set_ylim([-1.5, 1])
    plt.show()



if __name__ == "__main__":
    config_args()
    main(10)        # Flowstar error at step 27 because of too large input range

