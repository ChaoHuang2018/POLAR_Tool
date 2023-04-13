import os
import sys

CROWN_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "alpha-beta-CROWN/complete_verifier/")
print(CROWN_DIR)
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

from model_defs import AttitudeController, POLARController
 
import traceback


def config_args():
    # Add arguments specific for this front-end.
    h = ["general"]
    arguments.Config.add_argument("--network", type=str, help='Path the network model, e.g., models/POLAR/ACC/acc_tanh20x20x20_ for ACC', hierarchy=h + ["network"])
    arguments.Config.add_argument("--mode", type=str, default="verified-acc", choices=["verified-acc", "runnerup", "clean-acc", "specify-target"],
            help='Verify against all labels ("verified-acc" mode), or just the runnerup labels ("runnerup" mode), or using a specified label in dataset ("speicify-target" mode, only used for oval20).', hierarchy=h + ["mode"])
    arguments.Config.add_argument('--complete_verifier', choices=["bab", "mip", "bab-refine", "skip"], default="bab",
            help='Complete verification verifier. "bab": branch and bound with beta-CROWN; "mip": mixed integer programming (MIP) formulation; "bab-refine": branch and bound with intermediate layer bounds computed by MIP.', hierarchy=h + ["complete_verifier"])
    arguments.Config.add_argument('--no_incomplete', action='store_false', dest='incomplete',
            help='Enable/Disable initial alpha-CROWN incomplete verification (this can save GPU memory when disabled).', hierarchy=h + ["enable_incomplete_verification"])
    arguments.Config.add_argument('--no_complete', action='store_false', dest='complete',
            help='Enable/Disable initial alpha-CROWN incomplete verification (this can save GPU memory when disabled).', hierarchy=h + ["enable_complete_verification"])
    arguments.Config.add_argument("--crown", action='store_true', help='Compute CROWN verified accuracy before verification (not used).', hierarchy=h + ["get_crown_verified_acc"])
    
    h = ["flowstar"]
    arguments.Config.add_argument("--flowstar", type=str, help='flowstar exe file name', default = 'flowstar_1step_v1', hierarchy=h + ["flowstar"])
    arguments.Config.add_argument("--order", type=int, default = 6, help='flowstar order', hierarchy=h + ["order"])
    

    h = ["model"]
    arguments.Config.add_argument("--name", type=str, default="please_specify_model_name", help='Name of model. Model must be defined in the load_verification_dataset() function in utils.py.', hierarchy=h + ["name"])
    #arguments.Config.add_argument("--path", type=str, default="please_specify_model_path", help='Path of model.', hierarchy=h + ["model_path"])
    arguments.Config.add_argument("--input_ids", nargs='+', type=float, default=[], help='NN input dims among the state variables', hierarchy=h + ["input_ids"])
    arguments.Config.add_argument("--plt_ids", nargs='+', type=float, default=[], help='The state variable ids to be plotted by flowstar', hierarchy=h + ["plt_ids"])
    arguments.Config.add_argument("--plt_name", type=str, default=None, help='Name of plt file name. ', hierarchy=h + ["plt_name"])
    
    h = ["init"]
    arguments.Config.add_argument("--min", nargs='+', type=float, default=[], help='Min initial input vector.', hierarchy=h + ["min"])
    arguments.Config.add_argument("--max", nargs='+', type=float, default=[], help='Max initial input vector.', hierarchy=h + ["max"])

    h = ["debug"]
    arguments.Config.add_argument("--lp_test", type=str, default=None, choices=["MIP", "LP", "LP_intermediate_refine", "MIP_intermediate_refine", None], help='Debugging option, do not use.', hierarchy=h + ['lp_test'])

    arguments.Config.parse_config()


def flowstar(exp_name, flowstar_name, plt_name, step, ux_min, ux_max):
    
    u_min = ux_min['u']
    u_max = ux_max['u']
    x_min = ux_min['x']
    x_max = ux_max['x']
 
    assert len(x_min) == len(x_max) and len(u_min) == len(u_max)
    print("******CTRL BOUND*********", u_min, u_max)
    explode = False
    command = ['6', plt_name, str(step)]
    for j in range(len(x_min)):
        command += [str(i) for i in [x_min[j], x_max[j]]] 
    for j in range(len(u_min)):
        command += [str(i) for i in [u_min[j], u_max[j]]]
    command = "::".join(command) + "::"
    print("Running command: {}".format(command))
    try:
        flowstar_res = subprocess.check_output(" ".join([f'./{exp_name}/{flowstar_name}', command]), shell=True, text=True)
        lines = flowstar_res.split('\n')
        #print(lines)
        for i in range(len(x_min)):
            x_range = [float(j) for j in lines[i].split(' ')]
            x_min[i] = x_range[0]
            x_max[i] = x_range[1]
            print("state variable {}th bounds: {} and {}".format(i, x_min[i], x_max[i]))
        return explode, x_min, x_max
    except subprocess.CalledProcessError as e:
        print("Flowstar Error at step", step)
        return True, None, None

    

def main():
    print(f'Experiments at {time.ctime()} on {socket.gethostname()}')
    torch.manual_seed(arguments.Config["general"]["seed"])
    random.seed(arguments.Config["general"]["seed"])
    np.random.seed(arguments.Config["general"]["seed"])
    if arguments.Config["general"]["device"] != 'cpu':
        torch.cuda.manual_seed_all(arguments.Config["general"]["seed"])
        # Always disable TF32 (precision is too low for verification).
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False

    if arguments.Config["general"]["deterministic"]:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True)

    if arguments.Config["general"]["double_fp"]:
        torch.set_default_dtype(torch.float64)

    if arguments.Config["specification"]["norm"] != np.inf and arguments.Config["attack"]["pgd_order"] != "skip":
        print('Only Linf-norm attack is supported, the pgd_order will be changed to skip')
        arguments.Config["attack"]["pgd_order"] = "skip"

    step_ids = range(arguments.Config["data"]["start"],  arguments.Config["data"]["end"])
    exp_name = arguments.Config['model']['name']
    flowstar_name = arguments.Config['flowstar']['flowstar']
    plt_name = arguments.Config['model']['plt_name']
    if plt_name is None:
        plt_name = exp_name
    network_path = os.path.join(os.path.dirname(__file__), arguments.Config['model']['path'])
    #model_ori = POLARController(network_path).to(arguments.Config["general"]["device"])
    
    model_ori_pos = AttitudeController(network_path, 1).to(arguments.Config["general"]["device"])
    model_ori_neg = AttitudeController(network_path, -1).to(arguments.Config["general"]["device"])
    input_ids = arguments.Config['model']['input_ids']
    # The initial state min, max as a single range are loaded into a list
    data_min = arguments.Config["init"]["min"]
    data_max = arguments.Config["init"]["max"]
    print(data_min, data_max)
    #plt_ids = arguments.Config['model']['plt_ids']
    #if len(plt_ids) == 0:
    #    plt_ids = [(i, j) for (i, j) in zip(range(len(data_min)), range(1, len(data_min)))]
    #for plt_idx in plt_ids:
    #    assert len(plt_idx) == 2
    #    print("To be plotted: {}".format(plt_idx))
    
    # Run step by step
    ux_min = {'x': [i for i in data_min], 'u': [None for _ in range(arguments.Config["data"]["num_classes"])]}
    ux_max = {'x': [i for i in data_max], 'u': [None for _ in range(arguments.Config["data"]["num_classes"])]}

    #Xs = np.zeros((len(list(step_ids))+1, len(plt_ids), 4))
    #Xs[0] = np.asarray([[data_min[plt_idx[0]], data_max[plt_idx[0]], data_min[plt_idx[1]], data_max[plt_idx[1]]] for plt_idx in plt_ids])
    
    #plt_path = os.path.join("./outputs", "_".join([plt_name, "crown_flowstar"]), "_".join([plt_name, f"{len(step_ids)}steps.m"]))
    #print(plt_path)
    #os.system(f"rm {plt_path}")
    #os.system(f"touch {plt_path}")
    #start_time = time.time()
    #for step in step_ids:
    for step in range(1):
        if len(input_ids) > 0:
            input_min = [data_min[i] for i in input_ids]
            input_max = [data_max[i] for i in input_ids]
        else:
            input_min = data_min
            input_max = data_max
        print(input_min, input_max)
        os.system("rm ./abcrown_flowstar_tmp")
    
        if arguments.Config["general"]["enable_complete_verification"]:
            u_min = crown_verify(step, model_ori_pos, input_min, input_max)
            u_max = crown_verify(step, model_ori_neg, input_min, input_max)
        else:
            u_min = ctrl_input_bound(model_ori_pos, input_min, input_max, alpha_only=True)
            u_max = ctrl_input_bound(model_ori_neg, input_min, input_max, alpha_only=True)
        

        print(">>>>>>>>>>>>>>>>>>> Step {}: control output lower bound {} upper bound {}".format(step, u_min, u_max))
        
        with open("./abcrown_flowstar_tmp", "w") as f:
            for i in range(len(u_min)):
                f.write(f"{u_min[i]}::{u_max[i]}::")
            f.close()
         

        """
        ux_min['u'] = u_min
        ux_max['u'] = u_max
        ux_min['x'] = data_min[:]
        ux_max['x'] = data_max[:]
         
        explode, data_min, data_max = flowstar(exp_name, flowstar_name, plt_name, step, ux_min, ux_max)

       
        
        if explode:
            Xs = Xs[:step + 1, :, :]
            break
        else:
            print(plt_path)
            with open(plt_path, "a") as f_o:
                with open(os.path.join("./outputs", "_".join([plt_name, "crown_flowstar"]), f"step{step}.m"), "r") as f_i:
                    for line in f_i:
                        f_o.write(line)
                    f_i.close()
                f_o.close()
            Xs[step+1] = np.asarray([[data_min[plt_idx[0]], data_max[plt_idx[0]], data_min[plt_idx[1]], data_max[plt_idx[1]]] for plt_idx in plt_ids])
        """
    """        
    end_time = time.time()
    with open(os.path.join(exp_name, f"outputs/{plt_name}.txt"), 'w') as f:
        f.write(f"Experiment name: {exp_name}\n")
        f.write(f"Complete: {Xs.shape[0]}/{len(list(step_ids))}\n")
        f.write(f"Total time: {end_time - start_time}\n")
        f.close()

    for idx in range(len(plt_ids)):
        fig, ax = plt.subplots()
        for bound in Xs[:, idx]:
            x0_min, x0_max, x1_min, x1_max = bound
            ax.plot([x0_min]*2, [x1_min, x1_max], 'b')
            ax.plot([x0_max]*2, [x1_min, x1_max], 'b')
            ax.plot([x0_min, x0_max], [x1_min]*2, 'b')
            ax.plot([x0_min, x0_max], [x1_max]*2, 'b')

        #ax.set_xlim([-0.4, 1.2])
        #ax.set_ylim([-0.6, 0.8])
        #plt.show()
        plt.savefig(os.path.join(exp_name, f"outputs/{plt_name}_x{plt_ids[idx][0]}x{plt_ids[idx][1]}.eps"))
    """

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
    verified_status, init_global_lb, saved_bounds, saved_slopes = incomplete_verifier(model_ori, x, arguments.Config["specification"]["norm"], \
        y, data_ub=data_ub, data_lb=data_lb, eps=perturb_eps)
    lower_bounds, upper_bounds = saved_bounds[1], saved_bounds[2]
    arguments.Config["bab"]["timeout"] -= (time.time()-start_incomplete)
    print("lowerbound: {}".format(lower_bounds[-1][0]))
    print("upperbound: {}".format(upper_bounds[-1][0]))
    if alpha_only:
        return model_ori.unsign_offset_scale(lower_bounds[-1][0].cpu().numpy())

    


def crown_verify(step, model_ori, data_min, data_max):
    ret, lb_record, attack_success = [], [], []
    mip_unsafe, mip_safe, mip_unknown = [], [], []

    cnt = 0
    orig_timeout = arguments.Config["bab"]["timeout"]

    init_global_lb = saved_bounds = saved_slopes = None
     # Extract each range from the range list
    """
    for idx in range(len(X_max)):
        data_max = X_max[idx]
        data_min = X_min[idx]
        x = (data_max + data_min)/2.
        y = None
    """
    data_ub = torch.tensor(data_max).float().flatten().unsqueeze(0).to(arguments.Config["general"]["device"])
    data_lb = torch.tensor(data_min).float().flatten().unsqueeze(0).to(arguments.Config["general"]["device"])
    data = (data_ub + data_lb)/2.
    x = data.unsqueeze(0).to(dtype=torch.get_default_dtype(), device=arguments.Config["general"]["device"])
    perturb_eps = (data_ub - data).unsqueeze(0).to(dtype=torch.get_default_dtype(), device=arguments.Config["general"]["device"])
    
    #model_ori.scale(\
    #    w = (x_med - x_lb).flatten().cpu().numpy().tolist(), \
    #        b = x_med.flatten().cpu().numpy().tolist(), \
    #            device = arguments.Config["general"]["device"])

    #x = 0. * x_med
    #data_ub = x_ub / x_ub
    #data_lb = - x_lb / x_lb
    #data = x
    #perturb_eps = 1.
    with torch.no_grad():
        u = model_ori(x)
        print("Median input {} ==> output {}".format(x, u))
    
    
    y = None

    if arguments.Config["general"]["enable_incomplete_verification"] or arguments.Config["general"]["complete_verifier"] == "bab-refine":
        print(">>>>>>>>>>>>>>>Incomplete verification is enabled by default. The intermediate lower and upper bounds will be reused in bab and mip.")
        start_incomplete = time.time()
        #model_ori.filter(1, device = arguments.Config["general"]["device"])



        init_global_lb = saved_bounds = saved_slopes = None
        ############ incomplete_verification execution
        verified_status, init_global_lb, saved_bounds, saved_slopes = incomplete_verifier(
            model_ori = model_ori, data = data_ub,
            norm = arguments.Config["specification"]["norm"], \
            y = None, data_ub=data_ub, data_lb=data_lb, eps=None)
        ############
        print("verified_status: ", verified_status)
        print("init_global_lb: ", init_global_lb)
        print("saved bounds: ", saved_bounds)
        # lower_bounds, upper_bounds = saved_bounds[1], saved_bounds[2]
        arguments.Config["bab"]["timeout"] -= (time.time()-start_incomplete)
        ret.append([step, -1, 0, time.time()-start_incomplete, -1, np.inf, np.inf])
        #return lower_bounds[0].flatten().detach().cpu().numpy().tolist(), upper_bounds[0].flatten().detach().cpu().numpy().tolist()
 
        if arguments.Config["general"]["mode"] == "verified-acc":
            if arguments.Config["general"]["enable_incomplete_verification"] and init_global_lb is not None:
                # We have initial incomplete bounds.
                labels_to_verify = init_global_lb.argsort().squeeze().tolist()
            else:
                labels_to_verify = list(range(arguments.Config["data"]["num_classes"]))

        #elif arguments.Config["general"]["mode"] == "runnerup":
        #    labels_to_verify = [u_pred.argsort(descending=True)[1]]
        else:
            raise ValueError("unknown verification mode")

        pidx_all_verified = True
        print("labels_to_verify: ", labels_to_verify)


        u_lb = []
        u_ub = []

    for pidx in range(model_ori.output_size): #labels_to_verify:
        if isinstance(pidx, torch.Tensor):
            pidx = pidx.item()
        # Filter out all non-pidx output channels so that they output 0 constantly
        print('##### [Step {}] Tested against {} ######'.format(step, pidx))

        model_ori.filter(pidx, arguments.Config["general"]["device"])
        y = None

        init_global_lb = saved_bounds = saved_slopes = None
        if arguments.Config["general"]["enable_incomplete_verification"] or arguments.Config["general"]["complete_verifier"] == "bab-refine":
            print(">>>>>>>>>>>>>>>Incomplete verification is enabled by default. The intermediate lower and upper bounds will be reused in bab and mip.")
            start_incomplete = time.time()

            # Redo incomplete_verification since the neural network structure is changed
            ############ incomplete_verification execution
            verified_status, init_global_lb, saved_bounds, saved_slopes = incomplete_verifier(
                model_ori = model_ori, data = data,
                norm = arguments.Config["specification"]["norm"], \
                y = y, data_ub=data_ub, data_lb=data_lb, eps=None)

            ############
            print("verified_status: ", verified_status)
            print("init_global_lb: ", init_global_lb)
            print("saved bounds: ", saved_bounds)
            if saved_bounds is not None:
                lower_bounds, upper_bounds = saved_bounds[1], saved_bounds[2]

            #print(lower_bounds)
            #print(upper_bounds)

            torch.cuda.empty_cache()
            gc.collect()

            start_inner = time.time()
            targeted_attack_images = None

            try:
                if (saved_bounds is not None) and arguments.Config["general"]["enable_incomplete_verification"]:
                    # Reuse results from incomplete results, or from refined MIPs.
                    # skip the prop that already verified
                    print(">>>>>>>>>>>>>>> Reuse results from incomplete results, or from refined MIPs. Skip the prop that already verified")
                    rlb, rub = list(lower_bounds), list(upper_bounds)
                    rlb[-1] = rlb[-1][0, pidx]
                    rub[-1] = rub[-1][0, pidx]
                    if init_global_lb[0].min().item() - arguments.Config["bab"]["decision_thresh"] <= -100.:
                        print(f"Initial alpha-CROWN with worst bound {init_global_lb[0].min().item()}. We will run branch and bound.")
                        l, u, nodes, glb_record = rlb[-1].item(), float('inf'), 0, []
                    elif init_global_lb[0, pidx] >= arguments.Config["bab"]["decision_thresh"]:
                        print(f"Initial alpha-CROWN verified for label {pidx} with bound {init_global_lb[0, pidx]}")
                        l, u, nodes, glb_record = rlb[-1].item(), float('inf'), 0, []
                    else:
                        if arguments.Config["bab"]["timeout"] < 0:
                            print(f"Step {step} test agains {pidx} verification failure (running out of time budget).")
                            l, u, nodes, glb_record = rlb[-1].item(), float('inf'), 0, []
                        else:
                            # feed initialed bounds to save time
                            l_, u_, nodes, glb_record = bab(model_ori, x, pidx, arguments.Config["specification"]["norm"], y=y, eps=perturb_eps, data_ub=data_ub, data_lb=data_lb,
                                        lower_bounds=lower_bounds, upper_bounds=upper_bounds, reference_slopes=saved_slopes, attack_images=targeted_attack_images)
                            l = - u_
                            u = - l_
                else:
                    print(">>>>>>>>>>>>>>> Skipped incomplete verification, and refined MIPs. Run complete_verifier: {}".format(arguments.Config["general"]["complete_verifier"]))
                    assert arguments.Config["general"]["complete_verifier"] == "bab"  # for MIP and BaB-Refine.
                    # Main function to run verification

                    ################# Run complete verification directly
                    l_, u_, nodes, glb_record = bab(model_ori, x, pidx, arguments.Config["specification"]["norm"], y=y, eps=perturb_eps,
                                                data_ub=data_ub, data_lb=data_lb, attack_images=targeted_attack_images)
                    #################
                    l = - u_
                    u = - l_

                #temp = l
                #l = - u #/(model_ori.output_size - 1.)
                #u = - temp#/(model_ori.output_size - 1.)
                assert l <= u, "lower bound {} is no less than upper bound {}".format(l, u)
                #l = (l - 1) * 4
                #u = (u - 1) * 4
                u_lb.append(l)
                u_ub.append(u)

                time_cost = time.time() - start_inner
                print('Step {} test against {} verification end, final lower bound {}, upper bound {}, time: {}'.format(step, pidx, u_lb[-1], u_ub[-1], time_cost))

                ret.append([step, pidx, l, nodes, time_cost, u, np.inf])
                arguments.Config["bab"]["timeout"] -= time_cost

                if u < arguments.Config["bab"]["decision_thresh"]:
                    verified_status = "unsafe-bab"
                    print("!!!!!!!!!!!!! Unsafe-bab status. Break the loop")
                    continue
                    break
                #elif l < arguments.Config["bab"]["decision_thresh"]:
                #    if not arguments.Config["bab"]["attack"]["enabled"]:
                #        pidx_all_verified = False
                #        # break to run next sample save time if any label is not verified.
                #        break

            except KeyboardInterrupt:
                print('Step {} input range {} test against {} time {}:', step, idx, pidx, time.time()-start_inner, "\n",)
                print(ret)
                pidx_all_verified = False
                break

    if not pidx_all_verified:
        print(f'Result: Step {step} verification failure (with branch and bound).')
    else:
        print(f'Result: Step {step} verification success (with branch and bound)!')
    # Make sure ALL tensors used in this loop are deleted here.
    del init_global_lb, saved_bounds, saved_slopes
     
    return model_ori.unsign_offset_scale(np.asarray(u_lb))
    
    return u_lb, u_ub


if __name__ == "__main__":
    config_args()
    main()       

