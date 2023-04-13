import os
import socket
import random
import time
import gc

import numpy as np
import pandas as pd

import torch
import arguments

from auto_LiRPA import BoundedModule, BoundedTensor
from auto_LiRPA.perturbations import PerturbationLpNorm
from bab_verification_general import mip, incomplete_verifier, bab


from model_defs import AttitudeController


def config_args():
    # Add arguments specific for this front-end.
    h = ["general"]
    arguments.Config.add_argument("--mode", type=str, default="verified-acc", choices=["verified-acc", "runnerup", "clean-acc", "specify-target"],
            help='Verify against all labels ("verified-acc" mode), or just the runnerup labels ("runnerup" mode), or using a specified label in dataset ("speicify-target" mode, only used for oval20).', hierarchy=h + ["mode"])
    arguments.Config.add_argument('--complete_verifier', choices=["bab", "mip", "bab-refine", "skip"], default="bab",
            help='Complete verification verifier. "bab": branch and bound with beta-CROWN; "mip": mixed integer programming (MIP) formulation; "bab-refine": branch and bound with intermediate layer bounds computed by MIP.', hierarchy=h + ["complete_verifier"])
    arguments.Config.add_argument('--no_incomplete', action='store_false', dest='incomplete',
            help='Enable/Disable initial alpha-CROWN incomplete verification (this can save GPU memory when disabled).', hierarchy=h + ["enable_incomplete_verification"])
    arguments.Config.add_argument("--crown", action='store_true', help='Compute CROWN verified accuracy before verification (not used).', hierarchy=h + ["get_crown_verified_acc"])

    h = ["model"]
    arguments.Config.add_argument("--model", type=str, default="please_specify_model_name", help='Name of model. Model must be defined in the load_verification_dataset() function in utils.py.', hierarchy=h + ["name"])

    h = ["init"]
    arguments.Config.add_argument("--min", nargs='+', type=float, default=[-0.45, -0.55, 0.65, -0.75, 0.85, -0.65], help='Min initial input vector.', hierarchy=h + ["min"])
    arguments.Config.add_argument("--max", nargs='+', type=float, default=[-0.44, -0.54, 0.66, -0.74, 0.86, -0.64], help='Max initial input vector.', hierarchy=h + ["max"])

    h = ["debug"]
    arguments.Config.add_argument("--lp_test", type=str, default=None, choices=["MIP", "LP", "LP_intermediate_refine", "MIP_intermediate_refine", None], help='Debugging option, do not use.', hierarchy=h + ['lp_test'])

    arguments.Config.parse_config()

def build_model_file(idx, step, x_min, x_max):
    path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "flowstar-2.1.0/benchmark3.model")

    lines_ori = []
    with open(path, 'r') as f:
        lines_ori = f.readlines()
        f.close()
    for i_line in range(len(lines_ori)):
        if "gnuplot interval" in lines_ori[i_line]:
            lines_ori[i_line] = lines_ori[i_line].split("gnuplot interval ")[0] + "gnuplot interval x{}, x{}\n".format(idx, idx + 1)
        elif "output" in lines_ori[i_line]:
            lines_ori[i_line] = lines_ori[i_line].split("output ")[0] + "output benchmark3_x{}_x{}_step{}\n".format(idx, idx + 1, step)
        elif "x0 in" in lines_ori[i_line]:
            lines_ori[i_line] = lines_ori[i_line].split("x0 in ")[0] + "x0 in [{}, {}]\n".format(x_min[1], x_max[1])
        elif "x1 in" in lines_ori[i_line]:
            lines_ori[i_line] = lines_ori[i_line].split("x1 in ")[0] + "x1 in [{}, {}]\n".format(x_min[2], x_max[2])
        elif "u in" in lines_ori[i_line]:
            lines_ori[i_line] = lines_ori[i_line].split("u in ")[0] + "u in [{}, {}]\n".format(x_min[0], x_max[0])
        else:
            continue
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),  \
        "flowstar-2.1.0/benchmark3_x{}_x{}_step{}.model".format(idx, idx + 1, step))
    with open(model_path, 'w') as f:
        for line in lines_ori:
            f.write(line)
        f.close()
    return model_path

def run_model_file(idx, step):
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),  \
        "flowstar-2.1.0/benchmark3_x{}_x{}_step{}.model".format(idx, idx + 1, step))
    flowstar_path = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "flowstar-2.1.0/")
    os.system("{}/flowstar < {}".format(flowstar_path, model_path))
    plt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),  \
        "outputs/benchmark3_x{}_x{}_step{}.plt".format(idx, idx + 1, step))
    os.system("gnuplot '{}'".format(plt_path))
    return plt_path

def read_plt_file(idx, step):
    plt_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),  \
        "outputs/benchmark3_x{}_x{}_step{}.plt".format(idx, idx + 1, step))
    min_x = float('inf')
    max_x = -float('inf')
    min_y = float('inf')
    max_y = -float('inf')
    with open(plt_path, 'r') as f:

        line = f.readline()
        while "plot" not in line:
            line = f.readline()
            continue
        line = f.readline()
        while line != "\n":
            line = line.split('\n')[0]
            min_x = min_x if min_x < float(line.split(' ')[0]) else float(line.split(' ')[0])
            max_x = max_x if max_x > float(line.split(' ')[0]) else float(line.split(' ')[0])

            min_y = min_y if min_y < float(line.split(' ')[1]) else float(line.split(' ')[1])
            max_y = max_y if max_y > float(line.split(' ')[1]) else float(line.split(' ')[1])
            line = f.readline()
        f.close()
        return min_x, max_x, min_y, max_y

def flowstar(step, x_min, x_max):
    print(">>>>>>>>>>>>>Start to run flowstar!!>>>>>>>>>>>")
    model_paths = []
    for idx in range(1):
        model_path = build_model_file(idx, step, x_min, x_max)
        model_paths.append(model_path)
    for idx in range(1):
        plt_path = run_model_file(idx, step)
    for idx in range(1):
        x_min[1 + idx], x_max[1 + idx], x_min[2 + idx], x_max[2 + idx] = read_plt_file(idx, step)
    return x_min[1:], x_max[1:]

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


    save_path = 'Verified_ret_[{}]_start={}_end={}_iter={}_b={}_timeout={}_branching={}-{}-{}_lra-init={}_lra={}_lrb={}_PGD={}.npy'. \
        format(arguments.Config['model']['name'], arguments.Config["data"]["start"],  arguments.Config["data"]["end"], arguments.Config["solver"]["beta-crown"]["iteration"], arguments.Config["solver"]["beta-crown"]["batch_size"],
               arguments.Config["bab"]["timeout"], arguments.Config["bab"]["branching"]["method"], arguments.Config["bab"]["branching"]["reduceop"],
               arguments.Config["bab"]["branching"]["candidates"], arguments.Config["solver"]["alpha-crown"]["lr_alpha"], arguments.Config["solver"]["beta-crown"]["lr_alpha"], arguments.Config["solver"]["beta-crown"]["lr_beta"], arguments.Config["attack"]["pgd_order"])
    print(f'saving results to {save_path}')

    step_ids = range(arguments.Config["data"]["start"],  arguments.Config["data"]["end"])


    # Load model
    model_ori = AttitudeController(os.path.join(os.path.dirname(__file__), "models/POLAR/benchmark3/nn_3_relu")).to(arguments.Config["general"]["device"])

    # The initial state min, max as a single range are loaded into a list
    data_min = torch.tensor(arguments.Config["init"]["min"]).unsqueeze(0).to(arguments.Config["general"]["device"])
    data_max = torch.tensor(arguments.Config["init"]["max"]).unsqueeze(0).to(arguments.Config["general"]["device"])
    x = (data_max + data_min)/2.
    perturb_eps = torch.max(x - data_min, 1)[0]

    # Initialize lists of current and next state ranges and control output ranges
    X_min, X_max, U_min, U_max, X_nxt, X_min_nxt, X_max_nxt = [data_min], [data_max], [], [], [], [], []

    # Test run the initial control output given a medium state

    with torch.no_grad():
        for i in range(3):
            u_pred = model_ori(x)
            print("Given medium input {}".format(x))
            print("Attitude controller's output {}".format(u_pred))

            for idx in range(model_ori.output_size):
                model_ori.filter(idx, arguments.Config["general"]["device"])
                #model_ori = model_ori.to(arguments.Config["general"]["device"])
                u_pred = model_ori(x)
                print("Attitude controller filtered {}th output {}".format(idx, u_pred))
                model_ori.filter(device = arguments.Config["general"]["device"])
                #model_ori.to(arguments.Config["general"]["device"])

    u_pred = model_ori(x)
    print("Given medium input {}".format(x))
    print("Attitude controller's output {}".format(u_pred))
    import torch.optim as optim

    data_min = data_min.flatten().detach().cpu().numpy().tolist()
    data_max = data_max.flatten().detach().cpu().numpy().tolist()

    # Run step by step
    for step in step_ids:
        u_min, u_max = crown_verify(step, model_ori, data_min, data_max)
        print(">>>>>>>>>>>>>>>>>>> Step {}: control output lower bound {} upper bound {}".format(step, u_min, u_max))

        x_min = u_min + data_min
        x_max = u_max + data_max

        data_min, data_max = flowstar(step, x_min, x_max)



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
    x_ub = torch.tensor(data_max).flatten().unsqueeze(0).to(arguments.Config["general"]["device"])
    x_lb = torch.tensor(data_min).flatten().unsqueeze(0).to(arguments.Config["general"]["device"])
    x_med = (x_ub + x_lb)/2.


    model_ori.scale(\
        w = (x_med - x_lb).flatten().cpu().numpy().tolist(), \
            b = x_med.flatten().cpu().numpy().tolist(), \
                device = arguments.Config["general"]["device"])

    x = 0. * x_med
    data_ub = x_ub / x_ub
    data_lb = - x_lb / x_lb
    data = x
    perturb_eps = 1.


    with torch.no_grad():
        u_pred = model_ori(x)
        y_pred = np.argmax(u_pred.flatten().detach().cpu().numpy())
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
            y = None, data_ub=data_ub, data_lb=data_lb, eps=perturb_eps)
        ############
        print("verified_status: ", verified_status)
        print("init_global_lb: ", init_global_lb)
        print("saved bounds: ", saved_bounds)
        # lower_bounds, upper_bounds = saved_bounds[1], saved_bounds[2]
        arguments.Config["bab"]["timeout"] -= (time.time()-start_incomplete)
        ret.append([step, -1, 0, time.time()-start_incomplete, -1, np.inf, np.inf])
        #return lower_bounds[0].flatten().detach().cpu().numpy().tolist(), upper_bounds[0].flatten().detach().cpu().numpy().tolist()

        save_path = 'Verified_ret_[{}]_start={}_end={}_iter={}_b={}_timeout={}_branching={}-{}-{}_lra-init={}_lra={}_lrb={}_PGD={}.npy'. \
            format(arguments.Config['model']['name'], arguments.Config["data"]["start"],  arguments.Config["data"]["end"], arguments.Config["solver"]["beta-crown"]["iteration"], arguments.Config["solver"]["beta-crown"]["batch_size"],
            arguments.Config["bab"]["timeout"], arguments.Config["bab"]["branching"]["method"], arguments.Config["bab"]["branching"]["reduceop"],
            arguments.Config["bab"]["branching"]["candidates"], arguments.Config["solver"]["alpha-crown"]["lr_alpha"], arguments.Config["solver"]["beta-crown"]["lr_alpha"], arguments.Config["solver"]["beta-crown"]["lr_beta"], arguments.Config["attack"]["pgd_order"])
        print(f'saving results to {save_path}')


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
                            l, u, nodes, glb_record = bab(model_ori, x, pidx, arguments.Config["specification"]["norm"], y=y, eps=perturb_eps, data_ub=data_ub, data_lb=data_lb,
                                        lower_bounds=lower_bounds, upper_bounds=upper_bounds, reference_slopes=saved_slopes, attack_images=targeted_attack_images)
                else:
                    print(">>>>>>>>>>>>>>> Skipped incomplete verification, and refined MIPs. Run complete_verifier: {}".format(arguments.Config["general"]["complete_verifier"]))
                    assert arguments.Config["general"]["complete_verifier"] == "bab"  # for MIP and BaB-Refine.
                    # Main function to run verification

                    ################# Run complete verification directly
                    l, u, nodes, glb_record = bab(model_ori, x, pidx, arguments.Config["specification"]["norm"], y=y, eps=perturb_eps,
                                                data_ub=data_ub, data_lb=data_lb, attack_images=targeted_attack_images)
                    #################

                #temp = l
                #l = - u #/(model_ori.output_size - 1.)
                #u = - temp#/(model_ori.output_size - 1.)
                assert l <= u, "lower bound {} is no less than upper bound {}".format(l, u)
                l = (l - 1) * 2
                u = (u - 1) * 2
                u_lb.append(l)
                u_ub.append(u)

                time_cost = time.time() - start_inner
                print('Step {} test against {} verification end, final lower bound {}, upper bound {}, time: {}'.format(step, pidx, l, u, time_cost))

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
    return u_lb, u_ub


if __name__ == "__main__":
    config_args()
    main()
