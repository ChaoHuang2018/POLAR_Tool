### Instructions on running Alpha-beta-crown + flow*

1. Requirements: pull alpha-beta-crown from <href>https://github.com/huanzhang12/alpha-beta-CROWN</href> and put the folder in the root directory `POLAR_Tool/`
2. Update the `Makefile` in each of the `./benchmark1-6` directories, by adding an object `flowstar_1step_v1`. Refer to the example `Makefile` in `./acc` directory. Then, make the objects. 
3. Open terminal in the current directory (POLAR_Tool/examples/). Run command `python abcrown_flowstar_verifier.py --config ./benchmark dir/benchmark yaml file`. For instance, 
   * for acc, use command `python abcrown_flowstar_verifier.py --config ./acc/acc.yaml`.
   * for benchmark 1 relu, use command `python abcrown_flowstar_verifier.py --config ./benchmark1/benchmark1.yaml`.
   * for benchmark 1 sigmoid, use command `python abcrown_flowstar_verifier.py --config ./benchmark1/benchmark1_sigmoid.yaml`.
   * for benchmark 1 tanh, use command `python abcrown_flowstar_verifier.py --config ./benchmark1/benchmark1_tanh.yaml`.
   * for other benchmarks, just change the benchmark number in the above 3 commands.
4. The python script will generate two files. One is a `.m` file including all the flowpipes. The other one is a `.eps` file that only plots the input intervals of the neural networks. 
   * For acc, the `.m` file will be in the `POLAR_Tool/examples/outputs/abcrown_flowstar_acc_tanh20x20x20_crown_flowstar` directory
   * For benchmark 1/2/.../6, the `.m` file will be in `POLAR_Tool/examples/outputs/abcrown_flowstar_benchmark$NUM_$NEURALNETWORK_crown_flowstar` where $NUM is the index of the benchmark and $NEURALNETWORK is the name of the neural network file, e.g., nn_1_relu, nn_1_sigmoid_crown.
  
### Run ACC
1. Go to `POLAR_Tool/examples/acc` directory.
2. Run `./run_acc.sh`.
3. The generated flowpipes are plotted in `POLAR_Tool/examples/acc/outputs/acc_tanh20x20x20_x4x5_steps_50_1.m`
4. For matlab simulation, run `run("simulate_with_NN_rl.m")` in Matlab.
