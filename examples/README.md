# Instructions on running Alpha-beta-crown + flow*

### Requirements

* Download <a href=https://github.com/huanzhang12/alpha-beta-CROWN> alpha-beta-crown </a> (recommend the <a href=https://github.com/huanzhang12/alpha-beta-CROWN/tree/7745bd1e6e10ea482c021e5d65b0a56daf0be4a6> November 21 release</a>) and put the folder in the root directory `POLAR_Tool/`.


### Usage
* Under `POLAR_Tool/examples/`, run `python abcrown_flowstar_verifier.py --config ./$benchmark_dir/$benchmark.yaml`

### Example 
* For [*acc*](./acc/), run `python abcrown_flowstar_verifier.py --config ./acc/acc.yaml`.
* For [*benchmark 1 relu*](./benchmark1/), run `python abcrown_flowstar_verifier.py --config ./benchmark1/benchmark1.yaml`.
* For [*benchmark 1 sigmoid*](./benchmark1), run `python abcrown_flowstar_verifier.py --config ./benchmark1/benchmark1_sigmoid.yaml`.
* For [*benchmark 1 tanh*](./benchmark1), run `python abcrown_flowstar_verifier.py --config ./benchmark1/benchmark1_tanh.yaml`.
* For other benchmarks, please find the corresponging the yaml file and specify the file location as in the example commands.

### Output
* The python script will generate two files: a `.m` file including all the flowpipes and an `.eps` file that only plots the input intervals of the neural networks. 
* For [*acc*](./acc/), the `.m` file will be in the `POLAR_Tool/examples/outputs/abcrown_flowstar_acc_tanh20x20x20_crown_flowstar` directory
* For benchmark 1/2/.../6, the `.m` file will be in `POLAR_Tool/examples/outputs/abcrown_flowstar_benchmark$NUM_$NEURALNETWORK_crown_flowstar` where $NUM is the index of the benchmark and $NEURALNETWORK is the name of the neural network file, e.g., nn_1_relu, nn_1_sigmoid_crown.


# Instruction on using the onnx converter
### Requirement

* Install python onnx library `pip install onnx`

### Usage
* Copy the `onnx_converger` file to the same directory of the C++ source file.
* In C++ source file, use command `system("python onnx_converter $ONNX_FILE_PATH");` to generatge POLAR input neural network file. The generated file has the same name as the onnx file except for the `.onnx` suffix.
