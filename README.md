# POLAR Official version
POLAR [1] is a reachability analysis framework for neural-network controlled systems (NNCSs) based on polynomial arithmetic. Compared with existing arithmetic approaches that use standard Taylor models, our framework uses a novel approach to iteratively overapproximate the neuron output ranges layer-by-layer with a combination of Bernstein polynomial approximation for continuous activation functions and Taylor model arithmetic for the other operations. This approach can overcome the main drawback in the standard Taylor model arithmetic, i.e. its inability to handle functions that cannot be well approximated by Taylor polynomials, and significantly improve the accuracy and efficiency of reachable states computation for NNCSs. To further tighten the overapproximation, our method keeps the Taylor model remainders symbolic under the linear mappings when estimating the output range of a neural network. 

Experiment results across a suite of benchmarks show that POLAR significantly outperforms the state-of-the-art techniques on both efficiency and tightness of reachable set estimation.

## Tight Bounds for Bernstein Polynomial Approximation of ReLU
The following special treatment for Bernstein polynomial (BP) approximation on ReLU units was also implemented in the submission to the AINNCS category in ARCH-COMP 2022 in addition to the techniques described in [1].
 
Thanks to the characteristic of the ReLU activation function, we can directly obtain tight bounds for the BP approximation as shown in the following [figure](/tests/bp_relu.png) where the dashes lines represent upper and lower bounds of the BP approximation. 
<p align="center">
  <img src="/tests/bp_relu.png" />
</p>
 

The Taylor model (TM) overapproximation $p(x)+I$ of $\text{ReLU}(x)$ is given by $p(x) = BP(x) - \frac{BP(0)}{2}$ and $I = [-\frac{BP(0)}{2}, \frac{BP(0)}{2}])$ where $BP(0)$ is the Bernstein polynomial $BP(x)$ evaluated at $x=0$. It can be shown that for $x \in [a, b]$ with $a < 0 < b$, the bounds of the interval remainder $I$ are tight for any order-k BP approximation with $k \geq 1$.


## Installation

#### System Requirements
Ubuntu 18.04, MATLAB 2016a or later


#### Dependencies
POLAR relies on the Taylor model arithmetic library provided by Flow*. Please install Flow* with the same directory of POLAR. You can either use the following command or follow the manual of Flow* for installation.

- Install dependencies through apt-get install
```
sudo apt-get install m4 libgmp3-dev libmpfr-dev libmpfr-doc libgsl-dev gsl-bin bison flex gnuplot-x11 libglpk-dev gcc-8 g++-8 libopenmpi-dev libpthread-stubs0-dev
```
- Download Flow*
```
git clone https://github.com/chenxin415/flowstar.git
```

- Compile Flow*
```
cd flowstar/flowstar-toolbox
make
```

#### Compile POLAR
```
cd POLAR
make
```

## Example Usage
*Example 1.* Consider the following nolinear control system ([benchmark 1](/examples/benchmark1)):

![formula](https://render.githubusercontent.com/render/math?math=\dot{x}_0=x_1,\quad\dot{x}_1=ux_1^2-x_0)

where ![formula](https://render.githubusercontent.com/render/math?math=u) is computed from a NN controller ![formula](https://render.githubusercontent.com/render/math?math=\kappa) that has two hidden layers, twenty neurons in each layer, and ReLU and tanh as activation functions. Given a control stepsize ![formula](https://render.githubusercontent.com/render/math?math=\delta_c=0.2), we hope to verify whether the system will reach ![formula](https://render.githubusercontent.com/render/math?math=[0,0.2]\times[0.05,0.3]) from the initial set ![formula](https://render.githubusercontent.com/render/math?math=[0.8,0.9]\times[0.5,0.6]) over the time interval ![formula](https://render.githubusercontent.com/render/math?math=[0,7]).

Partial code of the benchmark 1 dynamics ([*reachnn_benchmark_1.cpp* file](/examples/benchmark1/reachnn_benchmark_1.cpp)) are shown as follows:

```C++
// Declaration of the state and input variables.
unsigned int numVars = 4;
Variables vars;

int x0_id = vars.declareVar("x0");
int x1_id = vars.declareVar("x1");
int t_id = vars.declareVar("t");
int u_id = vars.declareVar("u");

// Define the continuous dynamics.
ODE<Real> dynamics({"x1",
                    "u*x1^2-x0",
                    "1",
                    "0"}, vars);
...
// Define initial state set.
double w = stod(argv[1]);
Interval init_x0(0.85 - w, 0.85 + w), init_x1(0.55 - w, 0.55 + w), init_u(0); // w=0.05
...
// Define the neural network controller.
string net_name = argv[6];  // relu_tanh
string nn_name = "nn_1_"+net_name;
NeuralNetwork nn(nn_name);
...
// Order of Bernstein Polynomial
unsigned int bernstein_order = stoi(argv[3]); // 4
// Order of Taylor Model
unsigned int order = stoi(argv[4]); // 6
// Define the time interval
int steps = stoi(argv[2]);  // 35
...
// Define target set
vector<Constraint> targetSet;
Constraint c1("x0 - 0.2", vars);		// x0 <= 0.2
Constraint c2("-x0", vars);			// x0 >= 0
Constraint c3("x1 - 0.3", vars);		// x1 <= 0.3
Constraint c4("-x1 + 0.05", vars);		// x1 >= 0.05
...
```

The NN controller is specified in [*nn_1_relu_tanh* file](/examples/benchmark1/nn_1_relu_tanh) as follows
```C++
2 // number of inputs
1 // number of outputs
2 // number of hidden layers
20 // number of nodes in the first hidden layer
20 // number of nodes in the second hidden layer
ReLU // Activation function of the first layer
ReLU // Activation function of the second layer
tanh // Activation function of the output layer
// Weights of the first hidden layer
-0.0073867239989340305
...
0.014101211912930012
// Bias of the first hidden layer
-0.07480818033218384
...
0.29650038480758667
...
0 // Offset of the neural network
4 // Scala of the neural network
```
Then we can verify the NNCS with the following command under [*example1/benchmark1*](/examples/benchmark1/):
```bash
make reachnn_benchmark_1 && ./reachnn_benchmark_1 0.05 35 4 6 1 relu_tanh
```
where 0.05 is the width of the initial set, 35 is the total steps that need to be verified, 4 is the order of Bernstein Polynomial, 6 is the order of Taylor Model, 1 specifies option to use symbolic remainder and relu_tanh specifies the NN controller with ReLU and tanh activation functions which points to [*nn_1_relu_tanh*](/examples/benchmark1/nn_1_relu_tanh) file. 

A bash script `run.sh` is also available for each benchmark to run POLAR.

The computed flowpipes are shown in [the figure](/examples/benchmark1/outputs/reachnn_benchmark_1_relu_tanh_1.eps).

![alt text](/examples/benchmark1/outputs/reachnn_benchmark_1_relu_tanh_1.png)

The output file of POLAR shows the verification results:

```C++
Verification result: Yes(35)  // Verification result at the 35th step
Running Time: 11.000000 seconds // Total computation time in seconds
```

Here, "Yes" means that the target set contains the overapproximation of the reachable set. In other words, every trajectory of the system is guaranteed to reach the target set at time T. If the result returns "No", it means that the target set and the overapproximation of the reachable set are mutually exclusive. Every trajectory of the system will fall outside of the target set. "Unknown" means that the target set intersects with the overapproximation of the reachable set. It is unknown whether every system trajectory will fall inside the target set.
<!-- 
## Examples - POLAR results

### Example #1 
./run.sh

### Checking Result
All results will be stored in ./outputs/

For SYSTEM, the results include a txt file that show the verification result and the POLAR running time, and a M file (with .m extension) that is used to plot the reachable sets computed by POLAR. One can check the result of SYSTEM by following commands.



```

vim SYSTEM_0.txt # verification result

```


```

SYSTEM_0.m # plotted reachable sets. Run the command in MATLAB.

``` -->

## Contributors
[Chao Huang](https://chaohuang2018.github.io/main/), [Jiameng Fan](https://www.jiamengf.com), [Xin Chen](https://udayton.edu/directory/artssciences/computerscience/chen-xin.php), [Zhilu Wang](http://zhulab.ece.northwestern.edu/people/zhilu.html), [Yixuan Wang](https://wangyixu14.github.io/), [Weichao Zhou](https://sites.google.com/view/zwc662/), [Wenchao Li](http://sites.bu.edu/depend/people/), [Qi Zhu](http://users.eecs.northwestern.edu/~qzhu/)

## References
[1] Yixuan Wang, Weichao Zhou, Jiameng Fan, Zhilu Wang, Jiajun Li, Xin Chen, Chao Huang, Wenchao Li and Qi Zhu.
[POLAR-Express: Efficient and Precise Formal Reachability Analysis of Neural-Network Controlled Systems](https://arxiv.org/abs/2304.01218)

[2] Chao Huang, Jiameng Fan, Xin Chen, Wenchao Li and Qi Zhu.
[POLAR: A Polynomial Arithmetic Framework for Verifying Neural-Network Controlled Systems](https://dl.acm.org/doi/abs/10.1007/978-3-031-19992-9_27), Proceedings of the 20th International Symposium on Automated Technology for Verification and Analysis (ATVA 2022).
