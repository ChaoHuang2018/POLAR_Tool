# POLAR Official version
POLAR [1] is a reachability analysis framework for neural-network controlled systems (NNCSs) based on polynomial arithmetic. Compared with existing arithmetic approaches that use standard Taylor models, our framework uses a novel approach to iteratively overapproximate the neuron output ranges layer-by-layer with a combination of Bernstein polynomial interpolation for continuous activation functions and Taylor model arithmetic for the other operations. This approach can overcome the main drawback in the standard Taylor model arithmetic, i.e. its inability to handle functions that cannot be well approximated by Taylor polynomials, and significantly improve the accuracy and efficiency of reachable states computation for NNCSs. To further tighten the overapproximation, our method keeps the Taylor model remainders symbolic under the linear mappings when estimating the output range of a neural network. 

Experiment results across a suite of benchmarks show that POLAR significantly outperforms the state-of-the-art techniques on both efficiency and tightness of reachable set estimation.

## Installation

#### System Requirements
Ubuntu 18.04, MATLAB 2016a or later


#### Installation of Flow*
POLAR relies on the Taylor model arithmetic library provided by Flow*. Please install Flow* with the same directory of POLAR. You can either use the following command or follow the manual of Flow* for installation.

- Install dependencies through apt-get install
```
sudo apt-get install m4 libgmp3-dev libmpfr-dev libmpfr-doc libgsl-dev gsl-bin bison flex gnuplot-x11 libglpk-dev gcc-8 g++-8 libopenmpi-dev
```
- Download Flow*
```
git clone https://github.com/chenxin415/flowstar.git
```

- Compile Flow*
```
cd flowstar
make
```

#### Compile POLAR
```
cd POLAR
make
```

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

```

## Contributors
[Chao Huang](https://chaohuang2018.github.io/main/), [Jiameng Fan](https://www.jiamengf.com), [Xin Chen](https://udayton.edu/directory/artssciences/computerscience/chen-xin.php), [Wenchao Li](http://sites.bu.edu/depend/people/), [Qi Zhu](http://users.eecs.northwestern.edu/~qzhu/)

## References
[1] C.Huang, J.Fan, W.Li, X.Chen, and Q.Zhu.
[POLAR: A Polynomial Arithmetic Framework for Verifying Neural-Network Controlled Systems]()
