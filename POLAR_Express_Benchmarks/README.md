# NNCS_Reach_Benchmarks
We obtained Table III in [POLAR-Express paper](https://arxiv.org/pdf/2304.01218.pdf) by running the examples in this repo with different tools.  

## Summary
It remains an open problem to verify the reachability of neural network controlled systems (NNCS). We collect common benchmarks for recent SOTA tools in this repo. The benchmarks range from 2-dimensional simple systems with shallow neural network controllers to 12-dimensional systems with "deep" NNs. The NNCS verification tools include [CORA](https://tumcps.github.io/CORA/), [Juliareach](https://github.com/JuliaReach/ClosedLoopReachability.jl), [RINO](https://github.com/sputot/RINO), [Verisig 2.0](https://github.com/Verisig/verisig) [NNV](https://github.com/transafeailab/nnv), [POLAR-Express](https://github.com/ChaoHuang2018/POLAR_Tool). 

## Note
This repo is not runnable, rather we provide the system dynamics of each benchmark with its NN controller, in equivalent formats required by different tools. To run/test these benchmarks with a specific tool, you have to navigate to its repo, follow its instructions, copy, and paste the files in this repo to the appropriate locations, compile (if any), and run it.  

The benchmark examples for POLAR-Express can be found at 
```
../examples/
```
folder which are runnable to preduce the results in the paper. 

$\alpha$, $\beta$-crown + Flow* in [POLAR-Exress paper](https://arxiv.org/pdf/2304.01218.pdf) is also located the above folder. 

If you find the benchmarks useful for your research, you can cite
```
@article{wan2023polar,
  title={POLAR-Express: Efficient and Precise Formal Reachability Analysis of Neural-Network Controlled Systems},
  author={Wan, Yixuan and Zhou, Weichao and Fan, Jiameng and Wang, Zhilu and Li, Jiajun and Chen, Xin and Huang, Chao and Li, Wenchao and Zhu, Qi},
  journal={arXiv preprint arXiv:2304.01218},
  year={2023}
},

@inproceedings{huang2022polar,
  title={POLAR: A polynomial arithmetic framework for verifying neural-network controlled systems},
  author={Huang, Chao and Fan, Jiameng and Chen, Xin and Li, Wenchao and Zhu, Qi},
  booktitle={International Symposium on Automated Technology for Verification and Analysis},
  pages={414--430},
  year={2022},
  organization={Springer}
}
```
