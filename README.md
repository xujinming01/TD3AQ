# Mixed-Integer Optimal Control via Reinforcement Learning: A Case Study on Hybrid Vehicle Energy Management

PyTorch implementation of Twin Delayed Deep Deterministic Actor-Q (TD3AQ) algorithm for mixed-integer optimal control problems.

## Abstract

Many optimal control problems require the simultaneous output of continuous and discrete control variables.
Such problems are usually formulated as mixed-integer optimal control (MIOC) problems,
which are challenging to solve due to the complexity of the solution space. 
Numerical methods such as branch-and-bound are computationally expensive and unsuitable for real-time control. 
This brief proposes a novel continuous-discrete reinforcement learning (CDRL) algorithm, twin delayed deep deterministic actor-Q (TD3AQ), for MIOC problems.
TD3AQ combines the advantages of both actor-critic and Q-learning methods, and can handle the continuous and discrete action spaces simultaneously.
The proposed algorithm is evaluated on a plug-in hybrid electric vehicle (PHEV) energy management problem, where real-time control of the continuous variable, engine torque, and discrete variables, gear shift and clutch engagement/disengagement is essential to maximize fuel economy while satisfying driving constraints. 
Simulation results on different drive cycles show that TD3AQ achieves near-optimal control compared to dynamic programming (DP) and outperforms baseline reinforcement learning algorithms.

## License

This project is licensed under the [MIT License](LICENSE). See
the [LICENSE](LICENSE) file for more details.

## Citation

If you find this work useful for your research, please consider citing the paper:

```BibTeX citation entry
@article{xu2023mixed,
  title={Mixed-Integer Optimal Control via Reinforcement Learning: A Case Study on Hybrid Vehicle Energy Management},
  author={Xu, Jinming and Lin, Yuan},
  journal={arXiv preprint arXiv:2305.01461},
  year={2023}
}
```
