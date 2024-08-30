# DQAS4JSSP
This is the repository for the paper [Differentiable Quantum Architecture Search for Job Shop Scheduling Problem(JSSP-DQAS)](https://arxiv.org/pdf/2401.01158).

Authors: Yize Sun1, Jiarui Liu, Yunpu Ma, Volker Tresp

We implement the differentiable quantum architecture search [1] for job shop scheduling problem showing the potential for industrial application. We redefine the operation pool and extend DQAS to a framework JSSP-DQAS by evaluating circuits to generate circuits for JSSP automatically. The experiments conclude that JSSP-DQAS can automatically find noise-resilient circuit architectures that perform much better than manually designed circuits. It helps to improve the efficiency of solving JSSP.

Please use the following BibTex for citation:
```
@inproceedings{sun2024differentiable,
  title={Differentiable Quantum Architecture Search For Job Shop Scheduling Problem},
  author={Sun, Yize and Liu, Jiarui and Ma, Yunpu and Tresp, Volker},
  booktitle={ICASSP 2024-2024 IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP)},
  pages={236--240},
  year={2024},
  organization={IEEE}
}

```

[1]: Zhang, Shi-Xin, et al. "Differentiable quantum architecture search." Quantum Science and Technology 7.4 (2022): 045023.
