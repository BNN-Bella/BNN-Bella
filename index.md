## Welcome to GitHub Pages of "Bayesian Low Rank Learning (Bella)"

Reproduce our results: [GitHub](https://github.com/BNN-Bella/BNN-Bella/)

Check out our paper: [Bayesian Low-Rank LeArning (Bella)](https://arxiv.org/pdf/2407.20891)

Cite our research: 
```
@article{doan2024bayesian,
  title={Bayesian low-rank learning (bella): A practical approach to bayesian neural networks},
  author={Doan, Bao Gia and Shamsi, Afshar and Guo, Xiao-Yu and Mohammadi, Arash and Alinejad-Rokny, Hamid and Sejdinovic, Dino and Ranasinghe, Damith C and Abbasnejad, Ehsan},
  journal={arXiv preprint arXiv:2407.20891},
  year={2024}
}
```
---

#### ABSTRACT

Bayesian learning has demonstrated merits in robustness and resilience to unseen and out-of-distribution inputs, yet its adoption in large-scale tasks remains limited. In particular, deep ensemble methods (Seligmann et al. 2024; Lakshminarayanan, Pritzel, and Blundell 2017) have proven to be highly effective, but they suffer from a high computational cost. This paper introduces an alternative framework that mitigates the computational burden of ensemble Bayesian deep learning. The approach is inspired by the recent success of low-rank adapters and is named Bayesian Low-Rank LeArning (Bella). We show that (1) Bella achieves a dramatic reduction in the number of trainable parameters required to approximate a Bayesian posterior; and (2) it not only maintains, but in some instances surpasses the performance—in accuracy and outof-distribution generalisation—of conventional Bayesian and non-Bayesian methods. An extensive empirical evaluation on large-scale tasks (ImageNet, CAMELYON17, DomainNet, VQA) with CLIP and LLaVA demonstrates the effectiveness and versatility of Bella in building scalable and Bayesian deep models for real-world applications.

---

#### Key Contributions
- **Introducing Bella (Bayesian Low-Rank Learning)**: A novel framework designed to reduce the computational burden of ensemble Bayesian deep learning.
- **Efficiency in Parameter Reduction**: Bella achieves a dramatic reduction in the number of trainable parameters needed to approximate a Bayesian posterior.
- **Performance Improvement**: Bella maintains or surpasses the performance of conventional Bayesian learning methods and non-Bayesian baselines in terms of accuracy and out-of-distribution generalization.
- **Empirical Evaluation**: Extensive evaluation on large-scale tasks such as ImageNet, CAMELYON17, DomainNet, VQA with CLIP, and LLaVA, demonstrating the scalability and effectiveness of Bella in real-world applications.

---
<img src="https://raw.githubusercontent.com/BNN-Bella/BNN-Bella/main/Figs/fig1-camelyon-ver8.png" alt="Bella Framework" width="500"/>

Train and Test Error (\%) landscapes of CAMELYON17. Training landscape demonstrates the learned 3 particle approximation of the modes of the posterior from a pre-trained model $\boldsymbol\theta$---modification of parameters in the constrained region ($\Delta\boldsymbol\theta$) leads to approaching posterior modes in inference. 
Here, we observe a key benefit of our Bella approximation compared to a point estimate---while a single parameter particle (e.g. $\theta +\Delta\boldsymbol\theta_1$) do not generalize well, Bayesian prediction with Equation (1), effectively an average over multiple parameter settings, leads to better performance.







