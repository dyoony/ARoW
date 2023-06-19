# ARoW
This repository contains the code for ICML 2023 paper "Improving adversarial robustness by putting more regularizations on less robust samples" by Dongyoon Yang, Insung Kong and Yongdai Kim.

# Train
We set regularization $2 \lambda$ in our paper to $\lambda$ in our code. i.e. for training ARow with $\lambda=3.5$ in our paper, you should run with $\lambda=7.0$.


`python main.py --loss arow --dataset cifar10 --swa --model resnet18 --lamb 7 --ls 0.2`

Note that we set perturb_loss=ce is used for CIFAR-100 due to training stability.

`python main.py --loss arow --dataset cifar100 --swa --model resnet18 --lamb 7 --ls 0.2 --perturb_loss ce`

# Evaluation

The trained models can be evaluated by running eval.py which contains the standard accuracy and robust accuracies against PGD and AutoAttack.

`python eval.py --datadir {data_dir} --model_dir {model_dir} --swa --model resnet18 --attack_method autoattack`

# Citation


```
@inproceedings{
    dongyoon2023improving,
    title={Improving adversarial robustness by putting more regularizations on less robust samples},
    author={Dongyoon Yang, Insung Kong and Yongdai Kim},
    booktitle={International Conference on Machine Learning},
    year={2023},
    url={https://arxiv.org/abs/2206.03353}
}
```
