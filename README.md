# FedFQ
The algorithm example of "FedFQ: Federated Learning with Fine-Grained Quantization". Paper：https://arxiv.org/abs/2408.08977
# Overview 
We propose a communicationefficient FL algorithm with a fine-grained adaptive quantization strategy (FedFQ). FedFQ addresses the trade-off between achieving high communication compression ratios and
maintaining superior convergence performance by introducing parameter-level quantization.  Specifically, we have designed a Constraint-Guided Simulated Annealing(CGSA) algorithm to determine specific quantization schemes.
# Code
The CGSA algorithm proposed by us and the benchmark algorithm for federated learning are both in [src/algorithm]. Here is an example of running the code:
   ```python
python3 main.py \
    --exp_name CGSA_CIFAR10_CNN_IID --seed 42 --device cuda \
    --dataset CIFAR10 \
    --split_type iid --test_size -1 \
    --model_name SimpleCNN --crop 24 --randhf 0.1 --randjit 0.1 --imnorm --hidden_size 64 \
    --algorithm CGSA --eval_fraction 1 --eval_type local --eval_every 1 --eval_metrics acc1 acc5 \
    --K 100 --R 1000 --E 5 --C 0.1 --B 50 --beta 0 \
    --optimizer SGD --lr 0.15 --lr_decay 0.99 --lr_decay_step 1 --criterion CrossEntropyLoss
```
For detailed information on the training parameters for federated learning, please refer to main.py. The compression rate, initial temperature, cooling rate, and number of iterations for the CGSA algorithm can be set in [src/algorithm/CGSA.py].

