# A PyTorch 1.0 Implementation of GGNN on bAbI
This is a Implementation of GGNN based on paper [Gated Graph Sequence Neural Networks](https://arxiv.org/abs/1511.05493). I only focus on the experiments of GGNN on bAbI dataset. There is also another good pytorch implementation of GGNN by JamesChuanggg [Link](https://github.com/JamesChuanggg/ggnn.pytorch). However, it doesn't include task 18, 19 (the GraphLevel Output), and in original paper, it use 10 generated datasets to achieve average performance while JamesChuanggg only use one. This implementation is a complete version of GGNN. Wish it may help you.

1. Folder bAbI: Tested on bAbI Task 4,15,16,18,19
