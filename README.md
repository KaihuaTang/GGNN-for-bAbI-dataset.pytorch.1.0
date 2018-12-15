# A PyTorch 1.0 Implementation of GGNN on bAbI
This is a Implementation of GGNN based on paper [Gated Graph Sequence Neural Networks](https://arxiv.org/abs/1511.05493). I only focus on the experiments of GGNN on bAbI dataset. There is also another good pytorch implementation of GGNN by JamesChuanggg [Link](https://github.com/JamesChuanggg/ggnn.pytorch). However, it doesn't include task 18, 19 (the GraphLevel Output), and in original paper, it use 10 generated datasets to achieve average performance while JamesChuanggg only use one. This implementation is a complete version of GGNN. Wish it may help you.

## Requirements
- python==3.6
- PyTorch=1.0 or 0.4 (0.3 is not tested)
- Dataset is included in this project, you don't need to download. (following JamesChuanggg [Link](https://github.com/JamesChuanggg/ggnn.pytorch))

## Train
```
Task 4:  python main.py --task_id 4 
Task 15: python main.py --task_id 15
Task 16: python main.py --task_id 16 --hidden_dim 20 --epoch 150 (Task 16 is easy to stuck in local optim, if so please try again)
Task 18: python main.py --task_id 18 --epoch 50
Task 19: python main.py --task_id 19 --epoch 50
```
Definition of arguments
```
--train_size (1~1000): the number of training instances we use, here we use 50 as default as original paper
--task_id (4,15,16,18,19): since original paper only test GGNN on these 5 tasks
--question_id (0~3): for task 4, there are four types of questions, for the rest task, just use defualt value 0
--data_id (0~10): if you set it as 0, it will train 10 different model on 10 different datasets and return average performance, if you set it as 1~10, just run the corresponding dataset
--hidden_dim: hidden size of feature vector
--n_steps: GGNN will be iteratively run n times
--epoch: number of epoch
--resume: if you want to resume from an existing model, please define the name of existing model on config.MODEL_PATH
--name: name of your model, which will be saved to ./model/yourname.pth
```
The rest parameters that we don't often change are defined in config.py


## References
- [Gated Graph Sequence Neural Networks](https://arxiv.org/abs/1511.05493), ICLR 2016
- [yujiali/ggnn](https://github.com/yujiali/ggnn)
- [JamesChuanggg/ggnn.pytorch](https://github.com/JamesChuanggg/ggnn.pytorch)
