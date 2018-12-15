##########################
# Reimplementation of Gated Graph Sequence Neural Networks (GGNN)
# Paper Link: https://arxiv.org/abs/1511.05493
# Code Author: Kaihua Tang
# Environment: Python 3.6, Pytorch 1.0
##########################

import os
import argparse
import torch
import torch.nn as nn
import torch.optim as optim

import config
import utils
import model
from train import train

parser = argparse.ArgumentParser()
parser.add_argument('--train_size', type=int, default=50, help='number of instance we used to train, 1~1000')
parser.add_argument('--task_id', type=int, default=4, help='bAbI task id')
parser.add_argument('--question_id', type=int, default=0, help='question id for those tasks have several types of questions')
parser.add_argument('--data_id', type=int, default=1, help='generated bAbI data id 1~10')
parser.add_argument('--hidden_dim', type=int, default=8, help='GGNN hidden state size')
parser.add_argument('--n_steps', type=int, default=5, help='propogation steps number of GGNN')
parser.add_argument('--epoch', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--resume', action='store_true', help='resume a pretrained model')
parser.add_argument('--name', type=str, default='model', help='name of model')

opt = parser.parse_args()
if config.VERBAL: print(opt)

def main(opt):
    average_accuracy = 0

    if opt.data_id == 0:
        # rum experiment 10 times using 10 different generated dataset
        for i in range(10):
            opt.data_id = i + 1
            train_output = train(opt)
            test_output =  train(opt, test=True, trained_model=train_output.get_net())
            average_accuracy += test_output.get_accuracy()
        average_accuracy = average_accuracy / 10
    else:
        # run experiment one time at data_id
        train_output = train(opt)
        test_output =  train(opt, test=True, trained_model=train_output.get_net())
        average_accuracy += test_output.get_accuracy()
    
    print('Test accuracy is: ' + str(average_accuracy) + ' for task: ' + str(opt.task_id) + ' at question: ' + str(opt.question_id) + ' using Num: ' + str(opt.train_size) + ' training data.' )

    results = {
        'train_size': opt.train_size,
        'task_id': opt.task_id,
        'question_id': opt.question_id,
        'data_id': opt.data_id,
        'hidden_dim': opt.hidden_dim,
        'n_steps': opt.n_steps,
        'epoch': opt.epoch,
        'weights': train_output.get_net().state_dict(),
    }
    torch.save(results, os.path.join('model', '{}.pth'.format(opt.name)))

    return None

if __name__ == "__main__":
    main(opt)