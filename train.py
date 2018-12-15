##########################
# Reimplementation of Gated Graph Sequence Neural Networks (GGNN)
# Paper Link: https://arxiv.org/abs/1511.05493
# Code Author: Kaihua Tang
# Environment: Python 3.6, Pytorch 1.0
##########################

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import data
import model
import config
import utils
import structure

# train
def train(opt, test=False, trained_model=None):
    output = structure.Output()
    mode = 'test' if test else 'train'
    babisplit, babiloader = data.get_loader(mode=mode, task_id=opt.task_id, data_id=opt.data_id, train_size=opt.train_size, val=False, ques_id=opt.question_id)
    opt.n_edge_type = babisplit.n_edge_type
    opt.n_node_type = babisplit.n_node_type
    opt.n_label_type = babisplit.n_label_type
    
    net = trained_model if test else model.GGNN(opt).cuda()
    net = net.double()
    if opt.resume:
        logs = torch.load(config.MODEL_PATH)
        net.load_state_dict(logs['weights'])

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(net.parameters(), lr=config.LEARNING_RATE)

    if test:
        net.eval()
    else:
        net.train()
        output.set_net(net)

    if config.VERBAL and not test:
        print('------------------------ Dataset: '+str(opt.data_id)+' -------------------------------')

    num_epoch = 1 if test else opt.epoch
    for i in range(num_epoch):
        total_loss = []
        total_accuracy = []
        for adj_matrix, annotation, target in babiloader:
            padding = torch.zeros(len(annotation), opt.n_node_type, opt.hidden_dim - config.ANNOTATION_DIM[str(opt.task_id)]).double()
            x = torch.cat((annotation, padding), 2)

            x = Variable(x.cuda())
            m = Variable(adj_matrix.cuda())
            a = Variable(annotation.cuda())
            t = Variable(target.cuda()).long()

            pred = net(x, a, m)
            if opt.task_id == 19:
                # consider each step as a prediction
                pred = pred.view(-1, pred.shape[-1])
                t = t.view(-1)
            loss = criterion(pred, t)
            if not test:
                net.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss.append(loss.item())

            accuracy = (pred.max(1)[1] == t).float().mean()
            total_accuracy.append(accuracy.item())

        if config.VERBAL:
            print(mode + ' Epoch: ' + str(i) + ' Loss: {:.3f} '.format(sum(total_loss) / len(total_loss)) + ' Accuracy: {:.3f} '.format(sum(total_accuracy) / len(total_accuracy)))
        output.set_loss(sum(total_loss) / len(total_loss))
        output.set_accuracy(sum(total_accuracy) / len(total_accuracy))

    return output