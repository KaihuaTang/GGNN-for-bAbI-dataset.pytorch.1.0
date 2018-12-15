##########################
# Reimplementation of Gated Graph Sequence Neural Networks (GGNN)
# Paper Link: https://arxiv.org/abs/1511.05493
# Code Author: Kaihua Tang
# Environment: Python 3.6, Pytorch 1.0
##########################

# Define some useful structure
class Output(object):
    def __init__(self):
        self.accuracy = 0
        self.loss = 0
        self.net = None
    
    def set_accuracy(self, value):
        self.accuracy = value

    def set_loss(self, value):
        self.loss = value

    def set_net(self, net):
        self.net = net

    def get_accuracy(self):
        return self.accuracy

    def get_loss(self):
        return self.loss

    def get_net(self):
        return self.net