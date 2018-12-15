##########################
# Reimplementation of Gated Graph Sequence Neural Networks (GGNN)
# Paper Link: https://arxiv.org/abs/1511.05493
# Code Author: Kaihua Tang
# Environment: Python 3.6, Pytorch 1.0
##########################

import config
import torch
import torch.nn as nn

class GGNN(nn.Module):
    """
    Gated Graph Sequence Neural Networks (GGNN)
    Mode: SelectNode
    Implementation based on https://arxiv.org/abs/1511.05493
    """
    def __init__(self, opt):
        super(GGNN, self).__init__()
        self.task_id = opt.task_id
        self.hidden_dim = opt.hidden_dim
        self.annotation_dim = config.ANNOTATION_DIM[str(opt.task_id)]
        self.n_node = opt.n_node_type
        self.n_edge = opt.n_edge_type
        self.n_output = opt.n_label_type
        self.n_steps = opt.n_steps

        self.fc_in = nn.Linear(self.hidden_dim, self.hidden_dim * self.n_edge)
        self.fc_out = nn.Linear(self.hidden_dim, self.hidden_dim * self.n_edge)

        self.gated_update = GatedPropagation(self.hidden_dim, self.n_node, self.n_edge)

        if self.task_id == 18 or self.task_id == 19:
            self.graph_aggregate =  GraphFeature(self.hidden_dim, self.n_node, self.n_edge, self.annotation_dim)
            self.fc_output = nn.Linear(self.hidden_dim, self.n_output)
        else:
            self.fc1 = nn.Linear(self.hidden_dim+self.annotation_dim, self.hidden_dim)
            self.fc2 = nn.Linear(self.hidden_dim, 1)
            self.tanh = nn.Tanh()

    def forward(self, x, a, m):
        '''
        init state x: [batch_size, num_node, hidden_size] , pad zero from annoatation
        annoatation x: [batch_size, num_node, 1] 
        adj matrix m: [batch_size, num_node, num_node * n_edge_types * 2]
        output out: [batch_size, n_label_types], for task 4, 15, 16, n_label_types == num_nodes
        '''
        x, a, m = x.double(), a.double(), m.double()
        all_x = [] # used for task 19, to track 
        for i in range(self.n_steps):
            in_states = self.fc_in(x)
            out_states = self.fc_out(x)
            in_states = in_states.view(-1,self.n_node,self.hidden_dim,self.n_edge).transpose(2,3).transpose(1,2).contiguous()
            in_states = in_states.view(-1, self.n_node*self.n_edge, self.hidden_dim)
            out_states = out_states.view(-1,self.n_node,self.hidden_dim,self.n_edge).transpose(2,3).transpose(1,2).contiguous()
            out_states = out_states.view(-1, self.n_node*self.n_edge, self.hidden_dim)
            x = self.gated_update(in_states, out_states, x, m)
            all_x.append(x)

        if self.task_id == 18:
            output = self.graph_aggregate(torch.cat((x, a), 2))
            output = self.fc_output(output)
        elif self.task_id == 19:
            step1 = self.graph_aggregate(torch.cat((all_x[0], a), 2))
            step1 = self.fc_output(step1).view(-1,1,self.n_output)
            step2 = self.graph_aggregate(torch.cat((all_x[1], a), 2))
            step2 = self.fc_output(step2).view(-1,1,self.n_output)
            output = torch.cat((step1,step2), 1)
        else:
            output = self.fc1(torch.cat((x, a), 2))
            output = self.tanh(output)
            output = self.fc2(output).sum(2)
        return output


class GraphFeature(nn.Module):
    '''
    Output a Graph-Level Feature
    '''
    def __init__(self, hidden_dim, n_node, n_edge, n_anno):
        super(GraphFeature, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_node = n_node
        self.n_edge = n_edge
        self.n_anno = n_anno

        self.fc_i = nn.Linear(self.hidden_dim + self.n_anno, self.hidden_dim)
        self.fc_j = nn.Linear(self.hidden_dim + self.n_anno, self.hidden_dim)
        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x):
        '''
        input x: [batch_size, num_node, hidden_size + annotation]
        output x: [batch_size, hidden_size]
        '''
        x_sigm = self.sigmoid(self.fc_i(x))
        x_tanh = self.tanh(self.fc_j(x))
        x_new = (x_sigm * x_tanh).sum(1)

        return self.tanh(x_new)


class GatedPropagation(nn.Module):
    '''
    Gated Recurrent Propagation
    '''
    def __init__(self, hidden_dim, n_node, n_edge):
        super(GatedPropagation, self).__init__()

        self.hidden_dim = hidden_dim
        self.n_node = n_node
        self.n_edge = n_edge

        self.gate_r = nn.Linear(self.hidden_dim*3, self.hidden_dim)
        self.gate_z = nn.Linear(self.hidden_dim*3, self.hidden_dim)
        self.trans  = nn.Linear(self.hidden_dim*3, self.hidden_dim)

        self.sigmoid = nn.Sigmoid()
        self.tanh = nn.Tanh()

    def forward(self, x_in, x_out, x_curt, matrix):
        matrix_in  = matrix[:, :, :self.n_node*self.n_edge]
        matrix_out = matrix[:, :, self.n_node*self.n_edge:]

        a_in  = torch.bmm(matrix_in, x_in)
        a_out = torch.bmm(matrix_out, x_out)
        a = torch.cat((a_in, a_out, x_curt), 2)

        z = self.sigmoid(self.gate_z(a))
        r = self.sigmoid(self.gate_r(a))

        joint_input = torch.cat((a_in, a_out, r * x_curt), 2)
        h_hat = self.tanh(self.trans(joint_input))
        output = (1 - z) * x_curt + z * h_hat

        return output
        