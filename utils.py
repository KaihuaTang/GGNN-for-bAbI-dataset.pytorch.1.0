##########################
# Reimplementation of Gated Graph Sequence Neural Networks (GGNN)
# Paper Link: https://arxiv.org/abs/1511.05493
# Code Author: Kaihua Tang
# Environment: Python 3.6, Pytorch 1.0
##########################

import config
import torch
import numpy as np

def get_file_location(mode, task_id, info_type='graphs', data_id=1, rnn=False):
    '''
    mode = 'train' or 'test'
    task_id = 4, 15, 16, 18, 19
    info_type = 'edge_types', 'graphs', 'labels', 'node_ids', 'question_types', 
    data_id = 1~10, random generated 10 dataset
    rnn = whether load rnn data
    '''
    root = 'babi_data/'
    processed = 'processed_'+str(data_id)+'/'
    
    if rnn:
        # TODO Later
        output_path = None
    else:
        output_path = root + processed + mode + '/' + str(task_id) + '_' + info_type + '.txt'
    #if config.VERBAL: print('Load from '+ output_path)
    return output_path


def print_param(model):
    """
    Prints parameters of a model
    :param opt:
    :return:
    """
    st = {}
    strings = []
    total_params = 0
    for p_name, p in model.named_parameters():

        if not ('bias' in p_name.split('.')[-1] or 'bn' in p_name.split('.')[-1]):
            st[p_name] = ([str(x) for x in p.size()], np.prod(p.size()), p.requires_grad)
        total_params += np.prod(p.size())
    for p_name, (size, prod, p_req_grad) in sorted(st.items(), key=lambda x: -x[1][1]):
        strings.append("{:<50s}: {:<16s}({:8d}) ({})".format(
            p_name, '[{}]'.format(','.join(size)), prod, 'grad' if p_req_grad else '    '
        ))
    return '\n {:.1f}M total parameters \n ----- \n \n{}'.format(total_params / 1000000.0, '\n'.join(strings))


def clip_grad_norm(named_parameters, max_norm, clip=False, verbose=False):
    """Clips gradient norm of an iterable of parameters.

    The norm is computed over all gradients together, as if they were
    concatenated into a single vector. Gradients are modified in-place.

    Arguments:
        parameters (Iterable[Variable]): an iterable of Variables that will have
            gradients normalized
        max_norm (float or int): max norm of the gradients

    Returns:
        Total norm of the parameters (viewed as a single vector).
    """
    max_norm = float(max_norm)

    total_norm = 0
    param_to_norm = {}
    param_to_shape = {}
    for n, p in named_parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm ** 2
            param_to_norm[n] = param_norm
            param_to_shape[n] = p.size()

    total_norm = total_norm ** (1. / 2)
    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1 and clip:
        for _, p in named_parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)

    if verbose:
        print('---Total norm {:.3f} clip coef {:.3f}-----------------'.format(total_norm, clip_coef))
        for name, norm in sorted(param_to_norm.items(), key=lambda x: -x[1]):
            print("{:<50s}: {:.3f}, ({})".format(name, norm, param_to_shape[name]))
        print('-------------------------------', flush=True)

    return total_norm