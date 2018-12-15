##########################
# Reimplementation of Gated Graph Sequence Neural Networks (GGNN)
# Paper Link: https://arxiv.org/abs/1511.05493
# Code Author: Kaihua Tang
# Environment: Python 3.6, Pytorch 1.0
##########################

import utils
import config
import torch
import torch.utils.data as data
import numpy as np

def get_data_types(data_path):
    '''get edge/label/node/question type dictionary'''
    data_type = {}
    with open(data_path,'r') as f:
        for line in f:
            if len(line.strip()) == 0:
                pass
            else:
                line_tokens = line.strip('\n').split("=")
                assert(len(line_tokens)==2)
                data_type[line_tokens[0]] = line_tokens[1]
    return data_type


def load_graphs_from_file(file_name):
    ''' 
    load graph data from file 
    output = [data_size, 2, (num_fact/num_question), 3/num_answer]
    '''
    data_list = []
    edge_list = []
    target_list = []
    with open(file_name,'r') as f:
        for line in f:
            if len(line.strip()) == 0:
                data_list.append([edge_list,target_list])
                edge_list = []
                target_list = []
            else:
                digits = []
                line_tokens = line.split(" ")
                if line_tokens[0] == "?":
                    for i in range(1, len(line_tokens)):
                        digits.append(int(line_tokens[i]))
                    target_list.append(digits)
                else:
                    for i in range(len(line_tokens)):
                        digits.append(int(line_tokens[i]))
                    edge_list.append(digits)
    return data_list

def data_convert(data_list, n_annotation_dim, n_nodes, n_questions, task_id):
    '''
    data_preprocessing
    separate by answer type
    data_list: [data_size, 2, (num_fact/num_question), 3/num_answer]
    n_annotation_dim: original feature of node
    n_nodes: Number of nodes
    n_questions: Number of question types
    '''
    task_data_list = []
    for i in range(n_questions):
        task_data_list.append([])
    for item in data_list:
        fact_list = item[0]
        ques_list = item[1]
        for question in ques_list:
            question_type = question[0]
            if task_id == 19:
                question_output = np.zeros([2 if task_id == 19 else 1])
                assert(len(question) == 5)
                question_output[0] = question[-2]
                question_output[1] = question[-1]
            else:
                question_output = np.array(question[-1])
            annotation = np.zeros([n_nodes, n_annotation_dim])
            for i_anno in range(n_annotation_dim):
                annotation[question[i_anno + 1]-1][i_anno] = 1
            task_data_list[question_type-1].append([fact_list, annotation, question_output])
    return task_data_list

def create_adjacency_matrix(edges, n_nodes, n_edge_types):
    '''
    Incoming and outgoing are seperate
    edge type is considered
    '''
    a = np.zeros([n_nodes, n_nodes * n_edge_types * 2])
    for edge in edges:
        src_idx = edge[0]
        e_type = edge[1]
        tgt_idx = edge[2]
        a[tgt_idx-1][(e_type - 1) * n_nodes + src_idx - 1] =  1
        a[src_idx-1][(e_type - 1 + n_edge_types) * n_nodes + tgt_idx - 1] =  1
    return a

def get_loader(mode, task_id, data_id, train_size, val=False, ques_id=0):
    '''
    mode = 'train' or 'test'
    task_id = 4, 15, 16, 18, 19
    info_type = 'edge_types', 'graphs', 'labels', 'node_ids', 'question_types', 
    data_id = 1~10, random generated 10 dataset

    train_size = split train set and validation set
    val = validation or train
    ques_id = some task have different questions, default is 0

    Returns a data loader for the desired split 
    '''
    edge_type = get_data_types(utils.get_file_location(mode=mode, task_id=task_id, info_type='edge_types', data_id=data_id))
    node_type = get_data_types(utils.get_file_location(mode=mode, task_id=task_id, info_type='node_ids', data_id=data_id))
    question_type = get_data_types(utils.get_file_location(mode=mode, task_id=task_id, info_type='question_types', data_id=data_id))
    label_type = get_data_types(utils.get_file_location(mode=mode, task_id=task_id, info_type='labels', data_id=data_id))
    if len(label_type) == 0: label_type = node_type

    split = BABI(
        train_size = train_size,
        datapath = utils.get_file_location(mode=mode, task_id=task_id, info_type='graphs', data_id=data_id),
        edge_type = edge_type,
        node_type = node_type,
        question_type = question_type,
        label_type = label_type,
        is_validation = val,
        task_id = task_id,
        question_id = ques_id,
    )
    loader = torch.utils.data.DataLoader(
        split,
        batch_size=config.BATCH_SIZE,
        shuffle=(mode == 'train'),
        pin_memory=True,
        num_workers=config.NUM_WORKER,
    )
    return split, loader


class BABI(data.Dataset):
    """ BABI dataset """
    def __init__(self, train_size, datapath, edge_type, node_type, question_type, label_type, is_validation, task_id, question_id=0):
        super(BABI, self).__init__()
        
        self.edge_type = edge_type
        self.node_type = node_type
        self.question_type = question_type
        self.label_type = label_type

        self.n_edge_type = len(edge_type)
        self.n_node_type = len(node_type)
        self.n_question_type = len(question_type)
        self.n_label_type = len(label_type)
        
        self.task_id = task_id

        # data = [num_instances, 3(fact_list, annotation, question_output)]
        self.all_data = load_graphs_from_file(datapath)
        if is_validation:
            self.data = data_convert(self.all_data[train_size:], config.ANNOTATION_DIM[str(task_id)], self.n_node_type, self.n_question_type, task_id)[question_id]
        else:
            self.data = data_convert(self.all_data[:train_size], config.ANNOTATION_DIM[str(task_id)], self.n_node_type, self.n_question_type, task_id)[question_id]

    def __getitem__(self, item):
        adj_matrix = create_adjacency_matrix(self.data[item][0], self.n_node_type, self.n_edge_type)
        annotation = self.data[item][1]
        answer = self.data[item][2] - 1
        return adj_matrix, annotation, answer

    def __len__(self):
        return len(self.data)