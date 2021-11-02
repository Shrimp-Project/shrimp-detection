import torch

def one_hot_embedding(labels, num_classes):
    y = torch.eye(num_classes)
    return y[labels]