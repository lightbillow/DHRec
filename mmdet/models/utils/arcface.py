import torch.nn as nn
import torch.nn.functional as F
import torch
import math
from mmcv.cnn import normal_init, bias_init_with_prob

class ArcFace(nn.Module):
    def __init__(self, embedding_size, class_num, s=30.0, m=0.50):
        super().__init__()
        self.in_features = embedding_size
        self.out_features = class_num
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(class_num, embedding_size))
        nn.init.normal_(self.weight, 0, 1)
        # nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label = None):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = ((1.0 - cosine.pow(2)).clamp(0, 1)).sqrt()
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)  # drop to CosFace
        output = cosine * 1.0  # make backward works
        if not self.training:
            return output * self.s
        else:
            assert(label is not None)
            batch_size = len(output)
            output[range(batch_size), label] = phi[range(batch_size), label]
            return output * self.s

class FocalArcFace(nn.Module):
    def __init__(self, embedding_size, class_num, s=30.0, m=0.50):
        super().__init__()
        self.in_features = embedding_size
        self.out_features = class_num
        self.s = s
        self.m = m
        self.weight = nn.Parameter(torch.FloatTensor(class_num, embedding_size))
        nn.init.normal_(self.weight, 0, 1)
        # nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.th = math.cos(math.pi - m)
        self.mm = math.sin(math.pi - m) * m

    def forward(self, input, label = None):
        cosine = F.linear(F.normalize(input), F.normalize(self.weight))
        sine = ((1.0 - cosine.pow(2)).clamp(0, 1)).sqrt()
        phi = cosine * self.cos_m - sine * self.sin_m
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)  # drop to CosFace
        output = cosine * 1.0  # make backward works
        if self.training:
            assert(label is not None)
            batch_size = len(output)
            output[range(batch_size), label] = phi[range(batch_size), label]
        output = output * self.s
        output = output - output[:, -1].view(-1,1).expand_as(output)
        return output[:,:-1].contiguous()