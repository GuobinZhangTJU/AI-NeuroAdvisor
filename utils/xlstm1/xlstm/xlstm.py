import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from xlstm_main.xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)

# 定义 xLSTM 配置
cfg = xLSTMBlockStackConfig(
    mlstm_block=mLSTMBlockConfig(
        mlstm=mLSTMLayerConfig(
            conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=4
        )
    ),
    slstm_block=sLSTMBlockConfig(
        slstm=sLSTMLayerConfig(
            backend="cuda",
            num_heads=4,
            conv1d_kernel_size=4,
            bias_init="powerlaw_blockdependent",
        ),
        feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
    ),
    context_length=256,
    num_blocks=7,
    embedding_dim=128,
    slstm_at=[1],
)

class xLSTMModel(nn.Module):
    def __init__(self, xlstm_stages, xlstm_layers, xlstm_f_maps, xlstm_f_dim, out_features, is_train=True):
        self.num_stages = xlstm_stages
        self.num_layers = xlstm_layers
        self.num_f_maps = xlstm_f_maps
        self.dim = xlstm_f_dim
        self.num_classes = out_features
        self.is_train = is_train
        print(
            f"num_stages_classification: {self.num_stages}, num_layers: {self.num_layers}, num_f_maps:"
            f" {self.num_f_maps}, dim: {self.dim}")
        super(xLSTMModel, self).__init__()
        self.stage1 = SingleStageModel(self.num_layers,
                                       self.num_f_maps,
                                       self.dim,
                                       self.num_classes)
        self.stages = nn.ModuleList([
            copy.deepcopy(
                SingleStageModel(self.num_layers,
                                 self.num_f_maps,
                                 self.num_classes,
                                 self.num_classes,
                                 is_train=is_train))
            for s in range(self.num_stages - 1)
        ])
        self.smoothing = False

    def forward(self, x):
        out_classes = self.stage1(x)
        outputs_classes = out_classes.unsqueeze(0)

        for stage in self.stages:
            out_classes = stage(F.softmax(out_classes, dim=1))
            outputs_classes = torch.cat((outputs_classes, out_classes.unsqueeze(0)), dim=0)
        return outputs_classes

class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes, is_train=True):
        super(SingleStageModel, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.is_train = is_train

        self.xlstm_stack = xLSTMBlockStack(cfg)

        self.conv_out_classes = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        out = self.xlstm_stack(out)
        if self.is_train:
            out = self.conv_out_classes(out)
        return out

class xLSTMModel1(nn.Module):
    def __init__(self, xlstm_stages, xlstm_layers, xlstm_f_maps, xlstm_f_dim, out_features):
        self.num_stages = xlstm_stages
        self.num_layers = xlstm_layers
        self.num_f_maps = xlstm_f_maps
        self.dim = xlstm_f_dim
        self.num_classes = out_features
        print(
            f"num_stages_classification: {self.num_stages}, num_layers: {self.num_layers}, num_f_maps:"
            f" {self.num_f_maps}, dim: {self.dim}")
        super(xLSTMModel1, self).__init__()
        self.stage1 = SingleStageModel1(self.num_layers,
                                       self.num_f_maps,
                                       self.dim,
                                       self.num_classes)
        self.stages = nn.ModuleList([
            copy.deepcopy(
                SingleStageModel1(self.num_layers,
                                 self.num_f_maps,
                                 self.num_classes,
                                 self.num_classes))
            for s in range(self.num_stages - 1)
        ])
        self.smoothing = False

    def forward(self, x):
        out_classes, _ = self.stage1(x)
        outputs_classes = out_classes.unsqueeze(0)
        for s in self.stages:
            out_classes, out = s(F.softmax(out_classes, dim=1))
            outputs_classes = torch.cat(
                (outputs_classes, out_classes.unsqueeze(0)), dim=0)
        return outputs_classes

class SingleStageModel1(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(SingleStageModel1, self).__init__()
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)

        self.xlstm_stack = xLSTMBlockStack(cfg)
        self.conv_out_classes = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x):
        out = self.conv_1x1(x)
        out = self.xlstm_stack(out)
        out_classes = self.conv_out_classes(out)
        return out_classes, out
