import torch
import torch.nn as nn
import torch.nn.functional as F
from xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)


class MultiStageModel(nn.Module):
    def __init__(self, xLSTMBlockStack_config, out_features, is_train=True):
        self.num_classes = out_features  # 5
        self.is_train = is_train
        super(MultiStageModel, self).__init__()

        # 创建 xLSTMBlockStack 实例
        self.xlstm_stack = xLSTMBlockStack(xLSTMBlockStack_config)

        # 添加全连接层，将 xLSTMBlockStack 的输出转为分类输出
        self.fc = nn.Linear(xLSTMBlockStack_config.embedding_dim, out_features)

        self.smoothing = False

    def forward(self, x):
        out = self.xlstm_stack(x)
        out_classes = self.fc(out.mean(dim=1))

        if self.is_train:
            outputs_classes = out_classes.unsqueeze(0)
            return outputs_classes
        else:
            return out_classes

    @staticmethod
    def add_model_specific_args(parser):  # pragma: no cover
        mstcn_reg_model_specific_args = parser.add_argument_group(
            title='xlstm specific args options')
        mstcn_reg_model_specific_args.add_argument("--context_length",
                                                   default=128,
                                                   type=int)
        mstcn_reg_model_specific_args.add_argument("--num_blocks",
                                                   default=5,
                                                   type=int)
        mstcn_reg_model_specific_args.add_argument("--embedding_dim",
                                                   default=64,
                                                   type=int)
        mstcn_reg_model_specific_args.add_argument("--mlstm_conv1d_kernel_size",
                                                   default=3,
                                                   type=int)
        mstcn_reg_model_specific_args.add_argument("--mlstm_qkv_proj_blocksize",
                                                   default=2,
                                                   type=int)
        mstcn_reg_model_specific_args.add_argument("--mlstm_num_heads",
                                                   default=8,
                                                   type=int)
        mstcn_reg_model_specific_args.add_argument("--slstm_backend",
                                                   default="cpu",
                                                   type=str)
        mstcn_reg_model_specific_args.add_argument("--slstm_num_heads",
                                                   default=8,
                                                   type=int)
        mstcn_reg_model_specific_args.add_argument("--slstm_conv1d_kernel_size",
                                                   default=3,
                                                   type=int)
        mstcn_reg_model_specific_args.add_argument("--slstm_bias_init",
                                                   default="normal",
                                                   type=str)
        mstcn_reg_model_specific_args.add_argument("--feedforward_proj_factor",
                                                   default=2.0,
                                                   type=float)
        mstcn_reg_model_specific_args.add_argument("--feedforward_act_fn",
                                                   default="relu",
                                                   type=str)
        mstcn_reg_model_specific_args.add_argument("--slstm_at",
                                                   default=[0, 2, 4],
                                                   type=list)
        return parser


# 示例：如何创建配置并实例化 MultiStageModel
if __name__ == "__main__":
    cfg = xLSTMBlockStackConfig(
        mlstm_block=mLSTMBlockConfig(
            mlstm=mLSTMLayerConfig(
                conv1d_kernel_size=3,
                qkv_proj_blocksize=2,
                num_heads=8
            )
        ),
        slstm_block=sLSTMBlockConfig(
            slstm=sLSTMLayerConfig(
                backend="cpu",
                num_heads=8,
                conv1d_kernel_size=3,
                bias_init="normal",
            ),
            feedforward=FeedForwardConfig(proj_factor=2.0, act_fn="relu"),
        ),
        context_length=128,
        num_blocks=5,
        embedding_dim=64,
        slstm_at=[0, 2, 4],
    )

    model = MultiStageModel(cfg, out_features=5, is_train=True)
    x = torch.randn(4, 128, 64)  # 假设输入数据的形状
    outputs = model(x)
    print(outputs.shape)


