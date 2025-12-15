import os
import sys
import cv2
import pickle
import torch
import time
import argparse
from glob import glob
from tqdm import tqdm
from datetime import datetime
import numpy as np
import pandas as pd
import albumentations as A
from threading import Thread

from model.resnet50 import ResNet

from model.LSD_transformer import Transformer
from utils.parser import ParserUse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from dataset.CNCSP import VideoSample
from xlstm1.xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)

phase_dict = {}
phase_dict_key = ['CO', 'Durotomy', 'IU', 'RTE', 'IE', 'ITE', 'Hemostasis']
for i in range(len(phase_dict_key)):
    phase_dict[phase_dict_key[i]] = i

label_dict = {}
phase_dict_key = ['CO', 'Durotomy', 'IU', 'RTE', 'IE', 'ITE', 'Hemostasis']
for i, phase in enumerate(phase_dict_key):
    label_dict[i] = phase

class PhaseSeg(object):
    """
    The class performs generic object detection on a video file.
    It uses yolo5 pretrained model to make inferences and opencv2 to manage frames.
    Included Features:
    1. Reading and writing of video file using  Opencv2
    2. Using pretrained model to make inferences on frames.
    3. Use the inferences to plot boxes on objects along with labels.
    Upcoming Features:
    """
    def __init__(self, record=False, quiet=False, arg=None):
        """
        :param input_file: provide youtube url which will act as input for the model.
        :param out_file: name of a existing file, or a new file in which to write the output.
        :return: void
        """
        self.arg = arg
        self.quiet = quiet
        self.record = record
        self.model = self.load_model()
        self.video_file = os.path.join("./results/records", "record_{}.avi".format(self.arg.log_time))
        if not os.path.isdir(os.path.dirname(self.video_file)):
            os.makedirs(os.path.dirname(self.video_file))
        self.frame_feature_cache = None
        self.frame_cache_len = 2 ** (self.arg.mstcn_layers + 1) - 1
        self.temporal_feature_cache = None
        self.label2phase_dict = label_dict
        self.aug = A.Compose([
            A.Resize(250, 250),
            A.CenterCrop(224, 224),
            A.Normalize()
        ])
        self.frame_idx = 0
        self.player = cv2.VideoCapture(0)
        x_shape = int(self.player.get(cv2.CAP_PROP_FRAME_WIDTH))
        y_shape = int(self.player.get(cv2.CAP_PROP_FRAME_HEIGHT))
        four_cc = cv2.VideoWriter_fourcc(*"XVID")

        if self.record:
            self.out = cv2.VideoWriter(self.video_file, four_cc, 8, (x_shape, y_shape), True)
        self.thread = Thread(target=self.get_frame, args=())
        self.thread.daemon = True
        self.thread.start()

    def get_frame(self):
        while True:
            if self.player.isOpened():
                self.frame_idx += 1
                (self.status, self.frame) = self.player.read()

    def save_preds(self, timestamps, frame_idxs, preds):

        pd_label = pd.DataFrame({"Time": timestamps, "Frame": frame_idxs, "Phase": preds})
        pd_label = pd_label.astype({"Time": "str", "Frame": "int", "Phase": "str"})
        save_file = self.video_file.replace(".avi", ".txt")
        print(save_file)
        pd_label.to_csv(save_file, index=False, header=None, sep="\t")

    def load_model(self):
        """
        Function loads the yolo5 model from PyTorch Hub.
        """
        self.resnet = ResNet(out_channels=self.arg.out_classes, has_fc=False)
        paras = torch.load(self.arg.resnet_model)["model"]
        paras = {k: v for k, v in paras.items() if "fc" not in k}
        paras = {k: v for k, v in paras.items() if "embed" not in k}
        self.resnet.load_state_dict(paras, strict=True)
        self.resnet.cuda()
        self.resnet.eval()

        # self.fusion = MultiStageModel(mstcn_stages=self.arg.mstcn_stages, mstcn_layers=self.arg.mstcn_layers,
        #                               mstcn_f_maps=self.arg.mstcn_f_maps, mstcn_f_dim=self.arg.mstcn_f_dim,
        #                               out_features=self.arg.out_classes, mstcn_causal_conv=True, is_train=False)

        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=1
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="cuda",
                    num_heads=1,
                    conv1d_kernel_size=4,
                    bias_init="powerlaw_blockdependent",
                ),

                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=2048,
            num_blocks=2,
            embedding_dim=1560,
            slstm_at=[],

        )
        self.fusion_model = xLSTMBlockStack(cfg)
        paras = torch.load(self.arg.fusion_model)
        self.fusion.load_state_dict(paras)
        self.fusion.cuda()
        self.fusion.eval()

        self.transformer = Transformer(self.arg.mstcn_f_maps, self.arg.mstcn_f_dim, self.arg.out_classes, self.arg.trans_seq, d_model=self.arg.mstcn_f_maps)
        paras = torch.load(self.arg.trans_model)
        self.transformer.load_state_dict(paras)
        self.transformer.cuda()
        self.transformer.eval()

    def cache_frame_features(self, feature):
        if self.frame_feature_cache is None:
            self.frame_feature_cache = feature
        elif self.frame_feature_cache.shape[0] > self.frame_cache_len:
            self.frame_feature_cache = torch.cat([self.frame_feature_cache[1:], feature], dim=0)
        else:
            self.frame_feature_cache = torch.cat([self.frame_feature_cache, feature], dim=0)
        return self.frame_feature_cache

    def seg_frame(self, frame):
        """
        function scores each frame of the video and returns results.
        :param frame: frame to be infered.
        :return: labels and coordinates of objects found.
        """
        frame = self.aug(image=frame)["image"]
        with torch.no_grad():
            frame = np.expand_dims(np.transpose(frame, [2, 0, 1]), axis=0)
            frame = torch.tensor(frame).cuda()
            frame_feature = self.resnet(frame)
            # print(frame_feature.size())
            cat_frame_feature = self.cache_frame_features(frame_feature).unsqueeze(0)
            temporal_feature = self.fusion(cat_frame_feature.transpose(1, 2))

            # Temporal feature: [1, 5, 512], Frame feature：[1, 512, 2048]
            pred = self.transformer(temporal_feature.detach(), cat_frame_feature)[-1].cpu().numpy()
        return self.label2phase_dict[np.argmax(pred, axis=0)]

    def add_text(self, fc, results, fps, frame):
        cv2.putText(frame, "   Time: {:<55s}".format(fc), (30, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "  Phase: {:<15s}".format(results), (30, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, " Trainee: {:<15s}".format(fps), (30, 90),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return frame

    def infer(self):

        tfcc = 0
        preds = []
        timestamps = []
        frame_idxs = []
        results = ""
        fps = -1
        while True:
            tfcc += 1
            start_time = time.time()
            if not self.status:
                break
            date_time = datetime.now().strftime("%m/%d/%Y-%H:%M:%S.%f")
            frame = cv2.cvtColor(self.frame, cv2.COLOR_BGR2RGB)
            if tfcc % 5 == 1:
                results = self.seg_frame(frame)
                timestamps.append(date_time)
                frame_idxs.append(self.frame_idx)
                preds.append(results)
            end_time = time.time()
            if tfcc % 10 == 1:
                fps = 1/np.round(end_time - start_time, 3)

            frame = self.add_text(date_time, results, fps, self.frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if self.record:
                self.out.write(frame)
            if not self.quiet:
                cv2.imshow('frame', frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        self.save_preds(timestamps, frame_idxs, preds)
        self.player.release()
        cv2.destroyAllWindows()



class PhaseCom(object):
    def __init__(self, record=False, quiet=False, arg=None):
        """
        :param input_file: provide youtube url which will act as input for the model.
        :param out_file: name of a existing file, or a new file in which to write the output.
        :return: void
        """
        self.arg = arg
        self.quiet = quiet
        self.record = record
        self.model = self.load_model()
        self.frame_feature_cache = None
        self.frame_cache_len = 2 ** (self.arg.mstcn_layers + 1) - 1
        self.temporal_feature_cache = None
        self.label2phase_dict = label_dict
        self.aug = A.Compose([
            A.Resize(250, 250),
            A.CenterCrop(224, 224),
            A.Normalize()
        ])

    def save_preds(self, timestamps, frame_idxs, preds):

        pd_label = pd.DataFrame({"Time": timestamps, "Frame": frame_idxs, "Phase": preds})
        pd_label = pd_label.astype({"Time": "str", "Frame": "int", "Phase": "str"})
        save_file = self.video_file.replace(".avi", ".txt")
        print(save_file)
        pd_label.to_csv(save_file, index=False, header=None, sep="\t")

    def load_model(self):
        """
        Function loads the yolo5 model from PyTorch Hub.
        """
        self.resnet = ResNet(out_channels=self.arg.out_classes, has_fc=False)
        paras = torch.load(self.arg.resnet_model)["model"]
        paras = {k: v for k, v in paras.items() if "fc" not in k}
        paras = {k: v for k, v in paras.items() if "embed" not in k}
        self.resnet.load_state_dict(paras, strict=True)
        self.resnet.cuda()
        self.resnet.eval()

        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4, qkv_proj_blocksize=4, num_heads=1
                )
            ),
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend="cuda",
                    num_heads=1,
                    conv1d_kernel_size=4,
                    bias_init="powerlaw_blockdependent",
                ),

                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=2048,
            num_blocks=2,
            embedding_dim=1560,
            slstm_at=[],

        )


        self.fusion = xLSTMBlockStack(cfg)
        paras = torch.load(self.arg.fusion_model)
        self.fusion.load_state_dict(paras)
        self.fusion.cuda()
        self.fusion.eval()

        self.transformer = Transformer(self.arg.mstcn_f_maps, self.arg.mstcn_f_dim, self.arg.out_classes, self.arg.trans_seq, d_model=self.arg.mstcn_f_maps)
        paras = torch.load(self.arg.trans_model)
        self.transformer.load_state_dict(paras)
        self.transformer.cuda()
        self.transformer.eval()

    def cache_frame_features(self, feature):
        if self.frame_feature_cache is None:
            self.frame_feature_cache = feature
        elif self.frame_feature_cache.shape[0] > self.frame_cache_len:
            self.frame_feature_cache = torch.cat([self.frame_feature_cache[1:], feature], dim=0)
        else:
            self.frame_feature_cache = torch.cat([self.frame_feature_cache, feature], dim=0)
        return self.frame_feature_cache

    def seg_frame(self, frame):
        frame = cv2.resize(frame, (250, 250))
        frame = self.aug(image=frame)["image"]
        with torch.no_grad():
            frame = np.expand_dims(np.transpose(frame, [2, 0, 1]), axis=0)
            frame = torch.tensor(frame).cuda()
            frame_feature = self.resnet(frame)

            print(f"1. ResNet output shape: {frame_feature.shape}")

            # 如果resnet输出4D特征 [B, C, H, W]
            if len(frame_feature.shape) == 4:
                # 全局平均池化
                frame_feature = F.adaptive_avg_pool2d(frame_feature, (1, 1))
                # 转换为3D: [B, C, 1, 1] -> [B, C, 1]
                frame_feature = frame_feature.squeeze(-1)  # 移除最后一个维度
                print(f"2. After squeeze: {frame_feature.shape}")

            # 确保是3D张量 [B, C, L] 其中L是序列长度
            if len(frame_feature.shape) == 2:
                # 如果是2D [B, C]，添加序列维度
                frame_feature = frame_feature.unsqueeze(-1)  # [B, C] -> [B, C, 1]

            print(f"3. Before adapter: {frame_feature.shape}")

            # 添加适配层
            if not hasattr(self, 'feature_adapter'):
                # 使用Conv1d
                in_channels = frame_feature.shape[1]
                self.feature_adapter = nn.Conv1d(
                    in_channels=in_channels,
                    out_channels=1560,
                    kernel_size=1
                ).cuda()
                nn.init.kaiming_normal_(self.feature_adapter.weight, mode='fan_out', nonlinearity='relu')
                nn.init.constant_(self.feature_adapter.bias, 0)
                print(f"Created Conv1d adapter: {in_channels} -> 1560")

            # 通过适配层
            frame_feature = self.feature_adapter(frame_feature)  # [1, 2048, 1] -> [1, 1560, 1]
            print(f"4. After adapter: {frame_feature.shape}")

            # 移除序列维度以适应cache_frame_features
            frame_feature = frame_feature.squeeze(-1)  # [1, 1560, 1] -> [1, 1560]
            print(f"5. After squeeze: {frame_feature.shape}")

            cat_frame_feature = self.cache_frame_features(frame_feature).unsqueeze(0)
            print(f"6. Cat frame feature shape: {cat_frame_feature.shape}")

            temporal_feature = self.fusion(cat_frame_feature.transpose(1, 2))

            pred = self.transformer(temporal_feature.detach(), cat_frame_feature)[-1].cpu().numpy()

        return self.label2phase_dict[np.argmax(pred, axis=0)]
    # def seg_frame(self, frame):
    #     """
    #     function scores each frame of the video and returns results.
    #     :param frame: frame to be infered.
    #     :return: labels and coordinates of objects found.
    #     """
    #     frame = cv2.resize(frame, (250, 250))
    #     frame = self.aug(image=frame)["image"]
    #     with torch.no_grad():
    #         frame = np.expand_dims(np.transpose(frame, [2, 0, 1]), axis=0)
    #         frame = torch.tensor(frame).cuda()
    #         frame_feature = self.resnet(frame)
    #         # print(frame_feature.size())
    #         cat_frame_feature = self.cache_frame_features(frame_feature).unsqueeze(0)
    #         temporal_feature = self.fusion(cat_frame_feature.transpose(1, 2))
    #
    #         # Temporal feature: [1, 5, 512], Frame feature：[1, 512, 2048]
    #         pred = self.transformer(temporal_feature.detach(), cat_frame_feature)[-1].cpu().numpy()
    #     return self.label2phase_dict[np.argmax(pred, axis=0)]

    def add_text(self, fc, results, fps, frame):
        cv2.putText(frame, "   Time: {:<55s}".format(fc), (30, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, "  Phase: {:<15s}".format(results), (30, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv2.putText(frame, " Trainee: {:<15s}".format(fps), (30, 90),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        return frame