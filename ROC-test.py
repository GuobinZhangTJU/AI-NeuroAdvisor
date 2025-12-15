import os
import pickle
import random
import logging
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn import metrics
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.LSD_transformer import LSD_Transformer
from dataset.CNCSP import VideoSample
from utils.parser import ParserUse
import argparse

from xlstm1.xlstm import (
    xLSTMBlockStack,
    xLSTMBlockStackConfig,
    mLSTMBlockConfig,
    mLSTMLayerConfig,
    sLSTMBlockConfig,
    sLSTMLayerConfig,
    FeedForwardConfig,
)
torch.set_printoptions(threshold=np.inf)
os.environ['CUDA_VISIBLE_DEVICES'] = "1,2,3"

# Create phase and label dictionaries
phase_dict = {phase: i for i, phase in enumerate(['CO', 'Durotomy', 'IU', 'RTE', 'IE', 'ITE', 'Hemostasis'])}
label_dict = {i: phase for i, phase in enumerate(['CO', 'Durotomy', 'IU', 'RTE', 'IE', 'ITE', 'Hemostasis'])}

def test_model(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    logging.info("\n\n\n" + "|| "*10 + "Begin testing model")

    # Configuration for the model
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
        embedding_dim=696,
        slstm_at=[],
    )

    # Initialize and load models
    fusion_model = xLSTMBlockStack(cfg)
    fusion_model.cuda()
    fusion_model.eval()

    trans_model = LSD_Transformer(args.mstcn_f_maps, args.mstcn_f_dim, args.out_classes, args.trans_seq, d_model=args.mstcn_f_maps)
    trans_model.load_state_dict(torch.load(args.trans_model))
    trans_model.cuda()
    trans_model.eval()

    # Load data
    with open(args.data_file, "rb") as f:
        data_dict = pickle.load(f)
    with open(args.emb_file, "rb") as f:
        emb_dict = pickle.load(f)

    test_data = VideoSample(data_dict=data_dict, data_idxs=args.test_names, data_features=emb_dict, is_train=False, get_name=True)
    test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=0, drop_last=False)

    if not os.path.isdir(args.pred_folder):
        os.makedirs(args.pred_folder)

    pred_label_files = []
    all_gt_labels = []
    all_pred_probs = []

    with torch.no_grad():
        for data in tqdm(test_loader, desc="Predicting"):
            img_features0, img_names = data[0].cuda(non_blocking=True), data[1]
            img_features = torch.transpose(img_features0, 1, 2)
            features = fusion_model(img_features).squeeze(1)
            p_classes = trans_model(features.detach(), img_features0).squeeze()
            preds = torch.argmax(p_classes, dim=-1).cpu().numpy().tolist()
            p_classes = p_classes.cpu().numpy()

            pd_label = pd.DataFrame({"Frame": list(range(1, len(preds)+1, 1)),
                                     "Phase": preds,
                                     "CO": p_classes[:, 0].tolist(),
                                     "Durotomy": p_classes[:, 1].tolist(),
                                     "IU": p_classes[:, 2].tolist(),
                                     "RTE": p_classes[:, 3].tolist(),
                                     "IE": p_classes[:, 4].tolist(),
                                     "ITE": p_classes[:, 5].tolist(),
                                     "Hemostasis": p_classes[:, 6].tolist()})
            pd_label = pd_label.astype({"Frame": "int",
                                        "Phase": "int",
                                        "CO": "float",
                                        "Durotomy": "float",
                                        "IU": "float",
                                        "RTE": "float",
                                        "IE": "float",
                                        "ITE": "float",
                                        "Hemostasis": "float"}).replace({"Phase": label_dict})

            if "--" in img_names[0][0]:
                base_name = os.path.basename(args.trans_model).split("_")[-1].split(".")[0] + "_T_" + os.path.basename(img_names[0][0].split("--")[0]) + ".txt"
            else:
                base_name = os.path.basename(args.trans_model).split("_")[-1].split(".")[0] + "_T_" + "case16" + ".txt"
            save_file = os.path.join(args.pred_folder, base_name)
            pd_label.to_csv(save_file, index=False, header=None, sep="\t")
            pred_label_files.append(save_file)

            # Collect ground truth and predictions for ROC
            gt_label_file = os.path.join(args.label_dir, os.path.basename(save_file).split('_T_')[-1])
            gt_label = pd.read_csv(gt_label_file, header=None, sep="\t", names=["Frame", "Phase"], index_col=False)
            gt_label = gt_label.replace({"Phase": phase_dict})
            gt_labels = gt_label["Phase"].tolist()
            pred_probs = pd_label.drop(columns=["Frame", "Phase"]).values

            all_gt_labels.extend(gt_labels)
            all_pred_probs.extend(pred_probs)

    # Convert lists to numpy arrays for roc_curve function
    all_gt_labels = np.array(all_gt_labels)
    all_pred_probs = np.array(all_pred_probs)
    # Compute ROC curve and ROC AUC for each class
    n_classes = all_pred_probs.shape[1]
    fpr = dict()
    tpr = dict()
    roc_auc = dict()

    for i in range(n_classes):
        fpr[i], tpr[i], _ = roc_curve(all_gt_labels, all_pred_probs[:, i], pos_label=i)
        roc_auc[i] = auc(fpr[i], tpr[i])

    # Plot ROC curves
    plt.figure()
    for i in range(n_classes):
        plt.plot(fpr[i], tpr[i], lw=2, label='ROC curve of class {0} (area = {1:0.2f})'.format(label_dict[i], roc_auc[i]))

    plt.plot([0, 1], [0, 1], 'k--', lw=2)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC)')
    plt.legend(loc="lower right")
    plt.savefig(os.path.join(args.pred_folder, 'roc_curve.png'))
    plt.show()

    print("Finished")
    accs = []
    for pred_label_file in pred_label_files:
        base_name = os.path.basename(pred_label_file)
        gt_label_file = os.path.join(args.label_dir, os.path.basename(pred_label_file).split('_T_')[-1])

        gt_label = pd.read_csv(gt_label_file, header=None, sep="\t", names=["Frame", "Phase"], index_col=False)
        gt_label = gt_label.replace({"Phase": phase_dict})
        gt_label = gt_label["Phase"].tolist()
        pred_label = pd.read_csv(pred_label_file, header=None, sep="\t", names=["Frame", "Phase"], index_col=False)
        pred_label = pred_label.replace({"Phase": phase_dict})
        pred_label = pred_label["Phase"].tolist()
        logging.info(">> " * 10 + base_name)
        acc = metrics.accuracy_score(gt_label, pred_label)
        accs.append(acc)
        logging.info("Accuracy {:>10.5f}".format(acc))

    print("|| "*10, "Mean: {:10.5f}".format(sum(accs) / len(accs)))
    return args

if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--cfg", default="train", required=True, type=str, help="Config file")
    args.add_argument("-n", default="", help="Note for testing")

    args = args.parse_args()
    args = ParserUse(args.cfg, "test").add_args(args)

    args.makedir()
    logging.info(args)
    logging.info("=" * 20 + "\n\n\n")

    test_model(args)
