import os.path
import shutil
import warnings
warnings.filterwarnings("ignore")
import pickle
import random
import argparse
import numpy as np
from tqdm import tqdm

import torch
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataset.CNCSP import Dataset
from utils.parser import ParserUse

from model.resnet50 import ResNet


def features(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    model = ResNet(out_channels=args.out_classes, has_fc=False)
    paras = torch.load(args.resnet_model,map_location=torch.device('cpu'))["model"]
    paras = {k: v for k, v in paras.items() if "fc" not in k}
    paras = {k: v for k, v in paras.items() if "embed" not in k}
    model.load_state_dict(paras, strict=True)
    model.cuda()
    model.eval()

    with open(args.data_file, "rb") as f:
        data_dict = pickle.load(f)
    emb_dataset = Dataset(data_dict=data_dict, data_idxs=args.train_names + args.val_names + args.test_names, is_train=False, get_name=True, has_label=args.has_label)
    emb_loader = DataLoader(dataset=emb_dataset, batch_size=args.resnet_train_bs, num_workers=args.num_worker, shuffle=False, drop_last=False)

    feature_embs = {}
    with torch.no_grad():
        for data in tqdm(emb_loader, total=len(emb_loader)):
            imgs, base_names = data[0].cuda(non_blocking=True), data[-1]
            base_names = list(base_names)
            for idx, base_name in enumerate(base_names):
                if "--" in base_name:
                    base_names[idx] = base_name.split("--")[0]
                else:
                    base_names[idx] = base_name.split("-")[0]
            img_features = model(imgs).cpu().numpy()
            for idx, data_name in enumerate(base_names):
                # try:
                if data_name in feature_embs:
                    feature_embs[data_name].append(img_features[idx])
                else:
                    feature_embs[data_name] = [img_features[idx]]


    # Check length of embedding
    with open(args.data_file, "rb") as f:
        all_data = pickle.load(f)
    data_names = list(feature_embs.keys())
    for data_name in data_names:
        if len(all_data[data_name]["img"]) != len(feature_embs[data_name]):
            print(f"Error in data {data_name}")
            print(f"Number of images {len(all_data[data_name]['img'])}")
            print(f"Number of features {len(feature_embs[data_name])}")
            raise ValueError("#imgs != #features")

    args.emb_file = os.path.join(os.path.dirname(args.emb_file), f"emb_ESDSafety{args.log_time}.pkl")
    print(">>>"*10, "Emb dataset saved to ", args.emb_file)
    with open(args.emb_file, "wb") as f:
        pickle.dump(feature_embs, f)


    return args

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--cfg', default='train', required=True, type=str,
                        help='Your detailed configuration of the network')

    args = parser.parse_args()
    args = ParserUse(args.cfg, log="generate").add_args(args)

    ckpts = args.makedir()
    features(args)