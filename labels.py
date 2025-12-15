import os
import pickle
import pandas as pd
from tqdm import tqdm
from glob import glob
from os.path import join as pjoin

"""
Save raw data as format:
    data = {video_name1: 
                        [[file1, [phase_label1, other_label1]], [file2, [phase_label2, other_label2]]...]}
"""

root_dir = '<path>'
save_dir = "<path>"
img_dir = os.path.join(root_dir, 'Images')
label_dir = os.path.join(root_dir, 'Lables')

phase_dict = {}
phase_dict_key = ['CO', 'Durotomy', 'IU', 'RTE', 'IE', 'ITE', 'Hemostasis']
for i in range(len(phase_dict_key)):
    phase_dict[phase_dict_key[i]] = i

data_names = [os.path.basename(x) for x in glob(pjoin(img_dir, "*"))]
data_names = sorted(data_names)

sorted_datas = {}
for data_name in tqdm(data_names, desc="Saving dataset"):
    img_file_dir = pjoin(img_dir, data_name)
    img_files = glob(pjoin(img_file_dir, "*"))
    img_files.sort(key=lambda x: int(x.split("-")[-1].split(".")[0]))

    label_file = pjoin(label_dir, data_name + ".txt")
    if os.path.isfile(label_file):
        phase_label = pd.read_csv(label_file, header=None, sep="[ ]{1,}|\t", engine="python")
        if len(phase_label.columns) == 5:
            phase_label.columns = ["Frame", "Phase", "#1", "#2", "#3"]
        elif len(phase_label.columns) == 2:
            phase_label.columns = ["Frame", "Phase"]
        else:
            raise ValueError("The header of label file cannot be matched")
        phase_label = phase_label.astype({"Frame": int, "Phase": str})
        phase_label = phase_label.replace({"Phase": phase_dict})
        phase_labels = phase_label["Phase"].tolist()
        # img_files = img_files[:len(phase_labels)]  # remove all img files after the removement of tumor
        if len(img_files) != len(phase_labels):
            img_files = img_files[:len(phase_labels)]
    else:
        phase_labels = None

    data_dict = {"img": img_files, "phase": phase_labels}
    sorted_datas[data_name] = data_dict

save_file = os.path.join(save_dir, "DATA.pkl")
print("Save to ", save_file)
with open(save_file, 'wb') as f:
    pickle.dump(sorted_datas, f)
print(f"{len(sorted_datas)} videos saved to {save_file}")