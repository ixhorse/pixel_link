import os
import sys
import csv
from glob import glob
import pdb

home = os.path.expanduser('~')
data_root = os.path.join(home, 'data/Datafountain')
verify_label_path = os.path.join(data_root, 'traindataset/verify_lable.csv')
result_path = '/home/mcc/working/pixel_link/model/df/test/icdar2015_test/model.ckpt-174238/verify'

# label dict
with open(verify_label_path, 'r') as f:
    reader = csv.reader(f)
    label = list(reader)[1:]

label_dict = {}
for row in label:
    name = row[0][:-4]
    cor = [int(x) for x in row[1:9]]
    x = cor[0:8:2]
    y = cor[1:8:2]
    xmin = min(x)
    xmax = max(x)
    ymin = min(y)
    ymax = max(y)
    if name in label_dict:
        label_dict[name].append([xmin, ymin, xmax, ymax])
    else:
        label_dict[name] = [[xmin, ymin, xmax, ymax]]

# predictions
result_list = []
txt_list = glob(result_path + '/*.txt')
for txt in txt_list:
    name = os.path.split(txt)[-1][4:-4]
    with open(txt, 'r') as f:
        data = [x.strip() for x in f.readlines()]
    prediction = []
    for row in data:
        cor = [int(x) for x in row.split(',')]
        x = cor[0:8:2]
        y = cor[1:8:2]
        xmin = min(x)
        xmax = max(x)
        ymin = min(y)
        ymax = max(y)
        prediction.append([xmin, ymin, xmax, ymax])
    result_list.append([name, prediction])

# eval
def IOU(box1, box2):
    u_xmin = max(box1[0], box2[0])
    u_ymin = max(box1[1], box2[1])
    u_xmax = min(box1[2], box2[2])
    u_ymax = min(box1[3], box2[3])
    if u_xmax <= u_xmin or u_ymax <= u_ymin:
        return 0
    u_area = (u_xmax - u_xmin) * (u_ymax - u_ymin)
    box1_area = (box1[2] - box1[0]) * (box1[3] - box1[1])
    box2_area = (box2[2] - box2[0]) * (box2[3] - box2[1])
    return 1.0 * u_area / (box1_area + box2_area - u_area)

total_right = 0
total_wrong = 0
total_miss = 0
for name, prediction in result_list:
    label = label_dict[name]
    right = 0
    wrong = 0
    miss = 0
    for pbox in prediction:
        for lbox in label:
            if IOU(pbox, lbox) > 0.7:
                right += 1
                break
    wrong = len(prediction) - right
    miss = len(label) - right
    assert wrong >= 0 and miss >= 0
    total_right += right
    total_wrong += wrong
    total_miss += miss

print('R=%.4f, P=%.4f' % 
    (1.0 * total_right / (total_right + total_miss),
    1.0 * total_right / (total_right + total_wrong)))
