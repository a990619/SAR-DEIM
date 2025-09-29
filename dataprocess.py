import json
import os.path

import cv2

# 读取JSON文件
with open('/media/dzy/deep1/train_data2/sarship/coco/val.json', 'r', encoding='utf-8') as file:
    data = json.load(file)
allimgs = {}
for img in data['images']:
    allimgs[img["id"]] = img["file_name"]

allanns = {}
for ann in data['annotations']:
    if ann["image_id"] not in allanns.keys():
        allanns[ann["image_id"]] = []
    allanns[ann["image_id"]].append(ann["bbox"])


for anns in allanns.items():
    impath = allimgs[anns[0]]
    name = impath.split(os.sep)[-1]
    boxes = anns[1]
    boxes = [ [int(y) for y in x]  for x in boxes ]

    img = cv2.imread(os.path.join("/media/dzy/deep1/train_data2/sarship/images/val",name))
    for box in boxes:
        cv2.rectangle(img, (box[0],box[1]), (box[0]+box[2],box[1]+box[3]),  (255, 0, 0), 2)
    cv2.imwrite("11.jpg",img)
   # cv2.waitKey(1000)
    break


