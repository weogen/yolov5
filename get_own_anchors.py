

import torch

#model=torch.hub.load("ultralytics/yolov5","yolov5s",autoshape=False)
model=torch.load('yolov5s.pt')['model']
m=model.model[-1]
print(m.anchor_grid.squeeze())

