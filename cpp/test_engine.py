import torch
import numpy as np
import timeit

from efficientdet._C import Engine

img = np.random.randn(1, 3, 512, 512).astype(np.float32)
model = Engine.load("/research/object_detection/efficientdet/git/Yet-Another-EfficientDet-Pytorch/pretrained/efficientdet-d0.plan")
input_tensor = torch.from_numpy(img.copy()).to("cuda")
print("done")

for _ in range(10):
    t0 = timeit.default_timer()
    data = model(input_tensor)
    t1 = timeit.default_timer()
    print(t1-t0)