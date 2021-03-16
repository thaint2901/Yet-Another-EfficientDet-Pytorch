import onnxruntime as rt
sess = rt.InferenceSession("/research/object_detection/efficientdet/git/Yet-Another-EfficientDet-Pytorch/pretrained/efficientdet-d0.onnx")
print("====INPUT====")
for i in sess.get_inputs():
    print("Name: {}, Shape: {}, Dtype: {}".format(i.name, i.shape, i.type))
print("====OUTPUT====")
for i in sess.get_outputs():
    print("Name: {}, Shape: {}, Dtype: {}".format(i.name, i.shape, i.type))