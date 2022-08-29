import onnxruntime, onnx
import numpy as np

model_name = "model.onnx"
onnx_model = onnx.load(model_name)
#print('The model is:\n{}'.format(onnx_model))
#
#onnx.checker.check_model(onnx_model)


session = onnxruntime.InferenceSession(model_name)
#model.summary()

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name
print(input_name)
print(output_name)

#x = np.array([[106,106,0.28,0.28]]).astype(np.float32)
x = np.array([[70,70,-0.28,-0.28]]).astype(np.float32)

result = session.run([output_name], {input_name: x})[0]
print(result)

#x = np.array([[0], [0], [0], [1], [1], [1], [1], [1], [1], [0], [0], [0]])
#print(x.shape)
##u = model.predict(x)
#y = np.transpose(x)
#print(y.shape)
#u = model.predict(y)
#
#print(u)


