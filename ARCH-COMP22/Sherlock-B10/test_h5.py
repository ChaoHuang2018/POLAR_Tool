from tensorflow import keras
import numpy as np

model = keras.models.load_model('controllerB_nnv.h5')
model.summary()

x = np.array([[0.69], [-0.7], [-0.4], [0.59]])
print(x.shape)
#u = model.predict(x)
y = np.transpose(x)
print(y.shape)
u = model.predict(y)

print(u)


