from tensorflow import keras
import numpy as np

model = keras.models.load_model('controller_airplane.h5')
#model.summary()

x = np.array([[0], [0], [0], [1], [1], [1], [1], [1], [1], [0], [0], [0]])
print(x.shape)
#u = model.predict(x)
y = np.transpose(x)
print(y.shape)
u = model.predict(y)

print(u)


