from keras.models import load_model
import numpy as np

model = load_model('weights.h5')

values = np.array([[1, 93, 70, 31, 0, 30.4, 0.315, 23]])

print(values)
print(model.predict(values))
