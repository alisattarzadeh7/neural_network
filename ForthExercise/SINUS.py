import numpy as np
from keras.layers import Dense, Activation
from keras.models import Sequential
import matplotlib.pyplot as plt
import math

x = np.linspace(0, math.pi * 2, 1000)
y = np.sin(x)
model = Sequential([Dense(15, input_shape=(1,)), Activation('tanh'), Dense(7),
                    Activation('tanh'), Dense(3),Activation('tanh'),Dense(1)])

model.compile(loss='mean_squared_error', optimizer='SGD', metrics=['accuracy'])


model.fit(x, y, epochs=30000, batch_size=len(x), verbose=0)
predictions = model.predict(x)
plt.plot(x,y,'b', x, predictions, 'r--')
plt.ylabel('Y / Predicted Value')
plt.xlabel('X Value')
plt.legend(['Y','Predicted Value'])
plt.show()