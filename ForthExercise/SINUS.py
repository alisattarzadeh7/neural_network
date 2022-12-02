import numpy as np
from keras.layers import Dense, Activation
from keras.models import Sequential
import matplotlib.pyplot as plt
import math
fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)


x = np.linspace(0, math.pi*2, 100)


y = np.sin(x)
model = Sequential([Dense(16, input_shape=(1,)), Activation('tanh'), Dense(3),
                    Activation('tanh'),Dense(1)])

model.compile(loss='mean_squared_error', optimizer='SGD', metrics=['mean_squared_error'])

model.fit(x, y, epochs=1000, batch_size=len(x), verbose=0)
predictions = model.predict(x)

ax.plot(predictions, color='tab:red')
plt.show()