# I consulted chatgpt for how to use the keras to write this code.
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error

# Define Runge function
def runge(x):
    return 1 / (1 + 25 * x**2)

# Generate training and test data set
x_train = np.random.uniform(-1, 1, 200).reshape(-1, 1)
x_test = np.linspace(-1, 1, 100).reshape(-1, 1)

y_train = runge(x_train)
y_test = runge(x_test)

# Neural network
model = Sequential([
    Dense(20, activation='tanh', input_shape=(1,)),
    Dense(10, activation='sigmoid'),
    Dense(20, activation='tanh'),
    Dense(10, activation='sigmoid'),
    Dense(1, activation='linear')
])

model.compile(optimizer=Adam(learning_rate=0.01), loss='mse')

# Train the model
history = model.fit(x_train, y_train,
                    epochs=300,
                    batch_size=39,
                    validation_split=0.3,
                    verbose=0)
# Plot the true function and the neural network prediction together
y_pred = model.predict(x_test)
x = np.arange(-1,1,0.01)
y = runge(x)

plt.plot(x, y, label='True function')
plt.plot(x_test, y_pred, label='NN prediction')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.title('Runge Function Approximation')
plt.show()

# Show the training/validation loss curve
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('Training/Validation Loss')
plt.show()

# Compute and report errors
mse = mean_squared_error(y_test, y_pred)
print("MSE error:", mse)