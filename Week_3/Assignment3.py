#1.
# I consulted chatgpt for how to use the keras to write this code.
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
import tensorflow as tf

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

# Compute errors of the derivatives
x = tf.constant(x_test, dtype=tf.float32)
with tf.GradientTape() as tape:
  tape.watch(x)
  f_pred = model(x)
f_prime = tape.gradient(f_pred, x)
def runge_prime(x):
    return -50 * x / (1 + 25 * x**2)**2
mse_prime = mean_squared_error(runge_prime(x_test), f_prime.numpy())
print("MSE error of derivatives:", mse_prime)
#
#2.
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam
from sklearn.metrics import mean_squared_error
import tensorflow as tf

# Define Runge function
def runge(x):
    return 1 / (1 + 25 * x**2)
def runge_prime(x):
    return -50 * x / (1 + 25 * x**2)**2

# Generate training and test data set
x_train = np.linspace(-1, 1, 200).reshape(-1, 1)
x_test = np.linspace(-1, 1, 100).reshape(-1, 1)

y1 = runge(x_train)
y2 = runge_prime(x_train)
y_train = np.column_stack([y1, y2])
y_test = np.column_stack([runge(x_test), runge_prime(x_test)])

# Define loss function
def new_mse(y_true, y_pred):
  f_true = y_true[:,0]
  f_pred = y_pred[:,0]
  f_true_prime = y_true[:,1]
  f_pred_prime = y_pred[:,1]
  loss_f = tf.reduce_mean(tf.square(f_true-f_pred))
  loss_f_prime = tf.reduce_mean(tf.square(f_true_prime-f_pred_prime))
  return loss_f + 0.2*loss_f_prime

# Neural network
model = Sequential([
    Dense(500, activation='tanh', input_shape=(1,)),
    Dense(500, activation='tanh'),
    Dense(2, activation='linear')
])

model.compile(optimizer=Adam(learning_rate=0.01), loss= new_mse)

# Train the model
history = model.fit(x_train, y_train,
                    epochs=300,
                    batch_size=32,
                    validation_split=0.2,
                    verbose=0)
# Plot the true function and the neural network prediction together
y_pred = model.predict(x_test)
x = np.arange(-1,1,0.01)
y = np.column_stack([runge(x), runge_prime(x)])
y_prime = runge_prime(x)

plt.plot(x, y[:,0], label='True function')
plt.plot(x_test, y_pred[:,0], label='NN prediction')
plt.xlabel('x')
plt.ylabel('f(x)')
plt.legend()
plt.title('Runge Function Approximation')
plt.show()

plt.plot(x, y[:,1], label='True function')
plt.plot(x_test, y_pred[:,1], label='NN prediction')
plt.xlabel('x')
plt.ylabel('f_prime(x)')
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
