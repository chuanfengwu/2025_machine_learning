#consult chatgpt
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import re
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler



# reading xml file
xml_file = 'O-A0038-003.xml'
tree = ET.parse(xml_file)
root = tree.getroot()

# grab <Content>  (namespace)
ns = {'ns': 'urn:cwa:gov:tw:cwacommon:0.1'}
content_elem = root.find('.//ns:Content', ns)
if content_elem is None:
    content_elem = root.find('.//Content')

# content to text
content_text = content_elem.text.strip()


# change to float
values = [float(x) for x in re.findall(r'[-+]?\d*\.?\d+(?:[Ee][+-]?\d+)?', content_text)]

#
n_lon, n_lat = 67, 120
lon_start, lat_start = 120.0, 21.88
d = 0.03

#
grid = np.array(values).reshape(n_lat, n_lon)


#
data_points = []
for j in range(n_lat):
    for i in range(n_lon):
        lon = lon_start + i * d
        lat = lat_start + j * d
        val = grid[j, i]
        data_points.append((lon, lat, val))

df = pd.DataFrame(data_points, columns=['lon', 'lat', 'value'])


# classification data set
df_classification = df.copy()
df_classification['label'] = df_classification['value'].apply(lambda x: 0 if x == -999 else 1)
df_classification = df_classification[['lon', 'lat', 'label']]

# regression data set
df_regression = df[df['value'] != -999][['lon', 'lat', 'value']]

# training  set and test set and validation set
x = df_regression[['lon', 'lat']].values
y = df_regression['value'].values
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)
x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42, shuffle = True)
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_val = scaler.transform(x_val)
x_test = scaler.transform(x_test)

# regression model
model = Sequential([
    Dense(32, activation='relu', input_shape=(2,)),
    Dense(32, activation='relu'),
    Dense(1, activation='linear')
])

model.compile(optimizer=Adam(learning_rate=0.01), loss= 'mse')

# train the model
history = model.fit(x_train, y_train, validation_data=(x_val, y_val),
                    epochs=200,
                    batch_size=32,
                    verbose=0)

# Show the training/validation loss curve
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('MSE Loss')
plt.legend()
plt.title('Training/Validation Loss')
plt.show()

# Compute and report errors
y_pred = model.predict(x_test)
mse = mean_squared_error(y_test, y_pred)
print('MSE error:', mse)


# training  set and test set
x = df_classification[['lon', 'lat']].values
y = df_classification['label'].values
cls_x_train, cls_x_test, cls_y_train, cls_y_test = train_test_split(x, y, test_size=0.2, random_state=42)
cls_x_train, cls_x_val, cls_y_train, cls_y_val = train_test_split(cls_x_train, cls_y_train, test_size=0.2, random_state=42, shuffle = True)
cls_x_train = scaler.fit_transform(cls_x_train)
cls_x_val = scaler.transform(cls_x_val)
cls_x_test = scaler.transform(cls_x_test)

# classification model
model_cls = Sequential([
    Dense(32, activation='relu', input_shape=(2,)),
    Dense(32, activation='relu'),
    Dropout(0.2),
    Dense(32, activation='relu'),
    Dense(1, activation='sigmoid')
])

model_cls.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy')

# train the model
class_weight = {0: 1.5, 1: 1.0}
history = model_cls.fit(cls_x_train, cls_y_train, validation_data=(cls_x_val, cls_y_val),
                    epochs=200,
                    batch_size=32,
                    class_weight=class_weight,
                    verbose=0)

# Show the training/validation loss curve
plt.plot(history.history['loss'], label='Training loss')
plt.plot(history.history['val_loss'], label='Validation loss')
plt.xlabel('Epoch')
plt.ylabel('Binary Crossentropy Loss')
plt.legend()
plt.title('Training/Validation Loss')
plt.show()

# Compute and report errors
cls_y_pred = model_cls.predict(cls_x_test)
loss = log_loss(cls_y_test, cls_y_pred)
print('Log Loss:', loss)
#
# (7) define h 
# -----------------------------
def h_function(lon, lat):
    x = np.array([[lon, lat]])
    x_scaled = scaler.transform(x)
    c_prob = model_cls.predict(x_scaled, verbose=0)[0][0]
    if c_prob >= 0.5:
        temp = model.predict(x_scaled, verbose=0)[0][0]
        return temp
    else:
        return -999.0

#
print("\nExample outputs of h(lon, lat):")
for (lon, lat) in [(120.3, 22.5), (121.2, 23.8), (121.8, 24.5)]:
    x = np.array([[lon, lat]])
    x_scaled = scaler.transform(x)

    # 
    r_pred = model.predict(x_scaled, verbose=0)[0][0]
    # 
    c_pred = model_cls.predict(x_scaled, verbose=0)[0][0]
    # 
    h_val = h_function(lon, lat)

    print(f"R({lon:.2f}, {lat:.2f}) = {r_pred:.2f},  "
          f"C({lon:.2f}, {lat:.2f}) = {c_pred:.2f},  "
          f"h({lon:.2f}, {lat:.2f}) = {h_val:.2f}")


# -----------------------------
# (9) 視覺化 h(lon, lat)
# -----------------------------
# 產生經緯度格點（解析度可調）
lon_vals = np.linspace(df['lon'].min(), df['lon'].max(), 100)
lat_vals = np.linspace(df['lat'].min(), df['lat'].max(), 100)
lon_grid, lat_grid = np.meshgrid(lon_vals, lat_vals)

# 計算格點上的 h 值
grid_points = np.c_[lon_grid.ravel(), lat_grid.ravel()]
h_grid = np.array([h_function(lon, lat) for lon, lat in grid_points])
h_grid = h_grid.reshape(lon_grid.shape)

# -----------------------------
# 畫圖
# -----------------------------
plt.figure(figsize=(8,6))
# 將 -999 的缺測區域用灰色顯示
cmap = plt.cm.turbo
cmap.set_under('lightgray')

plt.contourf(lon_grid, lat_grid, h_grid, levels=100, cmap=cmap, vmin=0)
plt.colorbar(label='Predicted value (°C)')
plt.title('Piecewise Smooth Model h(lon, lat)')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

