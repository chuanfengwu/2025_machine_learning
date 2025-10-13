#consult chatgpt
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import re
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.model_selection import train_test_split
from matplotlib.colors import ListedColormap

# -----------------------------
# (1) 讀取 XML 檔
# -----------------------------
xml_file = 'O-A0038-003.xml'
tree = ET.parse(xml_file)
root = tree.getroot()

ns = {'ns': 'urn:cwa:gov:tw:cwacommon:0.1'}
content_elem = root.find('.//ns:Content', ns)
if content_elem is None:
    content_elem = root.find('.//Content')
content_text = content_elem.text.strip()

# -----------------------------
# (2) 數據轉換
# -----------------------------
values = [float(x) for x in re.findall(r'[-+]?\d*\.?\d+(?:[Ee][+-]?\d+)?', content_text)]
n_lon, n_lat = 67, 120
lon_start, lat_start, d = 120.0, 21.88, 0.03
grid = np.array(values).reshape(n_lat, n_lon)

# -----------------------------
# (3) 生成 DataFrame
# -----------------------------
data_points = []
for j in range(n_lat):
    for i in range(n_lon):
        lon = lon_start + i * d
        lat = lat_start + j * d
        val = grid[j, i]
        data_points.append((lon, lat, val))
df = pd.DataFrame(data_points, columns=['lon', 'lat', 'value'])

# -----------------------------
# (4) 建立分類資料集
# -----------------------------
df_classification = df.copy()
df_classification['label'] = df_classification['value'].apply(lambda x: 0 if x == -999 else 1)
df_classification = df_classification[['lon', 'lat', 'label']]

X = df_classification[['lon', 'lat']].values
y = df_classification['label'].values

# -----------------------------
# (5) 資料分割、標準化、特徵展開
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

poly = PolynomialFeatures(degree=2, include_bias=False)
X_train_poly = poly.fit_transform(X_train_scaled)
X_test_poly = poly.transform(X_test_scaled)

# -----------------------------
# (6) QDA
# -----------------------------
def qda_train(X, y):
    classes = np.unique(y)
    phi, means, covs = {}, {}, {}
    for c in classes:
        Xc = X[y == c]
        phi[c] = len(Xc) / len(X)
        means[c] = np.mean(Xc, axis=0)
        covs[c] = np.cov(Xc, rowvar=False)
    return phi, means, covs

def qda_predict_prob(X, phi, means, covs):
    log_p0, log_p1 = [], []
    for x in X:
        def log_likelihood(mu, cov, phi):
            diff = x - mu
            inv_cov = np.linalg.inv(cov)
            det_cov = np.linalg.det(cov)
            return -0.5 * diff.T @ inv_cov @ diff - 0.5 * np.log(det_cov) + np.log(phi)
        lp0 = log_likelihood(means[0], covs[0], phi[0])
        lp1 = log_likelihood(means[1], covs[1], phi[1])
        # log-sum-exp trick
        m = max(lp0, lp1)
        denom = m + np.log(np.exp(lp0 - m) + np.exp(lp1 - m))
        p1 = np.exp(lp1 - denom)
        log_p0.append(lp0)
        log_p1.append(lp1)
    return np.array(log_p1)  # only returning log p1 for simplicity

def qda_predict_label(X, phi, means, covs):
    probs = []
    for x in X:
        def log_likelihood(mu, cov, phi):
            diff = x - mu
            inv_cov = np.linalg.inv(cov)
            det_cov = np.linalg.det(cov)
            return -0.5 * diff.T @ inv_cov @ diff - 0.5 * np.log(det_cov) + np.log(phi)
        lp0 = log_likelihood(means[0], covs[0], phi[0])
        lp1 = log_likelihood(means[1], covs[1], phi[1])
        probs.append(1 if lp1 > lp0 else 0)
    return np.array(probs)

phi, means, covs = qda_train(X_train_poly, y_train)
y_pred = qda_predict_label(X_test_poly, phi, means, covs)
accuracy = np.mean(y_pred == y_test)
print(f"Accuracy: {accuracy:.4f}")

# -----------------------------
# (7) decision boundary（真實經緯度 + 黑線邊界）
# -----------------------------
lon_min, lon_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1
lat_min, lat_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1
h = 0.02
lon_grid, lat_grid = np.meshgrid(np.arange(lon_min, lon_max, h),
                                 np.arange(lat_min, lat_max, h))
X_grid = np.c_[lon_grid.ravel(), lat_grid.ravel()]

# transform to scaled polynomial feature space
X_grid_scaled = scaler.transform(X_grid)
X_grid_poly = poly.transform(X_grid_scaled)

# get predictions for grid points
Z = qda_predict_label(X_grid_poly, priors, means, covs)
Z = Z.reshape(lon_grid.shape)

# 繪圖
plt.figure(figsize=(8, 8))
cmap_light = ListedColormap(['#FFBBBB', '#BBFFBB'])
cmap_bold = ListedColormap(['#FF0000', '#00AA00'])

# 填色背景
plt.contourf(lon_grid, lat_grid, Z, cmap=cmap_light, alpha=0.5)

# ⚫ 黑色 decision boundary
plt.contour(lon_grid, lat_grid, Z, levels=[0.5], colors='black', linewidths=2)

# 測試資料點
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap=cmap_bold, edgecolor='k', s=20)

plt.title(f"QDA Decision Boundary (Real Longitude-Latitude)\nAccuracy = {accuracy:.3f}")
plt.xlabel("Longitude (°E)")
plt.ylabel("Latitude (°N)")
plt.axis('equal')
plt.show()
