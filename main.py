import numpy as np
from numpy import cos, sin

def get_data(path):
    with open(path, "r") as f:
        raw_data = f.read()
        # print(raw_data)
    raw_data = raw_data.split("\n")
    data = []
    for i in range(1, len(raw_data)):
        tmp = [np.float64(it) for it in raw_data[i].split()]
        data.extend([tmp])
    return np.array(data)[:, 1:]


data_path = "data.txt"
save_path = "result.txt"

m = 4_0000  # 比例尺 1/m
f = 0.15324  # 主距
x0, y0 = 0, 0
phi, omega, kappa = 0, 0, 0  # 俯仰角、横摇角和偏航角

data = get_data(data_path)  # 像坐标和地面坐标数据
n_points = data.shape[0]  # 控制点的个数
img_coord = data[:, :2] / 1000  # 像点坐标，除以 1000 将单位换算为 m
ground_coord = data[:, 2:]  # 地面坐标


Xs = np.mean(ground_coord[:, 0])
Ys = np.mean(ground_coord[:, 1])
Zs = m * f
# print(f"外方位元素初始值：{Xs,Ys,Zs,phi,omega,kappa}")

R = np.zeros((3, 3))  # 旋转矩阵
x = y = np.zeros((n_points, 1))

# V = AX - L
V = L = np.zeros((2 * n_points, 1))
A = np.zeros((2 * n_points, 6))
X = np.zeros((6, 1))

n_iter, MAX_ITER = 0, 150
while True:
    a1 = R[0, 0] = cos(phi) * cos(kappa) - sin(phi) * sin(omega) * sin(kappa)
    a2 = R[0, 1] = (-1.0) * cos(phi) * sin(kappa) - sin(phi) * sin(omega) * cos(kappa)
    a3 = R[0, 2] = (-1.0) * sin(phi) * cos(omega)
    b1 = R[1, 0] = cos(omega) * sin(kappa)
    b2 = R[1, 1] = cos(omega) * cos(kappa)
    b3 = R[1, 2] = (-1.0) * sin(omega)
    c1 = R[2, 0] = sin(phi) * cos(kappa) + cos(phi) * sin(omega) * sin(kappa)
    c2 = R[2, 1] = (-1.0) * sin(phi) * sin(kappa) + cos(phi) * sin(omega) * cos(kappa)
    c3 = R[2, 2] = cos(phi) * cos(omega)

    X_bar = a1 * (ground_coord[:, 0] - Xs) + b1 * (ground_coord[:, 1] - Ys) + c1 * (ground_coord[:, 2] - Zs)
    Y_bar = a2 * (ground_coord[:, 0] - Xs) + b2 * (ground_coord[:, 1] - Ys) + c2 * (ground_coord[:, 2] - Zs)
    Z_bar = a3 * (ground_coord[:, 0] - Xs) + b3 * (ground_coord[:, 1] - Ys) + c3 * (ground_coord[:, 2] - Zs)

    x = (-1.0) * f * X_bar / Z_bar
    y = (-1.0) * f * Y_bar / Z_bar

    L = (img_coord - np.array((x, y)).T).flatten()
    H = Zs - ground_coord[:, 2]

    A[::2, 0] = (-1.0) * f / H
    A[::2, 1] = 0
    A[::2, 2] = (-1.0) * x / H
    A[::2, 3] = (-1.0) * f * (1 + x**2 / f**2)
    A[::2, 4] = (-1.0) * x * y / f
    A[::2, 5] = y

    A[1::2, 0] = 0
    A[1::2, 1] = (-1.0) * f / H
    A[1::2, 2] = (-1.0) * y / H
    A[1::2, 3] = (-1.0) * x * y / f
    A[1::2, 4] = (-1.0) * f * (1 + y**2 / f**2)
    A[1::2, 5] = (-1.0) * x

    ATA_inv = np.linalg.inv(A.T @ A)
    X = ATA_inv @ A.T @ L

    Xs += X[0]
    Ys += X[1]
    Zs += X[2]
    phi += X[3]
    omega += X[4]
    kappa += X[5]

    n_iter += 1
    if (X[3] < 1e-6 and X[4] < 1e-6 and X[5] < 1e-6) or (n_iter > MAX_ITER):
        break

# print("迭代次数:%0.f\n" % n_iter)

R[0, 0] = cos(phi) * cos(kappa) - sin(phi) * sin(omega) * sin(kappa)
R[0, 1] = (-1.0) * cos(phi) * sin(kappa) - sin(phi) * sin(omega) * cos(kappa)
R[0, 2] = (-1.0) * sin(phi) * cos(omega)
R[1, 0] = cos(omega) * sin(kappa)
R[1, 1] = cos(omega) * cos(kappa)
R[1, 2] = (-1.0) * sin(omega)
R[2, 0] = sin(phi) * cos(kappa) + cos(phi) * sin(omega) * sin(kappa)
R[2, 1] = (-1.0) * sin(phi) * sin(kappa) + cos(phi) * sin(omega) * cos(kappa)
R[2, 2] = cos(phi) * cos(omega)

# 精度计算
V = A @ X - L
VV = np.sum(np.linalg.norm(V, ord=2) ** 2)
m0 = np.sqrt(VV / (2 * n_points - 6))
Qx = ATA_inv.diagonal()
m = m0 * np.sqrt(Qx)

with open(save_path, "w", encoding="utf-8") as f:
    f.writelines("外方位元素及其精度： \n")
    f.writelines(f"Xs:{Xs:.6f}  m:±{m[0]:.6f}\n")
    f.writelines(f"Ys:{Ys:.6f}  m:±{m[1]:.6f}\n")
    f.writelines(f"Zs:{Zs:.6f}  m:±{m[2]:.6f}\n")
    f.writelines(f"phi:{phi:.6f}  m:±{m[3]:.6f}\n")
    f.writelines(f"omega:{omega:.6f}  m:±{m[4]:.6f}\n")
    f.writelines(f"kappa:{kappa:.6f}  m:±{m[5]:.6f}\n")
    f.writelines("\n旋转矩阵：\n")
    f.write(f"{R}")

