import numpy as np
import cv2
import matplotlib.pyplot as plt

mov_ex = np.load('moving.npy')
fix_ex = np.load('fixed.npy')
flow_all = np.load('flow.npy')
warp_ex = np.load('warped.npy')

mov_img = mov_ex[0, ..., 0]
fix_img = fix_ex[0, ..., 0]
flow_pred = flow_all[0, ...]
warped_tf = warp_ex[0, ..., 0]

h, w = flow_pred.shape[:2]

map_x = (np.arange(w)[None, :] + flow_pred[...,1]).astype(np.float32)
map_y = (np.arange(h)[:, None] + flow_pred[...,0]).astype(np.float32)

warped_cv = cv2.remap(
    mov_img,
    map_x, map_y,
    interpolation=cv2.INTER_LINEAR,
    borderMode=cv2.BORDER_REFLECT
)

u = flow_pred[..., 0]
v = flow_pred[..., 1]
step = 8
x = np.arange(0, w, step)
y = np.arange(0, h, step)
X, Y = np.meshgrid(x, y)
U = u[Y, X]
V = v[Y, X]

plt.figure(figsize=(12, 4))

plt.subplot(1, 3, 1)
plt.imshow(mov_img, cmap='gray', origin='lower')
plt.title('Moving')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(fix_img, cmap='gray', origin='lower')
plt.title('Fixed')
plt.axis('off')

plt.subplot(1, 3, 3)
plt.imshow(warped_tf, cmap='gray', origin='lower')
plt.quiver(
    X, Y, U, V,
    angles='xy', scale_units='xy', scale=1,
    color='r', width=0.003, headwidth=3,
    headlength=4, alpha=0.8
)
plt.title('Warped with Flow')
plt.axis('off')

plt.tight_layout()
plt.show()
