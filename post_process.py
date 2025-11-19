import os
import cv2
import numpy as np
from PIL import Image

save_img_path=os.path.join('C:\\Users\\NWI3\\Desktop', 'tmp', f"normal_0001.png")


img = cv2.imread(save_img_path, cv2.IMREAD_UNCHANGED)[...,::-1]
mask = (img >= 32766) & (img <= 32768)

normal=img/(256*256 -1)
normal[mask] = 0.5
normal=2*normal-1
alpha_mask=np.all(normal==0.,axis=-1)

azim_deg = 303
elev_deg = 35.27

azim = np.radians(azim_deg)
elev = np.radians(elev_deg)
x = np.cos(elev) * np.cos(azim)
y = np.sin(elev)
z = np.cos(elev) * np.sin(azim)

cam_pos = np.array([x, y, z])
cam_dir = -cam_pos / np.linalg.norm(cam_pos)  # Looking toward origin
forward = cam_dir
up_guess = np.array([0, 1, 0])

right = np.cross(up_guess, forward)
right /= np.linalg.norm(right)

true_up = np.cross(forward, right)
true_up /= np.linalg.norm(true_up)

print(right,true_up,forward)
R_world_to_cam = np.stack([right, true_up, -forward], axis=0)
print(R_world_to_cam)
H, W, _ = normal.shape

normal=normal
normal_cam = normal.reshape(-1, 3) @ R_world_to_cam
normal_cam = normal_cam.reshape(H, W, 3)*np.array([-1,1,1])[None,None,:]

img=(normal_cam+1)/2
img=np.uint16(img*(256*256-1))

alpha=1-np.float32(alpha_mask)
alpha=np.uint16(alpha*(256*256-1))
saved_img=np.concatenate([img[:,:,::-1],alpha[:,:,None]],axis=-1)
cv2.imwrite(save_img_path.replace('01','02'), saved_img)

v,c=np.unique(normal_cam.reshape(-1,3), return_counts=True,axis=0)
for a,b in zip(v,c):
    if b>100:
        print(a,b)

v,c=np.unique(normal.reshape(-1,3), return_counts=True,axis=0)
for a,b in zip(v,c):
    if b>100:
        print(a,b)

print(len(np.unique(normal.reshape(-1,3),axis=0)))