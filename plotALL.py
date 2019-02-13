import numpy as np
import matplotlib.pyplot as plt
import cv2
from mpl_toolkits.mplot3d import Axes3D

import scipy.io as io
from collections import OrderedDict

def flip_3d( msk):
    msk[:, 1] = -msk[:, 1]
    return msk

case = 0

if case==0:
    prediction = './temp/valid_out.npy'
    gt = './temp/valid_gt.npy'
    im = './temp/valid_im.jpg'
if case== 1:
    prediction = './temp/train_out.npy'
    gt = './temp/train_gt.npy'
    im = './temp/train_im.jpg'


points = 1

fig = plt.figure()
ax0 = fig.add_subplot(231)
im = cv2.imread(im)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
plt.imshow(im)

xyz_total = np.load(prediction)

xyz_total = 1000 * xyz_total
children = io.loadmat('./utils/children_b.mat')['b']
art_select = [1,     2,     3,     4,     5,     6,     7,    8,    9,    10,    11,    12,    13,    14,    15,    16, 17]
art_select_ = [0,     1,     2,     3,     6,     7,     8,    12,    13,    15,    17,    18,    19,    25,    26,    27]

ax1 = fig.add_subplot(232, projection='3d')
for i in range(len(art_select)):
    art = art_select[i]-1
    child = children[art]
    child = child[0]

    if child.size:
        child = child[0]
        for j in range(len(child)):
            if child[j] in art_select:
                seg = []
                P2 = xyz_total[art,:]
                P1 = xyz_total[child[j]-1,:]
                ax1.plot([P1[0], P2[0]], [P1[1], P2[1]], zs=[P1[2], P2[2]])

if points:
    ax1.scatter(xyz_total[:,0], xyz_total[:,1], xyz_total[:,2],  s=5, c=None, depthshade=True)
ax1.set_xlabel('X Label')
ax1.set_ylabel('Y Label')
ax1.set_zlabel('Z Label')
ax1.view_init(25, 75)
ax1.set_xlim([-500, 500])
ax1.set_ylim([-500, 500])
ax1.set_zlim([-500, 500])


ax2 = fig.add_subplot(233, projection='3d')

xyz_total_gt = np.load(gt)
xyz_total_gt = 1000 * xyz_total_gt


print('3D error: '+str(np.mean(np.sqrt(np.sum(np.square(xyz_total_gt-xyz_total),axis=1))))+' mm')

for i in range(len(art_select)):
    art = art_select[i] - 1
    child = children[art]
    child = child[0]

    if child.size:
        child = child[0]
        for j in range(len(child)):
            if child[j] in art_select:
                seg = []
                P2 = xyz_total_gt[art, :]
                P1 = xyz_total_gt[child[j] - 1, :]
                ax2.plot([P1[0], P2[0]], [P1[1], P2[1]], zs=[P1[2], P2[2]])
if points:
    ax2.scatter(xyz_total_gt[:, 0], xyz_total_gt[:, 1], xyz_total_gt[:, 2], s=5, c=None, depthshade=True)
ax2.set_xlabel('X Label')
ax2.set_ylabel('Y Label')
ax2.set_zlabel('Z Label')
ax2.view_init(25, 75)
ax2.set_xlim([-500, 500])
ax2.set_ylim([-500, 500])
ax2.set_zlim([-500, 500])


# Plot 2D

if case==0:
    prediction = './temp/valid_out_2d.npy'
    gt = './temp/valid_gt_2d.npy'
    im = './temp/valid_im.jpg'
    recon = './temp/valid_recon.npy'
    org = './temp/vr.npy'
if case== 1:
    prediction = './train_out_2d.npy'
    gt = './temp/train_gt_2d.npy'
    im = './temp/train_im.jpg'
    recon = './temp/train_recon.npy'
    org = '/temp/tr.npy'

points = 0

ax3 = fig.add_subplot(234)
im = cv2.imread(im)
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
cmaps = OrderedDict()
cmaps['Qualitative'] = ['Pastel1', 'Pastel2', 'Paired', 'Accent',
                        'Dark2', 'Set1', 'Set2', 'Set3',
                        'tab10', 'tab20', 'tab20b', 'tab20c']
name =cmaps['Qualitative'][1]

rec = np.load(recon)

#rec = np.sum(rec[0,:,:,:],axis=2)
rec = np.argmax(rec[0,:,:,:],axis=2)
plt.imshow(rec,cmap = 'nipy_spectral')
rec_org = np.load(org)
rec_org = np.argmax(rec_org[0,:,:,:],axis=2)
ax0.imshow(rec_org,cmap = 'nipy_spectral')

xyz_total = np.load(prediction)
xyz_total = 256 * xyz_total


ax4 = fig.add_subplot(235)
ax4.imshow(im)

for i in range(len(art_select)):
    art = art_select[i]-1
    child = children[art]
    child = child[0]

    if child.size:
        child = child[0]
        for j in range(len(child)):
            if child[j] in art_select:
                seg = []
                P2 = xyz_total[art,:]
                P1 = xyz_total[child[j]-1,:]
                ax4.plot([P1[0], P2[0]], [P1[1], P2[1]])
if points:
    ax4.scatter(xyz_total[:,0], xyz_total[:,1], s=5, c=None, depthshade=True)


ax5 = fig.add_subplot(236)

ax5.imshow(im)
xyz_total_gt = np.load(gt)
xyz_total_gt = 256 * xyz_total_gt
print('2D error: '+str(np.mean(np.sqrt(np.sum(np.square(xyz_total_gt-xyz_total),axis=1))))+' px')


for i in range(len(art_select)):
    art = art_select[i] - 1
    child = children[art]
    child = child[0]

    if child.size:
        child = child[0]
        for j in range(len(child)):
            if child[j] in art_select:
                seg = []
                P2 = xyz_total_gt[art, :]
                P1 = xyz_total_gt[child[j] - 1, :]
                ax5.plot([P1[0], P2[0]], [P1[1], P2[1]])
if points:
    ax5.scatter(xyz_total_gt[:, 0], xyz_total_gt[:, 1], s=5, c=None, depthshade=True)

plt.show()