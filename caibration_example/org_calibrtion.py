import cv2, mmcv, glob, os
import numpy as np
import os.path as osp
from tqdm import tqdm
from datetime import datetime
import time


if __name__ == '__main__':
    DIM= (640, 480)
    K=np.array([[480,0,320],[0,480,240],[0,0,1]],dtype=np.float32)
    D=np.array([[0],[0],[0],[0]],dtype=np.float32)
    print(DIM)
    print(K)
    print(D)
    objPoints=np.array([[0.7,0,0],[0.7,0.3,0],[1,0,0],[1,0.3,0],[1,-0.3,0],[1.3,-0.3,0],[1.3,0,0]],dtype=np.float32)
    picPoints=np.array([[317,424],[74,423],[314,384],[154,381],[473,384],[431,363],[313,361]],dtype=np.float32)
    t0=time.time()
    retval, rvec, tvec = cv2.solvePnP(objPoints, picPoints, K, D)
    print(retval)
    dt1=time.time()-t0
    objPoints_test = np.array([[0.67, -0.11, 0]],dtype=np.float32)
    proj_pic_pts, jac = cv2.projectPoints(objPoints_test, rvec, tvec, K, D)
    #print(proj_pic_pts)
    dt2 = time.time() - t0
    print(dt1,dt2)
    img=cv2.imread('test.jpg')
    mmcv.imshow(img)


    rows = 500
    columns = 300
    square_size = 0.01 # 1cm

    objp = np.zeros((rows*columns, 3), np.float32)
    idx = 0
    for r in range(0, rows, 1):
        # for c in range(int(columns*0.25), columns, 1):
        for c in range(0, columns, 1):
            objp[idx, :] = [r, c-columns/2, 0] # x, y, z
            idx += 1
    objp *= square_size
    all_pic_pts, jac = cv2.projectPoints(objp, rvec, tvec, K, D)
