import os
import numpy as np
import cv2


image = cv2.imread('../Data/Val/2.jpg')
height, width, _ = image.shape
rho = 16
patch_size = (128, 128)
x_min, y_min = rho, rho
x_max, y_max = width - patch_size[1] - rho, height - patch_size[0] - rho
    
x, y = np.random.randint(x_min, x_max), np.random.randint(y_min, y_max)

CA = np.float32([[x, y], [x + patch_size[1], y], [x + patch_size[1], y + patch_size[0]], [x, y + patch_size[0]]])

perturbations = np.random.randint(-rho, rho, size=(4, 2)).astype(np.float32)
CB = CA + perturbations

PA = image[y:y+patch_size[0], x:x+patch_size[1]]
H_AB = cv2.getPerspectiveTransform(CA, CB)

print("-------------------------")
print(CA)
print("-------------------------")
print(CB)
print("-------------------------")
print(H_AB)



import numpy as np

def dlt_svd(CA, CB):
    if CA.shape != (4, 2) or CB.shape != (4, 2):
        raise ValueError("CA and CB must be 4x2 arrays representing four points.")

    N = 4

    A = np.zeros((2 * N, 9))
    for i in range(N):
        X = CA[i, 0]
        Y = CA[i, 1]
        x = CB[i, 0]
        y = CB[i, 1]
        A[2 * i] = [-X, -Y, -1, 0, 0, 0, x * X, x * Y, x]
        A[2 * i + 1] = [0, 0, 0, -X, -Y, -1, y * X, y * Y, y]

    U, S, Vt = np.linalg.svd(A)
    H = Vt[-1].reshape(3, 3)

    H = H / H[-1, -1]

    return H

print("-------------------------")
print(compute_homography(CA, CB))


def dlt(CA, CB):
    """
    Compute the homography matrix H from point correspondences CA and CB using the DLT algorithm.
    This version assumes H[2, 2] is 1 and avoids SVD for the solution.
    """
    if CA.shape != (4, 2) or CB.shape != (4, 2):
        raise ValueError("Input points CA and CB must be 4x2 matrices.")

    A = np.zeros((8, 9))
    for i in range(4):
        x, y = CB[i]
        X, Y = CA[i]
        A[2*i] = [-X, -Y, -1, 0, 0, 0, x*X, x*Y, x]
        A[2*i + 1] = [0, 0, 0, -X, -Y, -1, y*X, y*Y, y]

    A_tilde = A[:, :8]
    b_tilde = -A[:, 8]

    h_tilde, _, _, _ = np.linalg.lstsq(A_tilde, b_tilde, rcond=None)

    h = np.append(h_tilde, 1)
    H = h.reshape(3, 3)
    return H


print("-------------------------")
print(compute_homography_dlt(CA, CB))