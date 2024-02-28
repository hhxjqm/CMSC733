import os
import numpy as np
import cv2

def generate_data_for_image(image_path, image_idx, rho, patch_size):
    image = cv2.imread(image_path)
    height, width, _ = image.shape

    x_min, y_min = rho, rho
    x_max, y_max = width - patch_size[1] - rho, height - patch_size[0] - rho

    if x_min >= x_max or y_min >= y_max:
        print(image_path)
        rho = 1
        x_min, y_min = rho, rho
        x_max, y_max = width - patch_size[1] - rho, height - patch_size[0] - rho
        
    x, y = np.random.randint(x_min, x_max), np.random.randint(y_min, y_max)

    CA = np.float32([[x, y], [x + patch_size[1], y], [x + patch_size[1], y + patch_size[0]], [x, y + patch_size[0]]])

    perturbations = np.random.randint(-rho, rho, size=(4, 2)).astype(np.float32)
    CB = CA + perturbations

    PA = image[y:y+patch_size[0], x:x+patch_size[1]]
    H_AB = cv2.getPerspectiveTransform(CA, CB)
    PB = cv2.warpPerspective(image, H_AB, (width, height))[y:y+patch_size[0], x:x+patch_size[1]]

    H4Pt = CB - CA

    cv2.imwrite(os.path.join('./data/PA', f'PA_{image_idx}.jpg'), PA)
    cv2.imwrite(os.path.join('./data/PB', f'PB_{image_idx}.jpg'), PB)
    np.savetxt(os.path.join('./data/H4Pt', f'H4Pt_{image_idx}.csv'), H4Pt, delimiter=',')
    np.savetxt(os.path.join('./data/CA', f'CA_{image_idx}.csv'), CA, delimiter=',')

os.makedirs('./data/CA', exist_ok=True)
os.makedirs('./data/H4Pt', exist_ok=True)
os.makedirs('./data/PA', exist_ok=True)
os.makedirs('./data/PB', exist_ok=True)

rho = 16
patch_size = (128, 128)

print("small images!!!: ")
for i in range(1, 5001):
    image_filename = f'{i}.jpg'
    image_path = os.path.join('../Data/Train', image_filename)
    generate_data_for_image(image_path, i, rho, patch_size)

print("Data generation complete.")
