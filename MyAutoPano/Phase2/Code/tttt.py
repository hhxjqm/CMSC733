import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import cv2

image_path = r'..\Data\Train\2.jpg'

image = Image.open(image_path)
transform = transforms.ToTensor()
tensor_image = transform(image)

print(tensor_image.size())

def apply_homography_and_crop_patch(I1, max_warp=32):
    h, w = I1.shape[:2]
    perturbations = np.float32([
        [np.random.uniform(-max_warp, max_warp), np.random.uniform(-max_warp, max_warp)],
        [np.random.uniform(-max_warp, max_warp), np.random.uniform(-max_warp, max_warp)],
        [np.random.uniform(-max_warp, max_warp), np.random.uniform(-max_warp, max_warp)],
        [np.random.uniform(-max_warp, max_warp), np.random.uniform(-max_warp, max_warp)],
    ])
    
    src_points = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst_points = src_points + perturbations
    
    H = cv2.getPerspectiveTransform(src_points, dst_points)
    
    I1_warped = cv2.warpPerspective(I1, H, (w, h))
    
    return I1_warped, H

I1 = cv2.imread(image_path).astype(np.float32)
I1_warped, H = apply_homography_and_crop_patch(I1)


print("Homography Matrix H:\n", H)

output_image_path = 'warped_1.jpg'
cv2.imwrite(output_image_path, I1_warped)
