import torchvision.transforms as transforms
from PIL import Image

# 假设您已经有了一个图像的路径
image_path = r'..\Data\Train\500.jpg'

# 加载图像并转换为Tensor
image = Image.open(image_path)
transform = transforms.ToTensor()
tensor_image = transform(image)

# 打印图像的大小
print(tensor_image.size())  # 或者 print(tensor_image.shape)
