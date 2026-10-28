import torch
import torchvision.transforms
from PIL import Image
from torch import nn
from torch.nn import Sequential

images_path="../imgs/02.jpg"
image=Image.open(images_path)
print(image)
image=image.convert('RGB')
transform=torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),torchvision.transforms.ToTensor()])
image=transform(image)
print(image.shape)
class Beverly(nn.Module):
    def __init__(self):
        super(Beverly, self).__init__()
        self.moudl1 = Sequential(nn.Conv2d(3,32,5,1,2),
                                 nn.MaxPool2d(2),
                                 nn.Conv2d(32,32,5,1,2),
                                 nn.MaxPool2d(2),
                                 nn.Conv2d(32,64,5,1,2),
                                 nn.MaxPool2d(2),
                                 nn.Flatten(),
                                 nn.Linear(64*4*4,64),
                                 nn.Linear(64,10)
        )

    def forward(self,x):
        x=self.moudl1(x)
        return x

model=torch.load("Beverly_GPU_29.pth",map_location=torch.device("cpu"),weights_only=False)
print(model)
image=torch.reshape(image,(1,3,32,32))
model.eval()
with torch.no_grad():
    output=model(image)
print(output)
print(output.argmax(1))