import torch
from PIL import Image
from torchvision import models, transforms

# A pretrained network that recognizes the subject of an image
img = Image.open("/Users/mac/Downloads/egyptian-cats.webp")
# print(dir(models))
alexNet = models.AlexNet()
resNet = models.resnet101(pretrained=True)
# preprocess
preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])
img_t = preprocess(img)
# reshapes this tensor so that it has an additional dimension to represent the batch size.
batch_t = torch.unsqueeze(img_t, 0)
resNet.eval()
out = resNet(batch_t)
with open("/Users/mac/Downloads/research books/imagenet1000_clsidx_to_labels.txt") as f:
    labels = [line.strip() for line in f.readlines()]

# we can use the index which is a 1-D tensor to access the label
_, index = torch.max(out, 1)
percentage = torch.nn.functional.softmax(out, dim=1)[0] * 100
# print(labels[index[0]], percentage[index[0]].item())

# we can find second best etc
_, indices = torch.sort(out, descending=True)
best_five = [(labels[idx], percentage[idx].item()) for idx in indices[0][:5]]
print(best_five)
if __name__ == '__main__':
    print()
