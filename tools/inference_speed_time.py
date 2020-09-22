import torch
from torch import nn
from PIL import Image
import time
from torchvision import transforms
# Download an example image from the pytorch website
import urllib

def test_model(model):
    url, filename = ("https://github.com/pytorch/hub/raw/master/dog.jpg", "dog.jpg")
    try: urllib.URLopener().retrieve(url, filename)
    except: urllib.request.urlretrieve(url, filename)

    # sample execution (requires torchvision)
    input_image = Image.open(filename)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)
    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')

    time_list = []
    with torch.no_grad():
        for i in range(100):
            start = time.time()
            output = model(input_batch)
            time_list.append(time.time()-start)
    result = sum(time_list)/100
    return 1/result

shufflenet  = torch.hub.load('pytorch/vision:v0.5.0', 'shufflenet_v2_x1_0', pretrained=True)
shufflenet.eval()

googlenet = torch.hub.load('pytorch/vision:v0.5.0', 'googlenet', pretrained=True)
googlenet.eval()

densenet121 = torch.hub.load('pytorch/vision:v0.5.0', 'densenet121', pretrained=True)
densenet121.eval()

resnet50 = torch.hub.load('pytorch/vision:v0.5.0', 'resnet50', pretrained=True)
resnet50.eval()

print(f'inference time for shufflenet is {test_model(shufflenet)}')
print(f'inference time for googlenet is {test_model(googlenet)}')
print(f'inference time for densenet121 is {test_model(densenet121)}')
print(f'inference time for resnet50 is {test_model(resnet50)}')
