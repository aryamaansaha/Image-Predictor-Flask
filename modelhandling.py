import torch
from torchvision.models.densenet import DenseNet
import torchvision.transforms as transforms
import torchvision.models as models
from PIL import Image
import json
from decimal import Decimal

device = "cuda" if torch.cuda.is_available() else "cpu"

DenseNet = models.densenet121(pretrained=True).to(device)
DenseNet.eval()

# imagenet_classidx = json.load(open("imagenet_classes.json"))

# Text File to Dictionary
with open("imagenet_classes.txt") as f:
    keys, values = [], []
    for line in f:
        keys.append(int(line.split(":")[0].strip()))
        val = line.split(":")[1].replace("'", "")
        val = val.strip().split(",")
        val.pop()
        values.append(",".join(val))
        imagenet_dict = {keys[i]: values[i] for i in range(len(keys))}


def transform_image(filename):
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_image = Image.open(filename).convert("RGB")
    input_tensor = preprocess(input_image).to(device)
    input_batch = input_tensor.unsqueeze(0)  # taking batch_size as 1

    return input_batch


def get_prediction(filename):
    with torch.no_grad():
        img_batch = transform_image(filename)
        outputs = DenseNet(img_batch)
    probabilities = torch.nn.functional.softmax(outputs[0], dim=0)

    top5_prob, top5_catid = torch.topk(probabilities, 5)
    display_pred = []

    for i in range(top5_prob.size(0)):

        display_pred.append(
            f"Prediction : {imagenet_dict[top5_catid[i].item()]} ~  with probability {float(round(Decimal(str(top5_prob[i].item())), ndigits=4))*100}%"
        )
    return display_pred


# print(get_prediction("./images/cat2.jpeg")) # Debug purposes
