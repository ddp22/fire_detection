import cv2, os, argparse, random
import numpy as np

def init_parameter():   
    parser = argparse.ArgumentParser(description='Test')
    parser.add_argument("--videos", type=str, default='foo_videos/', help="Dataset folder")
    parser.add_argument("--results", type=str, default='foo_results/', help="Results folder")
    args = parser.parse_args()
    return args

args = init_parameter()

# Here you should initialize your method
import torch
from PIL import Image
from torchvision import transforms
model = torch.hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=True)
model.eval()

preprocess = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# For all the test videos
for video in os.listdir(args.videos):
    # Process the video
    ret = True
    cap = cv2.VideoCapture(video)
    i = 0
    pos_neg = 0
    while ret:
        ret, img = cap.read()
        i+=1
        # Here you should add your code for applying your method
        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model

        # move the input and model to GPU for speed if available
        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')

        with torch.no_grad():
            output = model(input_batch)

        if output == 1:
            pos_neg = 1
            break
        
        ########################################################
    cap.release()
    f = open(args.results+video+".txt", "w")
    # Here you should add your code for writing the results
    if pos_neg:
        t = i
        f.write(str(t))
    ########################################################
    f.close()