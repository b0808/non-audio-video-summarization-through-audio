import shutil
import numpy as np
import cv2
import os
import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import config


def video_to_frames(video):
    path = os.path.join(config.test_path, 'temporary_images')
    if os.path.exists(path):
        shutil.rmtree(path)
    os.makedirs(path)

    video_path = os.path.join(r'data/testing_data/video', video)
    count = 0
    image_list = []

    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        cv2.imwrite(os.path.join(config.test_path, 'temporary_images', f'frame{count}.jpg'), frame)
        image_list.append(os.path.join(config.test_path, 'temporary_images', f'frame{count}.jpg'))
        count += 1

    cap.release()
    return image_list


def model_cnn_load():
    model = models.vgg16(pretrained=True)
    model.classifier = torch.nn.Sequential(
        *list(model.classifier.children())[:1]
    ) #outout is (1,4096)
    model.eval()
    return model


def load_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (224, 224))
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    img = transform(img)
    return img


def extract_features(video, model):
    
    print(f'Processing video {video}')

    image_list = video_to_frames(video)
    print(len(image_list))
    # Sample 80 evenly spaced frames
    samples = np.round(np.linspace(0, len(image_list) - 1, 80)).astype(int)
    print(samples.shape)
    image_list = [image_list[sample] for sample in samples]
    
    images = torch.zeros((len(image_list), 3, 224, 224))
    
    for i in range(len(image_list)):
        img = load_image(image_list[i])
        images[i] = img

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    with torch.no_grad():
        features = model(images).squeeze() 
        features = features.cpu().numpy()
    shutil.rmtree(os.path.join(config.test_path, 'temporary_images'))

    return features


def extract_feats_pretrained_cnn():
    model = model_cnn_load()
    print('Model loaded')

    if not os.path.isdir(os.path.join(config.test_path, 'feat')):
        os.mkdir(os.path.join(config.test_path, 'feat'))

    video_list = os.listdir("C:/Users/HP/Desktop/one/T/Video-Captioning/data/testing_data/video")

    for video in video_list:
        outfile = os.path.join(config.test_path, 'feat', f'{video}.npy')
        img_feats = extract_features(video, model)
        np.save(outfile, img_feats)


if __name__ == "__main__":
    extract_feats_pretrained_cnn()
