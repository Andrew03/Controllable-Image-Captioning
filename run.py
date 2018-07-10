import argparse 
import os 
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import lib.utils.trainer as trainer
from lib.models.model import Model
from lib.utils.batch_data import batch_data
from lib.utils.data_loader import get_loader
from lib.utils.process_data import load_data
from lib.utils.vocabulary import load_vocab

""" Loading Data """
train_data, val_data, test_data, image_ids, topic_set = load_data("data")
data = {'train': train_data, 'val': val_data}
transform = {
    'train': transforms.Compose([
        transforms.Resize(256),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(), 
        transforms.RandomVerticalFlip(), 
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(), 
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
            std=[0.229, 0.224, 0.225])
    ])
}
vocabs = load_vocab("data", min_occurrences=5)
data_loaders = {
    x: get_loader(data[x], 1, vocabs, "data", transform[x], max_size=100) for x in ['train', 'val']
}

""" Defining Model and Training Variables """
device = torch.device("cuda")
model = Model(512, 196, 512, 512, len(vocabs['word_vocab']), len(vocabs['topic_vocab']), num_layers=2, dropout=0.3, tanh_after=False, is_normalized=False)
checkpoint = torch.load("tanh_after_batch_50_dropout_0.3_num_layers_2_lr_0.0001/checkpoint_10.pt")
model.load_state_dict(checkpoint['state_dict'])
model.to(device)

model.eval()
data =  next(iter(data_loaders["val"]))
captions = data['captions']
targets = captions.narrow(1, 1, captions.size(1) - 2)
images = data['images'].to(device)
topics = data['topics'].to(device)
outputs = model.sample(images, topics, beam_size=10)
print("topic: {}".format(vocabs['topic_vocab'](topics[0].item())))
print("OUTPUTS:")
for i in range(10):
    print(" ".join([vocabs['word_vocab'](x.item()) for x in outputs[i][1]][:-1]))
print("TARGETS:")
print(" ".join([vocabs['word_vocab'](x.item()) for x in targets[0]]))
