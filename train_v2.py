import argparse
import os
import pickle
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from lib.models.encoder_vgg16 import EncoderVGG16
from lib.models.decoder import Decoder
from lib.utils.batch_data import batch_data
from lib.utils.data_loader import get_loader
from lib.utils.process_data import load_data
from lib.utils.vocabulary import load_vocab

def main(args):
    """ Loading Data """
    train_data, val_data, test_data, image_ids, topic_set = load_data(args.data_dir)
    data = {'train': train_data, 'val': val_data}
    transforms = {
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
    vocabs = load_vocab(args.data_dir, min_occurrences=args.min_occurrences)
    data_loaders = {
        x: get_loader(data, args.batch_size, vocabs, args.data_dir, transforms[x], max_size=args.max_size) for x in ['train', 'val']
    }

    # Defining models
    print("Creating models...")
    encoder = EncoderVGG16(is_normalized=args.is_normalized)
    decoder = Decoder(512, 196, 512, 512, len(word_vocab), len(topic_vocab), 
                      num_layers=args.num_layers, dropout=args.dropout, tanh_after=args.is_tanh_after)