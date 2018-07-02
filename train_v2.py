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

def main(args):
    """ Loading Data """
    train_data, val_data, test_data, image_ids, topic_set = load_data(args.data_dir)
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
    vocabs = load_vocab(args.data_dir, min_occurrences=args.min_occurrences)
    data_loaders = {
        x: get_loader(data[x], args.batch_size, vocabs, args.data_dir, transform[x], max_size=args.max_size) for x in ['train', 'val']
    }

    """ Defining Model and Training Variables """
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    model = Model(512, 196, 512, 512, len(vocabs['word_vocab']), len(vocabs['topic_vocab']))
    if args.start_epoch > 0:
        path = os.path.join(args.output_dir, "checkpoint_{}.pt".format(args.start_epoch))
        checkpoint = torch.load(path) if args.use_cuda else torch.load(path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
    model.to(device)
    optimizer = optim.Adam(model.parameters(), args.lr)
    if args.start_epoch > 0:
        optimizer.load_state_dict(checkpoint['optimizer'])
        del checkpoint
        with open(os.path.join(args.output_dir, "logs.pkl")) as f:
            old_logs = pickle.load(f)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=1, verbose=True)

    """ Training the Model """
    logs, model_data = trainer.train_model(
        model=model, 
        criterion=nn.NLLLoss(), 
        optimizer=optimizer, 
        scheduler=scheduler,
        data_loaders=data_loaders,
        device=device,
        num_epochs=args.num_epochs,
        log_interval=args.log_interval
    )

    """ Updating Logs """
    if args.start_epoch > 0:
        for phase in ['train', 'val']:
            for score in ['loss', 'accuracy']:
                old_logs[phase][score].update(logs[phase][score])
        logs = old_logs

    """ Saving the Best Models """
    len_train_loader = len(data_loaders['train'])
    best_loss = logs['val'][0][len_train_loader]
    for epoch, loss in sorted(logs['val'].items(), key=lambda x: x[1]):
        if loss < best_loss:
            best_loss = loss
            torch.save(model_data[epoch], os.path.join(args.output_dir, "checkpoint_{}.pt".join(epoch)))

    with open(os.path.join(args.output_dir, "logs.pkl"), "rb") as f:
            pickle.dump(logs, f)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='data',
                        help='Path of the data directory. Default value of data')
    parser.add_argument('--start_epoch', type=int,
                        default=0,
                        help='Epoch to start training from. Set to a value greater than 0 to load a model. Default value of 0')
    parser.add_argument('--num_epochs', type=int,
                        default=10,
                        help='Number of epochs to train for. Default value of 10')
    parser.add_argument('--min_occurrences', type=int,
                        default=5,
                        help='The minimum number of times a word must appear in the train data to be included \
                            in the vocabulary. Default value of 5')
    parser.add_argument('--max_size', type=int,
                        default=None,
                        help='The maximum size of the vocabulary. If is None, then no max size. Default value of None')
    parser.add_argument('--batch_size', type=int,
                        default=1,
                        help='Size of a minibatch. Default value of 1')
    parser.add_argument('--lr', type=float,
                        default=0.0001,
                        help='Learning rate to train with. Default value of 0.0001')
    parser.add_argument('--output_dir', type=str,
                        required=True,
                        help='Directory to output logs and model data. Required')
    parser.add_argument('--log_interval', type=float,
                        default=100,
                        help='How often to log training results Default value of 100')
    parser.add_argument('--disable_cuda', action='store_true',
                        default=False,
                        help='Set to disable cuda.')
    args = parser.parse_args()
    args.use_cuda = not args.disable_cuda
    main(args)
