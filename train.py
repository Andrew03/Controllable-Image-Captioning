import argparse
import os
import pickle
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.multiprocessing import Process
import torchvision.transforms as transforms
from tqdm import tqdm
from lib.models.encoder_vgg16 import EncoderVGG16
from lib.models.decoder import Decoder
from lib.utils.batch_data import batch_data
from lib.utils.data_loader import get_loader
from lib.utils.process_data import load_data
from lib.utils.trainer import create_data_iter, train, validate
from lib.utils.vocabulary import load_vocab

def main(args):
    # Loading data
    print("Loading data...")
    train_data, dev_data, test_data, image_ids, topic_set = load_data(args.basedir)
    print("Loading vocabulary...")
    word_vocab = load_vocab(args.basedir, is_word_vocab=True, min_occurrences=args.min_occurrences)
    topic_vocab = load_vocab(args.basedir, is_word_vocab=False, min_occurrences=args.min_occurrences)

    # Defining models
    print("Creating models...")
    encoder = EncoderVGG16(is_normalized=args.is_normalized)
    decoder = Decoder(512, 196, 512, 512, len(word_vocab), len(topic_vocab), 
                      num_layers=args.num_layers, dropout=args.dropout, tanh_after=args.is_tanh_after)

    # Defining loss and optimizers
    loss_function = nn.NLLLoss()
    params = list(decoder.parameters())
    optimizer = optim.Adam(params, lr=args.lr)

    # Loading models and optimizers
    if args.load_checkpoint is not None:
        checkpoint = torch.load(args.load_checkpoint, map_location='cpu')
        print("loading from checkpoint {}".format(args.load_checkpoint))
        start_epoch = checkpoint['epoch']
        decoder.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        del checkpoint
        torch.cuda.empty_cache()
    else:
        start_epoch = 0

    # Enabling cuda if specified
    device = torch.device("cuda:{}".format(args.cuda_device) if torch.cuda.is_available() and args.use_cuda else "cpu")
    # check if this does anything or if we need to assign
    if torch.cuda.device_count() > 1 and not args.disable_multi_gpu:
        pass
    encoder.to(device)
    decoder.to(device)

    # Defining and loading output data, if it exists
    train_loss = {}
    val_loss = {}
    train_loss_file = "{}/train_loss.pkl".format(args.save_dir)
    val_loss_file = "{}/val_loss.pkl".format(args.save_dir)
    if os.path.isfile(train_loss_file) and os.path.isfile(val_loss_file):
        with open(train_loss_file, "rb") as f:
            train_loss = pickle.load(f)
        with open(val_loss_file, "rb") as f:
            val_loss = pickle.load(f)


    # Defining transformations
    train_transform = transforms.Compose([ 
	transforms.Resize(256),
	transforms.RandomCrop(224),
	transforms.RandomHorizontalFlip(), 
	transforms.RandomVerticalFlip(), 
	transforms.ToTensor(), 
	transforms.Normalize(mean=[0.485, 0.456, 0.406], 
	    std=[0.229, 0.224, 0.225])
    ])
    val_transform = transforms.Compose([ 
	transforms.Resize(256),
	transforms.CenterCrop(224),
	transforms.ToTensor(), 
	transforms.Normalize(mean=[0.485, 0.456, 0.406], 
	    std=[0.229, 0.224, 0.225])
    ])

    end_epoch = args.num_epochs + start_epoch
    for epoch in range(start_epoch, end_epoch):
        # Train section
        train_data_iter = create_data_iter(train_data, args.batch_size, word_vocab, topic_vocab, args.basedir, 
                                            train_transform, progress_bar=args.progress_bar, description="Train [{}/{}]".format(epoch + 1, end_epoch), max_size=args.max_num_batches)
        train_epoch_loss = train(train_data_iter, encoder, decoder, loss_function, optimizer, args.grad_clip, args.log_interval, args.progress_bar, use_cuda=args.use_cuda, cuda_device=args.cuda_device)
        train_loss.update({key + (epoch * len(train_data_iter)) : value for key, value in train_epoch_loss.items()})
        # Validation section
        val_data_iter = create_data_iter(dev_data, args.batch_size, word_vocab, topic_vocab, args.basedir,
                                          val_transform, progress_bar=args.progress_bar, description="Val [{}/{}]".format(epoch + 1, end_epoch), max_size=args.max_num_batches)
        val_epoch_loss = validate(val_data_iter, encoder, decoder, loss_function, progress_bar=args.progress_bar, use_cuda=args.use_cuda, cuda_device=args.cuda_device)
        val_loss[epoch * len(train_data_iter)] = val_epoch_loss
        # Saving the model
        save_path = "{}/checkpoint_{}.pt".format(args.save_dir, epoch + 1)
        torch.save({'epoch': epoch + 1,
                    'state_dict': decoder.state_dict(),
                    'optimizer': optimizer.state_dict()
                    }, save_path)
        if args.progress_bar:
            tqdm.write("Saved model at {}".format(save_path))
        else:
            print("Saved model at {}".format(save_path))

    with open(train_loss_file, "wb") as f:
        pickle.dump(train_loss, f)
    with open(val_loss_file, "wb") as f:
        pickle.dump(val_loss, f)
    print("Saved train logs and val logs at {} and {}".format(train_loss_file, val_loss_file))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--basedir', type=str,
                        default='.',
                        help='The root directory of the project. Default value of \'.\' (the current directory).')
    parser.add_argument('--min_occurrences', type=int,
                        default=5,
                        help='The minimum number of times a word must appear in the train data to be included \
                              in the vocabulary. Default value of 5.')
    parser.add_argument('--is_normalized', action='store_true',
                        help='Set to use the batch normalized vgg16.')
    parser.add_argument('--num_layers', type=int,
                        default=1,
                        help='The number of layers in the decoder. Default value of 1.')
    parser.add_argument('--dropout', type=float,
                        default=0.0,
                        help='The amount of dropout to apply to the model. Default value of 0.')
    parser.add_argument('--is_tanh_after', action='store_true',
                        help='Set to apply the tanh activation after summing the componenets.')
    parser.add_argument('--lr', type=float,
                        default=0.001,
                        help='The learning rate to apply to the model. Default value of 0.001.')
    parser.add_argument('--grad_clip', type=float,
                        default=5.0,
                        help='The gradient cilp to apply to the model. Defalut value of 5.0.')
    parser.add_argument('--load_checkpoint', type=str,
                        default=None,
                        help='The checkpoint to load the model from. Default value of None.')
    parser.add_argument('--cuda_device', type=int,
                        default=0,
                        help='The cuda device to use. Default value of 0.')
    parser.add_argument('--disable_cuda', action='store_true',
                        help='Set to disable cuda usage.')
    parser.add_argument('--disable_multi_gpu', action='store_true',
                        help='Set to disable multi-gpu usage.')
    parser.add_argument('--model_name', type=str,
                        required=True,
                        help='The name to save the model under. Required.')
    parser.add_argument('--batch_size', type=int,
                        default=16,
                        help='Minibatch size. Default value of 16.')
    parser.add_argument('--max_num_batches', type=int,
                        default=None,
                        help='Maximum number of batches in an epoch. Default value of None.')
    parser.add_argument('--num_epochs', type=int,
                        default=10,
                        help='Number of epochs to train for. Default value of 10.')
    parser.add_argument('--disable_progress_bar', action='store_true',
                        help='Set to disable progress bar output.')
    parser.add_argument('--log_interval', type=int,
                        default=100,
                        help='Number of minibatches to train on before logging. Default value of 100.')

    args = parser.parse_args()
    args.use_cuda = not args.disable_cuda
    args.progress_bar = not args.disable_progress_bar
    if args.lr <= 0.0 or \
        args.min_occurrences <= 0 or \
        args.num_layers <= 0 or \
        (args.disable_cuda and not args.disable_multi_gpu):
        print("Invalid arguments!")
    else:
        args.save_dir = "{}/data/checkpoints/{}".format(args.basedir, args.model_name)
        if not os.path.exists(args.save_dir):
            print("New model, saving at {}".format(args.save_dir))	
            os.makedirs(args.save_dir)
        main(args)
