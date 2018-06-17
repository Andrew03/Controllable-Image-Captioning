import argparse
import os
import torch
import torch.distributed as dist
from torch.multiprocessing import Process
from lib.utils.data_loader import get_split_data_set

def run(rank, size, train_data_set, dev_data_set, train_loss, dev_loss, load_checkpoint, 
        lr, num_epochs):
    """have 3 different output sets, one for each of them"""
    """display just the first one temporarily though"""
    train_set, bsz = partition_dataset()
    encoder = EncoderVGG16(is_normalized=is_normalized)
    decoder = Decoder(512, 196, 512, 512, len(word_vocab), len(topic_vocab),
                      num_layers=num_layers, dropout=dropout, tanh_after=is_tanh_after)
    optimizer = optim.SGD(decoder.parameters(), lr=lr)
    if args.load_checkpoint is not None:
        checkpoint = torch.load(args.load_checkpoint, map_location='cpu')
        if rank == 0:
            print("Loading from checkpoint {}".format(args.load_checkpoint))
        start_epoch = checkpoint['epoch']
        decoder.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict([checkpoint['optimizer']])
        del checkpoint
        torch.cuda.empty_cache()
    else:
        start_epoch = 0

    """ Enable Cuda Here"""

    # Defining transformations
    # Should make a function to do this
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

    num_batches = ceil(len(train_set.dataset) / float(bsz))
    for epoch in range(10):
        epoch_loss = 0.0
        for data, target in train_set:
            optimizer.zero_grad()
            output = model(data)
            loss = F.nll_loss(output, target)
            epoch_loss += loss.item()
            loss.backward()
            average_gradients(model)
            optimizer.step()
        print('Rank ', dist.get_rank(), ', epoch ',
              epoch, ': ', epoch_loss / num_batches)

    end_epoch = args.num_epochs + start_epoch

""" Gradient averaging. """
def average_gradients(decoder):
    size = float(dist.get_world_size())
    for param in decoder.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size

def init_processes(rank, size, fn, backend='tcp'):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size)

def main(args):
    print("Loading data...")
    train_data, dev_data, test_data, image_ids, topic_set = load_data(args.basedir)
    print("Loading vocabulary...")
    word_vocab = load_vocab(args.basedir, is_word_vocab=True, min_occurrences=args.min_occurrences)
    topic_vocab = load_vocab(args.basedir, is_word_vocab=False, min_occurrences=args.min_occurrences)

    train_data_set = get_split_data_set(train_data, args.batch_size, word_vocab, topic_vocab, args.basedir, train_transform, args.num_gpus)
    dev_data_set = get_split_data_set(dev_data, args.batch_size, word_vocab, topic_vocab, args.basedir, dev_transform, args.num_gpus)
    test_data_set = get_split_data_set(test_data, args.batch_size, word_vocab, topic_vocab, args.basedir, test_transform, args.num_gpus)

    train_loss = {}
    dev_loss = {}
    train_loss_file = "{}/train_loss.pkl".format(args.save_dir)
    dev_loss_file = "{}/dev_loss.pkl".format(args.save_dir)
    if os.path.isfile(train_loss_file) and os.path.isfile(dev_loss_file):
        with open(train_loss_file, "rb") as f:
            train_loss = pickle.load(f)
        with open(dev_loss_file, "rb") as f:
            dev_loss = pickle.load(f)

    processes = []
    for rank in range(args.num_gpus):
        p = Process(target=init_processes, args=(rank, args.num_gpus, train_data_set, 
                    dev_data_set, test_data_set, train_loss, dev_loss, args.load_checkpoint,
                    args.lr, args.num_epochs, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()

    with open(train_loss_file, "rb") as f:
        train_loss = pickle.load(f)
    with open(dev_loss_file, "rb") as f:
        dev_loss = pickle.load(f)


if __name__ == "__main__":
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
    parser.add_argument('--num_gpus', type=int,
                        default=3,
                        help='The number of gpus to use. Default value of 3.')
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
