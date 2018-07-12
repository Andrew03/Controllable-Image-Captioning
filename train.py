import torch.distributed as dist

def run(rank, size, split_data, vocabs, args):
    import os
    import pickle
    import torch
    import torch.optim as optim
    from torch.utils.data import DataLoader
    import lib.utils.trainer as trainer
    from lib.utils.data_loader import get_split_data_set
    from lib.models.model import Model

    for mode, data in split_data.items():
        data.select(rank)
    data_loaders = {
        x: DataLoader(dataset=split_data[x], shuffle=True, num_workers=1, collate_fn=collate_fn) for x in ['train', 'val']
    }
    device = torch.device("cuda:{}".format(rank) if torch.cuda.is_available() and args.use_cuda else "cpu")
    model = Model(512, 196, 512, 512, len(vocabs['word_vocab']), len(vocabs['topic_vocab']), num_layers=args.num_layers, dropout=args.dropout, tanh_after=args.tanh_after, is_normalized=args.is_normalized)
    if args.start_epoch > 0:
        path = os.path.join(args.output_dir, "checkpoint_{}.pt".format(args.start_epoch))
        checkpoint = torch.load(path) if args.use_cuda else torch.load(path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        del checkpoint
    model.to(device)
    optimizer = optim.Adam(model.parameters(), args.lr)
    if args.start_epoch > 0:
        optimizer.load_state_dict(checkpoint['optimizer'])
        del checkpoint
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=2, verbose=True)
    criterion = nn.NLLLoss()

    logs, model_data = trainer.train_model(
        model, 
        criterion, 
        optimizer, 
        scheduler, 
        data_loaders, 
        device, 
        args.output_dir, 
        rank, 
        args.num_gpus,
        average_gradients if size > 1 else no_average, 
        args.start_epoch, 
        args.num_epochs, 
        args.log_interval, 
        args.grad_clip)
    """Updating Logs"""
    if rank == 0 and args.start_epoch > 0:
        with open(os.path.join(args.output_dir, "logs.pkl"), "rb") as f:
            old_logs = pickle.load(f)
        for phase in ['train', 'val']:
            logs[phase].update(old_logs[phase])
        with open(os.path.join(args.output_dir, "logs.pkl"), "wb") as f:
            pickle.dump(logs, f)

""" Gradient averaging. """
def average_gradients(model):
    size = float(dist.get_world_size())
    for param in model.parameters():
        dist.all_reduce(param.grad.data, op=dist.reduce_op.SUM)
        param.grad.data /= size

def no_average(model):
    pass

def init_processes(rank, size, split_data, vocabs, args, fn, backend='gloo'):
    import os

    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = '127.0.0.1'
    os.environ['MASTER_PORT'] = '29500'
    dist.init_process_group(backend, rank=rank, world_size=size)
    fn(rank, size, split_data, vocabs, args)


def main(args):
    import torchvision.transforms as transforms
    from torch.multiprocessing import Process
    from lib.utils.process_data import load_data
    from lib.utils.vocabulary import load_vocab
    from lib.utils.data_loader import get_split_data_set

    """Loading Data"""
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
    while args.batch_size % args.num_gpus != 0:
        args.batch_size += 1
    split_data = {
        x: get_split_data_set(data[x], args.batch_size, vocabs, args.data_dir, transform[x], args.num_gpus, randomize=True, max_size=args.max_size) for x in ['train', 'val']
    }
    if args.num_gpus > 1:
        processes = []
        for rank in range(args.num_gpus):
            p = Process(target=init_processes, args=(rank, args.num_gpus, split_data, vocabs, args, run))
            p.start()
            processes.append(p)

        for p in processes:
            p.join()
    else:
        run(0, 1, split_data, vocabs, args)

if __name__ == "__main__":
    import argparse
    import os

    parser = argparse.ArgumentParser()
    parser.add_argument("--num_gpus", type=int,
                        default=1,
                        help='Number of gpus to use in training. Default value of 1')
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
    parser.add_argument('--tanh_after', action='store_true',
                        default=False,
                        help='Set to use tanh after.')
    parser.add_argument('--is_normalized', action='store_true',
                        default=False,
                        help='Set to disable normalize encoder.')
    parser.add_argument('--grad_clip', type=float,
                        default=5.0,
                        help='Gradient clip value. Default value of 5.0')
    parser.add_argument('--dropout', type=float,
                        default=0.5,
                        help='Dropout value. Default value of 0.5')
    parser.add_argument('--num_layers', type=int,
                        default=1,
                        help='Number of layers in decoder. Default value of 1')
    args = parser.parse_args()
    args.use_cuda = not args.disable_cuda
    if args.start_epoch == 0 and not os.path.isdir(args.output_dir):
        os.makedirs(args.output_dir)
        
    main(args)
