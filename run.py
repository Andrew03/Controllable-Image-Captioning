def run(model, data, vocabs, device, beam_size=10):
    model.eval()
    captions = data['captions']
    targets = captions.narrow(1, 1, captions.size(1) - 2)
    images = data['images'].to(device)
    topics = data['topics'].to(device)
    outputs = model.sample(images, topics, beam_size=beam_size)
    print("topic: {}".format(vocabs['topic_vocab'](topics[0].item())))
    print("OUTPUTS:")
    for i in range(10):
        print(" ".join([vocabs['word_vocab'](x.item()) for x in outputs[i][1]][:-1]))
    print("TARGETS:")
    print(" ".join([vocabs['word_vocab'](x.item()) for x in targets[0]]))

def main(args):
    import os.path
    import torch
    import torchvision.transforms as transforms
    from lib.models.model import Model
    from lib.utils.data_loader import get_loader
    from lib.utils.process_data import load_data
    from lib.utils.vocabulary import load_vocab

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
        x: get_loader(data[x], 1, vocabs, args.data_dir, transform[x], max_size=100) for x in ['train', 'val']
    }

    """ Defining Model and Training Variables """
    device = torch.device("cuda")
    model = Model(512, 196, 512, 512, len(vocabs['word_vocab']), len(vocabs['topic_vocab']), num_layers=args.num_layers, tanh_after=args.tanh_after, is_normalized=args.is_normalized)
    checkpoint = torch.load(os.path.join(args.output_dir, "checkpoint_{}.pt".format(args.checkpoint)))
    model.load_state_dict(checkpoint['state_dict'])
    model.to(device)

    data =  next(iter(data_loaders['val']))

    run(model, data, vocabs, device, args.beam_size)

if __name__ == "__main__":
    import argparse 
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='data',
                        help='Path of the data directory. Default value of data')
    parser.add_argument('--checkpoint', type=int,
                        required=True,
                        help='The checkpoint number to load. Required')
    parser.add_argument('--min_occurrences', type=int,
                        default=5,
                        help='The minimum number of times a word must appear in the train data to be included \
                            in the vocabulary. Default value of 5')
    parser.add_argument('--output_dir', type=str,
                        required=True,
                        help='Directory to output logs and model data. Required')
    parser.add_argument('--disable_cuda', action='store_true',
                        default=False,
                        help='Set to disable cuda.')
    parser.add_argument('--tanh_after', action='store_true',
                        default=False,
                        help='Set to use tanh after.')
    parser.add_argument('--is_normalized', action='store_true',
                        default=False,
                        help='Set to disable normalize encoder.')
    parser.add_argument('--num_layers', type=int,
                        default=1,
                        help='Number of layers in decoder. Default value of 1')
    parser.add_argument('--beam_size', type=int,
                        default=10,
                        help='Beam size to use in generation. Default value of 10')
    args = parser.parse_args()
    args.use_cuda = not args.disable_cuda
    main(args)
