import argparse

def get_corpus_bleu(model, data_loader, vocabs, device):
    import torch
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.cider.cider import Cider
    from pycocoevalcap.rouge.rouge import Rouge

    """Defining Scorers"""
    scorer_bleu = Bleu(4)
    scorer_rouge = Rouge()
    scorer_cider = Cider()

    sequences_ref = {}
    sequences_gen = {}

    bad_words = ['<SOS>', '<EOS>', '<UNK>']
    bad_toks = [vocabs['word_vocab'](i) for i in  bad_words]

    """Generation Loop"""
    for i, data in enumerate(data_loader):
        with torch.no_grad():
            captions = data['captions']
            length = captions.size(1) - 1
            targets = captions.narrow(1, 1, length)
            images = data['images'].to(device)
            topics = data['topics'].to(device)

            predictions = model.sample(images, topics, beam_size=args.beam_size)
            sequences_ref[i] = [" ".join([vocabs['word_vocab'](j.item()) for j in targets[0] if j.item() not in bad_toks])]
            sequences_gen[i] = [" ".join([vocabs['word_vocab'](j.item()) for j in predictions[0][1] if j.item() not in bad_toks])]

    """Getting Scores"""
    bleu_score, bleu_scores = scorer_bleu.compute_score(
        sequences_ref, sequences_gen)
    rouge_score, rouge_scores = scorer_rouge.compute_score(
        sequences_ref, sequences_gen)
    cider_score, cider_scores = scorer_cider.compute_score(
        sequences_ref, sequences_gen)
    scores = {'bleu_score': bleu_score, 'rouge_score': rouge_score, 'cider_score': cider_score} 
    print(scores)
    return scores


def evaluate(args):
    import pickle
    import torch
    import torchvision.transforms as transforms
    import os.path
    from tqdm import tqdm
    from lib.models.model import Model
    from lib.utils.batch_data import batch_data
    from lib.utils.data_loader import get_loader
    from lib.utils.process_data import load_data
    from lib.utils.vocabulary import load_vocab

    """Loading Data"""
    train_data, val_data, test_data, image_ids, topic_set = load_data(args.data_dir)
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225])
    ])
    vocabs = load_vocab(args.data_dir, min_occurrences=args.min_occurrences)
    data_loader = get_loader(test_data, 1, vocabs, args.data_dir, transform, max_size=args.max_size)

    """Defining Logging Structure"""
    logs = {}
    if os.path.exists(os.path.join(args.output_dir, "evaluation_scores.pkl")):
        with open(os.path.join(args.output_dir, "evaluation_scores.pkl"), "bb") as f:
            old_logs = pickle.load(f)

    """Defining Model and Training Variables"""
    device = torch.device("cuda" if torch.cuda.is_available() and args.use_cuda else "cpu")
    model = Model(512, 196, 512, 512, len(vocabs['word_vocab']), len(vocabs['topic_vocab']), num_layers=args.num_layers, dropout=args.dropout, tanh_after=args.tanh_after, is_normalized=args.is_normalized)
    for epoch in range(args.start_epoch, args.end_epoch, args.epoch_step):
        path = os.path.join(args.output_dir, "checkpoint_{}.pt".format(epoch))
        checkpoint = torch.load(path) if args.use_cuda else torch.load(path, map_location=lambda storage, loc: storage)
        model.load_state_dict(checkpoint['state_dict'])
        model.to(device)
        del checkpoint
        progress_bar = tqdm(iterable=data_loader, desc="Evaluate Model {}".format(epoch))
        model.eval()
        log = get_corpus_bleu(model, progress_bar, vocabs, device)
        logs[epoch] = log
    
    if os.path.exists(os.path.join(args.output_dir, "evaluation_scores.pkl")):
        logs.update(old_logs)
    with open(os.path.join(args.output_dir, "evaluation_scores.pkl"), "wb") as f:
        pickle.dump(logs, f)

def main(args):
    evaluate(args)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str,
                        default='data',
                        help='Path of the data directory. Default value of data')
    parser.add_argument('--min_occurrences', type=int,
                        default=5,
                        help='Minimum number of times a word must appear in the training data set to be included  \
                        in the vocabulary. Default value of 5')
    parser.add_argument('--max_size', type=int,
                        default=None,
                        help='Maximum size of data loader. Default value of None (no limit)')
    parser.add_argument('--disable_cuda', action='store_true',
                        default=False,
                        help='Set to disable cuda.')
    parser.add_argument('--num_layers', type=int,
                        default=1,
                        help='Number of layers in decoder. Default value of 1')
    parser.add_argument('--dropout', type=float,
                        default=0.3,
                        help='Dropout value. Default value of 0.3')
    parser.add_argument('--tanh_after', action='store_true',
                        default=False,
                        help='Set to apply tanh after.')
    parser.add_argument('--is_normalized', action='store_true',
                        default=False,
                        help='Set to normalize encoder.')
    parser.add_argument('--start_epoch', type=int,
                        required=True,
                        help='Model epoch to start evaluating from. Required')
    parser.add_argument('--end_epoch', type=int,
                        required=True,
                        help='Model epoch to finish evaluating at. Required')
    parser.add_argument('--epoch_step', type=int,
                        required=True,
                        help='Model epochs to step between during evaluating. Required')
    parser.add_argument('--output_dir', type=str,
                        required=True,
                        help='Path of the output directory. Required')
    parser.add_argument('--beam_size', type=int,
                        default=5,
                        help='Beam size to evaluate with. Default value of 5.')
    args = parser.parse_args()
    args.use_cuda = not args.disable_cuda
    args.end_epoch += 1
    if args.start_epoch == args.end_epoch:
        args.epoch_step = 1
    main(args)
