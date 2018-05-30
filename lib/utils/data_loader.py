import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

class CustomDataSet(Dataset):
    def __init__(self, data, batched_data, word_vocab, topic_vocab, basedir, transform=None):
        self.data = data
        self.batched_data = batched_data
        self.word_vocab = word_vocab
        self.topic_vocab = topic_vocab
        self.basedir = basedir
        self.transform = transform

    def __getitem__(self, index):
        """Returns one data batch (images, topics, captions, caption_lengths, image_ids)."""
        images = []
        topics = []
        captions = []
        img_ids = []
        for (image_id, topic, sentence) in self.batched_data(index):
            image = Image.open("{}/data/images/{}.jpg".format(self.basedir, image_id)).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)
            topics.append(self.topic_vocab(topic))
            img_ids.append(image_id)
            captions.append([self.word_vocab('<SOS>')] +
                            [self.word_vocab(token) for token in sentence] +
                            [self.word_vocab('<EOS>')])

        lengths = [len(caption) for caption in captions]
        return torch.stack(images, 0), torch.LongTensor(topics), torch.LongTensor(captions), lengths, img_ids

    def __len__(self):
        return len(self.batched_data)

def collate_fn(data):
    images, topics, captions, lengths, image_ids = zip(*data)
    return images[0], topics[0], captions[0], lengths[0], image_ids[0]


def _get_loader(data, batched_data, word_vocab, topic_vocab, basedir, transform, shuffle=True, num_workers=1):
    data_set = CustomDataSet(data, batched_data, word_vocab, topic_vocab, basedir, transform)
    return DataLoader(dataset=data_set, 
                      shuffle=shuffle, 
                      num_workers=num_workers, 
                      collate_fn=collate_fn)

def get_loader(data, batch_size, word_vocab, topic_vocab, basedir, transform, progress_bar=False, randomize=True, max_size=None, shuffle=True, num_workers=1):
    from batch_data import batch_data

    batched_data = batch_data(data, batch_size, progress_bar, randomize, max_size)
    return _get_loader(data, batched_data, word_vocab, topic_vocab, basedir, transform, shuffle, num_workers)
