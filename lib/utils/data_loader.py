import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image

"""Make this custom split"""
class CustomDataSet(Dataset):
    def __init__(self, data, batched_data, vocabs, data_dir, transform, num_partitions=1):
        self.data = data
        self.batched_data = batched_data
        self.vocabs = vocabs
        self.data_dir = data_dir
        self.transform = transform
        self.num_partitions = num_partitions
        self.current_partition = 0
        self.partition_size = len(batched_data) / self.num_partitions

    def select(self, partition):
        if partition < self.num_partitions:
            self.current_partition = partition

    def __getitem__(self, index):
        """Returns one data batch (images, topics, captions, caption_lengths, image_ids)."""
        images = []
        topics = []
        captions = []
        img_ids = []
        for (image_id, topic, sentence) in self.batched_data(index + self.current_partition * self.partition_size):
            image = Image.open("{}/images/{}.jpg".format(self.data_dir, image_id)).convert("RGB")
            image = self.transform(image)
            images.append(image)
            topics.append(self.vocabs['topic_vocab'](topic))
            img_ids.append(image_id)
            captions.append([self.vocabs['word_vocab']('<SOS>')] +
                            [self.vocabs['word_vocab'](token) for token in sentence] +
                            [self.vocabs['word_vocab']('<EOS>')])

        lengths = [len(caption) for caption in captions]
        images_tensor = torch.stack(images, 0)
        topics_tensor = torch.LongTensor(topics)
        captions_tensor = torch.LongTensor(captions)
        images_tensor.requires_grad_(True)
        topics_tensor.requires_grad_(True)
        captions_tensor.requires_grad_(True)
        return images_tensor, topics_tensor, captions_tensor, lengths, img_ids

    def __len__(self):
        return self.partition_size

def collate_fn(data):
    images, topics, captions, lengths, image_ids = zip(*data)
    return {'images': images[0], 'topics': topics[0], 'captions': captions[0], 'lengths': lengths[0], 'image_ids': image_ids[0]}

def get_loader(data, batch_size, vocabs, data_dir, transform, progress_bar=False, randomize=True, max_size=None, shuffle=True, num_workers=0):
    from batch_data import batch_data

    batched_data = batch_data(data, batch_size, progress_bar, randomize, max_size)
    data_set = CustomDataSet(data, batched_data, vocabs, data_dir, transform)
    return DataLoader(dataset=data_set, 
                      shuffle=shuffle, 
                      num_workers=num_workers, 
                      collate_fn=collate_fn)

def get_split_data_set(data, batch_size, vocabs, data_dir, transform, num_partitions, progress_bar=False, randomize=True, max_size=None):
    from torchnet.dataset import SplitDataset
    from batch_data import batch_data

    batched_data = batch_data(data, batch_size, progress_bar, randomize, max_size)
    data_set = CustomDataSet(data, batched_data, vocabs, data_dir, transform, num_partitions)
    return data_set
