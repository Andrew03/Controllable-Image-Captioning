import torchvision.transforms as transforms
import lib.utils.process_data as process_data
import lib.utils.vocabulary as vocabulary
import lib.utils.data_loader as data_loader
import torchnet as tnt
from lib.utils.batch_data import batch_data

transform = transforms.Compose([ 
    transforms.Resize(256),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(), 
    transforms.ToTensor(), 
    transforms.Normalize(mean=[0.485, 0.456, 0.406], 
      std=[0.229, 0.224, 0.225])
])

train_data, dev_data, test_data, image_ids, topic_set = process_data.load_data(".")
word_vocab = vocabulary.load_vocab(".", is_word_vocab=True)
topic_vocab = vocabulary.load_vocab(".", is_word_vocab=False)
split_data = data_loader.get_split_data_set(train_data, 32, word_vocab, topic_vocab, ".", transform, 3)
