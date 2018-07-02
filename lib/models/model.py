import torch.nn as nn
from encoder_vgg16 import EncoderVGG16
from decoder import Decoder

class Model(nn.Module):
    def __init__(self, vis_dim, vis_num, embed_dim, hidden_dim, word_vocab_size, topic_vocab_size, num_layers=1, dropout=0.0, tanh_after=True, is_normalized=False):
        super(Model, self).__init__()
        self.encoder = EncoderVGG16(is_normalized)
        self.decoder = Decoder(vis_dim, vis_num, embed_dim, hidden_dim, word_vocab_size, topic_vocab_size, num_layers, dropout, tanh_after)

    def parameters(self):
        return self.decoder.parameters()

    def forward(self, images, topics, captions):
        features = self.encoder(images)
        return self.decoder(features, topics, captions)[0]

    def sample(self, images, topics, beam_size=1, start_token=0, end_token=1):
        features = self.encoder(images)
        return self.decoder.sample(features, topics, beam_size, start_token, end_token)
