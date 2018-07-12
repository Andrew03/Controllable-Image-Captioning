import torch
import torch.nn as nn
import torch.nn.functional as F

def create_predict_input_captions(captions):
    captions = torch.LongTensor(captions)
    if torch.cuda.is_available():
        return captions.cuda()
    return captions

class Decoder(nn.Module):
    """
    Args:
        vis_dim: The size of each visual feature vector
        vis_num: The number of visual feature vectors
    """

    def __init__(self, vis_dim, vis_num, embed_dim, hidden_dim, word_vocab_size, topic_vocab_size, num_layers=1, dropout=0.0, tanh_after=True):
        super(Decoder, self).__init__()
        self.vis_num = vis_num
        self.tanh_after = tanh_after

        self.init_h_vis = nn.Linear(vis_dim, hidden_dim, bias=False)
        self.init_c_vis = nn.Linear(vis_dim, hidden_dim, bias=False)
        self.init_h_topic = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.init_c_topic = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.attn_vw = nn.Linear(vis_dim, 1)
        self.attn_hw = nn.Linear(hidden_dim, 1)
        self.attn_tw = nn.Linear(embed_dim, 1)

        self.topic_embed = nn.Embedding(topic_vocab_size, embed_dim)
        self.word_embed = nn.Embedding(word_vocab_size, embed_dim)
        self.lstm = nn.LSTM(vis_dim + embed_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, word_vocab_size)
        self.dropout_layer = nn.Dropout(p=dropout)

    def _init_hidden(self, features, topics):
        tanh = nn.Tanh()
        hidden_features = torch.sum(self.init_h_vis(features), 1) / self.vis_num
        hidden_topics = self.init_h_topic(topics)
        cell_features = torch.sum(self.init_c_vis(features), 1) / self.vis_num
        cell_topics = self.init_c_topic(topics)
        hidden = tanh(hidden_features + hidden_topics) if self.tanh_after else tanh(hidden_features) + tanh(hidden_topics)
        cell = tanh(cell_features + cell_topics) if self.tanh_after else tanh(cell_features) + tanh(cell_topics)
        return hidden.unsqueeze(0), cell.unsqueeze(0)

    def _compute_attention(self, features, topics, hidden_state):
        """
        features: B x vis_num x vis_dim
        hidden_state: (1 x B x hidden_size, 1 x B x hidden_size)
        """
        # add in L1 norm (sum up everything and divide everything by sum
        #features = torch.norm(features, 1, 2, )
        # B x vis_num x 1
        att_vw = self.attn_vw(features)
        att_hw = self.attn_hw(hidden_state.transpose(0, 1).repeat(1, self.vis_num, 1))
        att_tw = self.attn_tw(topics).unsqueeze(1).repeat(1, self.vis_num, 1)
        attention = att_vw + att_hw + att_tw
        attention_softmax = F.softmax(attention, dim=1)
        return torch.sum(features * attention_softmax, 1), attention_softmax

    def forward(self, features, topics, captions):
        """
        topic: B x 1
        features: B x vis_num x vis_dim
        captions: B x seq_length
        """
        topic_embeddings = self.topic_embed(topics)
        hidden = self._init_hidden(features, topic_embeddings)
        word_embeddings = self.word_embed(captions)
        word_space = None
        lengths = len(captions[0])
        average_attention = None
        """
        # out, hidden = self.lstm(
        # Try flatten before, and after, and at each step
        """
        # self.lstm.flatten_parameters()
        for i in range(lengths):
            index = torch.LongTensor([i])
            if word_embeddings.is_cuda:
                index = index.to(torch.device("cuda:{}".format(word_embeddings.get_device())))
            embedding = torch.index_select(word_embeddings, 1, index)
            attention, alpha = self._compute_attention(features, topic_embeddings, hidden[0])
            attention = attention.unsqueeze(1)
            average_attention = alpha if average_attention is None else average_attention + alpha
            input = self.dropout_layer(torch.cat([attention, embedding], 2))
            #self.lstm.flatten_parameters()
            out, hidden = self.lstm(input, hidden)
            words = self.output(self.dropout_layer(out))
            word_space = torch.cat([word_space, words], 1) if word_space is not None else words
        #self.lstm.flatten_parameters()
        return F.log_softmax(word_space, dim=2), F.softmax(word_space, dim=2)#, average_attention.tolist()

    def batch_sample(self, features, topics, beam_size=1, start_token=0, end_token=1):
        topic_embeddings = self.topic_embed(topics)
        hidden = self._init_hidden(features, topic_embeddings)
        batch_size = topics.size(0)
        completed_phrases = [[] for _ in range(batch_size)]
        best_phrases = [[] for _ in range(batch_size)]
        scores = []

        initial_caption = create_predict_input_captions([start_token]).repeat(batch_size, 1)
        print(initial_caption.size())
        embedding = self.word_embed(initial_caption).unsqueeze(1)
        attention, _ = self._compute_attention(features, topic_embeddings, hidden[0])
        print(embedding.size())
        print(attention.size())
        input = torch.cat([attention, embedding], 1).unsqueeze(1)
        out, hidden = self.lstm(input, hidden)
        word_scores = F.softmax(words, dim=2)
        print("word score size: {}".format(word_scores.size()))
        top_scores, top_captions = word_scores.topk(beam_size)


    def sample(self, features, topics, beam_size=1, start_token=0, end_token=1):
        topic_embeddings = self.topic_embed(topics)
        hidden = self._init_hidden(features, topic_embeddings)
        completed_phrases = []
        best_phrases = []
        score = 0

        initial_caption = create_predict_input_captions([start_token])
        embedding = self.word_embed(initial_caption)
        attention, _ = self._compute_attention(features, topic_embeddings,
                                               hidden[0])
        input = torch.cat([attention, embedding], 1).unsqueeze(1)
        out, hidden = self.lstm(input, hidden)
        words = self.output(out)
        word_scores = F.softmax(words, dim=2)
        top_scores, top_captions = word_scores.topk(beam_size)
        best_phrases = [[
            top_scores[0][0].data[i], [top_captions[0][0].data[i]]
        ] for i in range(beam_size)]
        next_captions = top_captions.squeeze(0).squeeze(0).unsqueeze(1)
        hidden = (hidden[0].repeat(1, beam_size, 1), hidden[1].repeat(
            1, beam_size, 1))

        for index in range(40):
            best_candidates = []
            embedding = self.word_embed(next_captions)
            attention, _ = self._compute_attention(features, topic_embeddings,
                                                   hidden[0])
            attention = attention.unsqueeze(1)
            input = torch.cat([attention, embedding], 2)
            out, hidden = self.lstm(input, hidden)
            words = self.output(out)
            word_scores = F.softmax(words, dim=2)
            top_scores, top_captions = word_scores.topk(beam_size)
            len_phrases = len(best_phrases[0][1])
            for i in range(len(best_phrases)):
                for j in range(beam_size):
                    best_candidates.extend([[
                        best_phrases[i][0] + top_scores[i][0].data[j],
                        best_phrases[i][1] + [top_captions[i][0].data[j]], i
                    ]])
            top_candidates = sorted(
                best_candidates,
                key=lambda score_caption: score_caption[0])[-beam_size:]
            temp_candidates = []
            for phrase in top_candidates:
                if phrase[1][-1] == end_token:
                    completed_phrases.append(
                        [phrase[0] / len(phrase[1]), phrase[1]])
                else:
                    temp_candidates.append(phrase)
            top_candidates = temp_candidates
            if len(completed_phrases) >= beam_size:
                return sorted(
                    completed_phrases,
                    key=lambda score_caption: score_caption[0],
                    reverse=True)[:beam_size]
            best_phrases = [[phrase[0], phrase[1]]
                            for phrase in top_candidates]
            next_captions = create_predict_input_captions(
                [[phrase[1][-1]] for phrase in top_candidates])
            hidden_0 = (torch.stack([
                hidden[0][0].select(0, phrase[2]) for phrase in top_candidates
            ]).unsqueeze(0))
            hidden_1 = (torch.stack([
                hidden[1][0].select(0, phrase[2]) for phrase in top_candidates
            ]).unsqueeze(0))
            hidden = (hidden_0, hidden_1)
        return sorted(
            completed_phrases,
            key=lambda score_caption: score_caption[0],
            reverse=True)[:beam_size]
