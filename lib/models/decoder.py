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
        self.num_layers = num_layers

        self.init_h_vis = nn.Linear(vis_dim, hidden_dim, bias=False)
        self.init_c_vis = nn.Linear(vis_dim, hidden_dim, bias=False)
        self.init_h_topic = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.init_c_topic = nn.Linear(embed_dim, hidden_dim, bias=False)
        self.attn_vw = nn.Linear(vis_dim, 1)
        self.attn_hw = nn.Linear(hidden_dim, 1)
        self.attn_tw = nn.Linear(embed_dim, 1)

        self.topic_embed = nn.Embedding(topic_vocab_size, embed_dim)
        self.word_embed = nn.Embedding(word_vocab_size, embed_dim)
        self.lstm = nn.LSTM(vis_dim + embed_dim, hidden_dim, batch_first=True, dropout=dropout, num_layers=num_layers)
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
        return hidden.unsqueeze(0).repeat(self.num_layers, 1, 1), cell.unsqueeze(0).repeat(self.num_layers, 1, 1)

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
        return torch.sum(features * attention_softmax, 1).unsqueeze(1), attention_softmax

    def single_forward(self, features, topic_embeddings, captions, hidden):
        embedding = self.word_embed(captions)
        attention = self._compute_attention(features, topic_embeddings, hidden[0].narrow(0, 0, 1))[0]
        input = self.dropout_layer(torch.cat([attention, embedding], 2))
        print("input device: {}, model device: {}, hidden device: {}".format(input.get_device(), list(self.lstm.parameters())[0].get_device(), hidden[0].get_device()))
        print("input size: {}, hidden size: {}".format(input.size(), hidden[0].size()))
        out, hidden = self.lstm(input, hidden)
        words = self.dropout_layer(self.output(self.dropout_layer(out)))
        word_scores = F.log_softmax(words, dim=2)
        return word_scores, hidden

    def forward(self, features, topics, captions):
        """
        topic: B x 1
        features: B x vis_num x vis_dim
        captions: B x seq_length
        """
        topic_embeddings = self.topic_embed(topics)
        hidden = self._init_hidden(features, topic_embeddings)
        word_space = None
        lengths = captions.size(1)
        average_attention = None
        for i in range(lengths):
            current_caption = captions.narrow(1, i, 1)
            word_scores, hidden = self.single_forward(features, topic_embeddings, current_caption, hidden)
            word_space = torch.cat([word_space, word_scores], 1) if word_space is not None else word_scores
        return word_space, hidden

    def sample(self, features, topics, beam_size=1, start_token=0, end_token=1):
        """Now just running through a fixed number of iterations and sorting after"""
        topic_embeddings = self.topic_embed(topics)
        hidden = self._init_hidden(features, topic_embeddings)
        completed_phrases, completed_scores = [], []
        input_captions = create_predict_input_captions([[start_token] for _ in range(features.size(0))])
        tracked_scores = torch.zeros([features.size(0), 1]).cuda()
        tracked_captions = input_captions

        for i in range(40):
            input_captions = tracked_captions.narrow(1, tracked_captions.size(1) - 1, 1)
            """Removing and storing completed phrases"""
            not_finished_indices = input_captions.squeeze(1).ne(end_token).nonzero()
            finished_indices = input_captions.squeeze(1).eq(end_token).nonzero()
            finished_sequences = tracked_captions.index_select(0, finished_indices.view(-1))
            completed_phrases.extend([[x.item() for x in sequence] for sequence in finished_sequences])
            completed_scores.extend([x.item() for x in tracked_scores.index_select(0, finished_indices.view(-1))])
            input_captions = input_captions.index_select(0, not_finished_indices.view(-1))
            tracked_captions = tracked_captions.index_select(0, not_finished_indices.view(-1))
            tracked_scores = tracked_scores.index_select(0, not_finished_indices.view(-1))
            hidden = (hidden[0].index_select(1, not_finished_indices.view(-1)), hidden[1].index_select(1, not_finished_indices.view(-1)))
            if input_captions.size(0) == 0:
                break

            """Generating next caption through beam search and pruning"""
            word_scores, hidden = self.single_forward(features, topic_embeddings, input_captions, hidden)
            top_scores, top_captions = word_scores.squeeze(1).topk(beam_size, 1) # Squeeze to get rid of seq len dimension
            current_scores = tracked_scores.repeat([1, beam_size]).reshape(-1, 1) + top_scores.reshape(-1, 1)
            current_captions = torch.cat([tracked_captions.repeat([1, beam_size]).reshape(-1, i + 1), top_captions.reshape(-1, 1)], 1)

            tracked_scores, score_indices = current_scores.reshape(-1, 1).topk(beam_size, 0)
            score_indices = score_indices.squeeze(1)
            tracked_captions = current_captions.reshape(-1, i + 2).index_select(0, score_indices)

            current_indices = score_indices / beam_size
            hidden = (hidden[0].index_select(1, current_indices), hidden[1].index_select(1, current_indices))
        return [x[1:] for _, x in sorted(zip(completed_scores, completed_phrases), key=lambda pair: pair[0] / len(pair[1]))][:-beam_size]


    def sample_v1(self, features, topics, beam_size=1, start_token=0, end_token=1):
        topic_embeddings = self.topic_embed(topics)
        hidden = self._init_hidden(features, topic_embeddings)
        completed_phrases = []
        best_phrases = []
        score = 0

        initial_caption = create_predict_input_captions([start_token])
        embedding = self.word_embed(initial_caption)
        attention = self._compute_attention(features, topic_embeddings, hidden[0])[0].squeeze(1)
        input = torch.cat([attention, embedding], 1).unsqueeze(1)
        out, hidden = self.lstm(input, hidden)
        words = self.output(out)
        word_scores = F.log_softmax(words, dim=2)
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
            attention = self._compute_attention(features, topic_embeddings, hidden[0])[0].squeeze(1)
            attention = attention.unsqueeze(1)
            input = torch.cat([attention, embedding], 2)
            out, hidden = self.lstm(input, hidden)
            words = self.output(out)
            word_scores = F.log_softmax(words, dim=2)
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
