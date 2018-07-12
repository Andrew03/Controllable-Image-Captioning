class BatchedData(object):
    def __init__(self, batch_size):
        self.batched_data = []
        self.index = 0
        self.batch_size = batch_size

    def add_batch(self, batch):
        if len(batch) == self.batch_size:
            self.batched_data.append(batch)
        else:
            print("not the correct size batch!")

    def __call__(self, index):
        if not index < len(self.batched_data):
            return []
        return self.batched_data[index]

    def __len__(self):
        return len(self.batched_data)

def batch_data(data, batch_size, progress_bar=True, randomize=True, max_size=None):
    import random
    from lib.utils.detect_notebook import is_notebook()
    if is_notebook():
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm

    batched_set = {}
    for (image_id, topic, sentence) in (tqdm(data) if progress_bar else data):
        caption_len = len(sentence)
        if caption_len not in batched_set.keys():
            batched_set[caption_len] = []
        batched_set[caption_len].append((image_id, topic, sentence))

    batched_data = BatchedData(batch_size)

    curr_size = 0
    keys = batched_set.keys()
    if randomize:
        random.shuffle(keys)
    for key in keys:
        if len(batched_set[key]) >= batch_size:
            batch = batched_set[key]
            if randomize:
                random.shuffle(batch)
            for j in range(len(batch) // batch_size):
                if max_size is not None and curr_size == max_size:
                    return batched_data
                batched_data.add_batch(batch[batch_size * j:batch_size * (j + 1)])
                curr_size += 1
    if randomize:
        random.shuffle(batched_data.batched_data)
    return batched_data
