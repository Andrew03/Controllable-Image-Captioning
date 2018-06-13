def create_data_iter(data, batch_size, word_vocab, topic_vocab, basedir, transform, progress_bar=True, description="", max_size=None):
    from tqdm import tqdm
    from data_loader import get_loader
    data_loader = get_loader(data, batch_size, word_vocab, topic_vocab, basedir, transform, max_size=max_size)
    data_iter = tqdm(iterable=data_loader, desc=description) if progress_bar else data_loader
    return data_iter

def _evaluate(images, topics, captions, encoder, decoder, loss_function, is_train=False, use_cuda=True):
    import torch
    from torch.nn.utils.rnn import pack_padded_sequence
        
    # Strip off <EOS> for input and <SOS> for targets
    inputs = captions[:, :-1]
    targets = captions[:, 1:].detach()
    features = encoder(images)
    len_targets = len(targets[0])
    targets = pack_padded_sequence(
        targets, [len_targets for i in range(len(captions))],
        batch_first=True)[0]
    if is_train:
        features.requires_grad_()
    predictions, _ = decoder(features, topics, inputs)
    loss = loss_function(predictions, targets)
    return loss

""" For a single training step (1 mini-batch) """
def train_step(images, topics, captions, encoder, decoder, loss_function, optimizer, grad_clip):
    import torch.nn as nn

    decoder.train()
    decoder.zero_grad()
    loss = _evaluate(images, topics, captions, encoder, decoder, loss_function, is_train=True)
    loss.backward()
    nn.utils.clip_grad_norm_(decoder.parameters(), grad_clip)
    optimizer.step()
    return loss.data.item()

def train(data_iter, encoder, decoder, loss_function, optimizer, grad_clip, 
          log_interval=100, progress_bar=True, description="Train", use_cuda=True, cuda_device=0):
    import torch

    log = {}
    loss_sum = 0.0
    for i, (images, topics, captions, caption_lengths, image_ids) in enumerate(data_iter, 1):
        device = torch.device("cuda:{}".format(cuda_device) if torch.cuda.is_available() and use_cuda else "cpu")
        images, topics, captions = images.to(device), topics.to(device), captions.to(device)
        loss_sum += train_step(images, topics, captions, encoder, decoder, loss_function, optimizer, grad_clip)
        if progress_bar:
            data_iter.set_postfix(loss=loss_sum/(i % log_interval if i % log_interval != 0 else log_interval))
        if i % log_interval == 0:
            log[i] = loss_sum
            loss_sum = 0.0
    # Leftover results
    if len(data_iter) % log_interval != 0:
        log[len(data_iter)] = loss_sum / (len(data_iter) % log_interval)
    return log

""" For a single validation step (1 mini-batch) """
def val_step(images, topics, captions, encoder, decoder, loss_function):
    import torch
    decoder.eval()
    with torch.no_grad():
        loss = _evaluate(images, topics, captions, encoder, decoder, loss_function, is_train=False)
    return loss.data.item()

""" Gets the loss over the entire validation set """
def validate(data_iter, encoder, decoder, loss_function, progress_bar=True, description="Val", use_cuda=True, cuda_device=0):
    import torch

    loss_sum = 0.0
    for i, (images, topics, captions, caption_lengths, image_ids) in enumerate(data_iter, 1):
        device = torch.device("cuda:{}".format(cuda_device) if torch.cuda.is_available() and use_cuda else "cpu")
        images, topics, captions = images.to(device), topics.to(device), captions.to(device)
        loss_sum += val_step(images, topics, captions, encoder, decoder, loss_function)
        if progress_bar:
            data_iter.set_postfix(loss=loss_sum/i)
    return loss_sum / len(data_iter)
