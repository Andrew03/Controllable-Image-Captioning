def find_accuracy(predicted, target, length):
    from nltk.translate.bleu_score import sentence_bleu
    return 0
    # compute bleu score?
    #top_predicted = [str(x) for x in predicted.argmax(dim=1).tolist()]
    top_predicted = predicted.argmax(dim=1).tolist()
    print(top_predicted)
    target = [str(x) for x in target.tolist()]
    bleu_score = 0
    for i in range(len(top_predicted) // length):
        reference = target[i * length : (i + 1) * length]
        candidate = top_predicted[i * length : (i + 1) * length]
        print(reference)
        print(candidate)
        bleu_score += sentence_bleu(reference, candidate)
    print(len(top_predicted) // length)
    return bleu_score / (len(top_predicted) // length)

def train_model(model, criterion, optimizer, scheduler, data_loaders, device, output_dir, start_epoch=0, num_epochs=10, log_interval=100, grad_clip=5.0):
    import torch
    from tqdm import tqdm
    from torch.nn.utils.rnn import pack_padded_sequence

    """ Defining Logging Structures """
    len_train_loader = len(data_loaders['train'])
    len_val_loader = len(data_loaders['val'])
    logs = {'train': {}, 'val': {}, 'loader_length': len_train_loader}
    model_data = {}

    """ Main Training Loop """
    end_epoch = start_epoch + num_epochs + 1
    for epoch in range(start_epoch + 1, end_epoch):
        for phase in ['train', 'val']:
            logs[phase][epoch] = {'loss': {}, 'accuracy': {}}
            is_train = phase == 'train'
            if is_train:
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_accuracy = 0.0

            progress_bar = tqdm(iterable=data_loaders[phase], desc="{} [{}/{}]".format(phase, epoch, end_epoch - 1))
            for i, data in enumerate(progress_bar, 1):
                captions = data['captions']
                optimizer.zero_grad()
                with torch.set_grad_enabled(is_train):
                    size = captions.size()
                    length = size[1] - 1
                    inputs = captions.narrow(1, 0, length)
                    targets = captions.narrow(1, 1, length)
                    packed_targets = pack_padded_sequence(targets, [length for _ in range(size[0])], batch_first=True)[0]

                    images = data['images'].to(device)
                    topics = data['topics'].to(device)
                    inputs = inputs.to(device)
                    packed_targets = packed_targets.to(device)

                    outputs = model(images, topics, inputs)
                    packed_outputs = pack_padded_sequence(outputs, [length for _ in range(size[0])], batch_first=True)[0]
                    loss = criterion(packed_outputs, packed_targets)

                    if is_train:
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        optimizer.step()
                        if i % log_interval == 0:
                            logs[phase][epoch]['loss'][i] = running_loss / log_interval
                            logs[phase][epoch]['accuracy'][i] = running_accuracy / log_interval
                            running_loss = 0.0
                            running_accuracy = 0.0
                running_loss += loss.item()
                running_accuracy += find_accuracy(outputs, targets, length)
                progress_bar.set_postfix(loss=running_loss/(i % log_interval if i % log_interval != 0 else log_interval), 
                    accuracy=running_accuracy/(i % log_interval if i % log_interval != 0 else log_interval))
            if is_train:
                logs[phase][epoch]['loss'][len_train_loader] = running_loss / (i % log_interval if i % log_interval != 0 else log_interval)
                logs[phase][epoch]['accuracy'][len_train_loader] = running_accuracy / (i % log_interval if i % log_interval != 0 else log_interval)
            else:
                logs[phase][epoch]['loss'][len_train_loader] = running_loss / len_val_loader
                logs[phase][epoch]['accuracy'][len_train_loader] = running_accuracy / len_val_loader
        model_data[epoch] = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        torch.save(model_data[epoch], os.path.join(output_dir, "checkpoint_{}.pt".format(epoch)))
    return logs, model_data

def no_average(model):
    return

def train_model_distributed(model, criterion, optimizer, scheduler, data_loaders, device, output_dir, rank=0, num_gpus=1, average_gradients=no_average, start_epoch=0, num_epochs=10, log_interval=100, grad_clip=5.0):
    import os
    import torch
    from tqdm import tqdm
    from torch.nn.utils.rnn import pack_padded_sequence

    """ Defining Logging Structures """
    # Modify this for storage reasons later
    log_interval = log_interval
    len_train_loader = len(data_loaders['train'])
    len_val_loader = len(data_loaders['val'])
    logs = {'train': {}, 'val': {}, 'loader_length': len_train_loader}
    model_data = {}

    """ Main Training Loop """
    end_epoch = start_epoch + num_epochs + 1
    for epoch in range(start_epoch + 1, end_epoch):
        for phase in ['train', 'val']:
            logs[phase][epoch] = {'loss': {}, 'accuracy': {}}
            is_train = phase == 'train'
            if is_train:
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_accuracy = 0.0

            progress_bar = tqdm(iterable=data_loaders[phase], desc="{} [{}/{}]".format(phase, epoch, end_epoch - 1)) if rank == 0 else data_loaders[phase]
            for i, data in enumerate(progress_bar, 1):
                captions = data['captions']
                optimizer.zero_grad()
                with torch.set_grad_enabled(is_train):
                    size = captions.size()
                    length = size[1] - 1
                    inputs = captions.narrow(1, 0, length)
                    targets = captions.narrow(1, 1, length)
                    packed_targets = pack_padded_sequence(targets, [length for _ in range(size[0])], batch_first=True)[0]

                    images = data['images'].to(device)
                    topics = data['topics'].to(device)
                    inputs = inputs.to(device)
                    packed_targets = packed_targets.to(device)

                    outputs = model(images, topics, inputs)
                    packed_outputs = pack_padded_sequence(outputs, [length for _ in range(size[0])], batch_first=True)[0]
                    loss = criterion(packed_outputs, packed_targets)

                    if is_train:
                        loss.backward()
                        average_gradients(model)
                        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
                        optimizer.step()
                        if i % log_interval == 0:
                            logs[phase][epoch]['loss'][i] = running_loss / log_interval
                            logs[phase][epoch]['accuracy'][i] = running_accuracy / log_interval
                            running_loss = 0.0
                            running_accuracy = 0.0
                running_loss += loss.item()
                running_accuracy += find_accuracy(outputs, targets, length)
                if rank == 0:
                    if is_train:
                        progress_bar.set_postfix(loss=running_loss/(i % log_interval if i % log_interval != 0 else 1), 
                            accuracy=running_accuracy/(i % log_interval if i % log_interval != 0 else 1))
                    else:
                        progress_bar.set_postfix(loss=running_loss/i, accuracy=running_accuracy/i)
            if is_train:
                logs[phase][epoch]['loss'][len_train_loader] = running_loss / (i % log_interval if i % log_interval != 0 else log_interval)
                logs[phase][epoch]['accuracy'][len_train_loader] = running_accuracy / (i % log_interval if i % log_interval != 0 else log_interval)
            else:
                logs[phase][epoch]['loss'][len_train_loader] = running_loss / len_val_loader
                logs[phase][epoch]['accuracy'][len_train_loader] = running_accuracy / len_val_loader
        scheduler.step(logs['val'][epoch]['loss'][len_train_loader])
        model_data[epoch] = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        if rank == 0:
            torch.save(model_data[epoch], os.path.join(output_dir, "checkpoint_{}.pt".format(epoch)))
    return logs, model_data
