def find_accuracy(predicted, target, length):
    from nltk.translate.bleu_score import sentence_bleu
    top_predicted = predicted.argmax(dim=2).tolist()
    target = target.tolist()
    bleu_score = 0
    for i in range(len(top_predicted)):
        reference = [str(x) for x in target[i][:-1]]
        candidate = [str(x) for x in top_predicted[i][:-1]]
        bleu_score += sentence_bleu(reference, candidate)
    return bleu_score / len(top_predicted)

def no_average(model):
    return

def train_model(model, criterion, optimizer, scheduler, data_loaders, device, output_dir, rank=0, num_gpus=1, average_gradients=no_average, start_epoch=0, num_epochs_teacher=10, num_epochs_no_teacher=10, log_interval=100, grad_clip=5.0):
    import os
    import torch
    from torch.nn.utils.rnn import pack_padded_sequence
    from lib.utils.detect_notebook import is_notebook
    if is_notebook():
        from tqdm import tqdm_notebook as tqdm
    else:
        from tqdm import tqdm

    """ Defining Logging Structures """
    # Modify this for storage reasons later
    log_interval = log_interval
    len_train_loader = len(data_loaders['train'])
    len_val_loader = len(data_loaders['val'])
    logs = {'train': {}, 'val': {}, 'loader_length': len_train_loader}
    model_data = {}

    """ Main Training Loop """
    end_epoch = start_epoch + num_epochs_teacher + num_epochs_no_teacher + 1
    for epoch in range(start_epoch + 1, end_epoch):
        is_teacher = epoch <= start_epoch + num_epochs_teacher + 1
        for phase in ['train', 'val']:
            logs[phase][epoch] = {'loss': {}, 'accuracy': {}}
            is_train = phase == 'train'
            if is_train:
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_accuracy = 0.0

            progress_bar = tqdm(iterable=data_loaders[phase], desc="{} ({}) [{}/{}]".format(phase, "teacher" if is_teacher else "no teacher", epoch, end_epoch - 1)) if rank == 0 else data_loaders[phase]
            for i, data in enumerate(progress_bar, 1):
                captions = data['captions']
                optimizer.zero_grad()
                with torch.set_grad_enabled(is_train):
                    length = captions.size(1) - 1
                    inputs = captions.narrow(1, 0, length)
                    targets = captions.narrow(1, 1, length)
                    packed_targets = pack_padded_sequence(targets, [length for _ in range(captions.size(0))], batch_first=True)[0]

                    images = data['images'].to(device)
                    topics = data['topics'].to(device)
                    inputs = inputs.to(device)
                    packed_targets = packed_targets.to(device)

                    outputs = model(images, topics, inputs, use_teacher=is_teacher)
                    packed_outputs = pack_padded_sequence(outputs, [length for _ in range(captions.size(0))], batch_first=True)[0]
                    loss = criterion(packed_outputs, packed_targets)

                    running_loss += loss.item()
                    running_accuracy += find_accuracy(outputs, targets, length)
                    if rank == 0:
                        if is_train:
                            progress_bar.set_postfix(loss=running_loss/(i % log_interval if i % log_interval != 0 else log_interval), 
                                accuracy=running_accuracy/(i % log_interval if i % log_interval != 0 else log_interval))
                        else:
                            progress_bar.set_postfix(loss=running_loss/i, accuracy=running_accuracy/i)

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
            if is_train:
                logs[phase][epoch]['loss'][len_train_loader] = running_loss / (i % log_interval if i % log_interval != 0 else log_interval)
                logs[phase][epoch]['accuracy'][len_train_loader] = running_accuracy / (i % log_interval if i % log_interval != 0 else log_interval)
            else:
                logs[phase][epoch]['loss'][len_train_loader] = running_loss / len_val_loader
                logs[phase][epoch]['accuracy'][len_train_loader] = running_accuracy / len_val_loader
        # scheduler.step(logs['val'][epoch]['loss'][len_train_loader])
        model_data[epoch] = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
        if rank == 0:
            torch.save(model_data[epoch], os.path.join(output_dir, "checkpoint_{}.pt".format(epoch)))
    return logs, model_data
