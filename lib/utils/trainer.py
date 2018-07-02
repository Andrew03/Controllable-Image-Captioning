def find_accuracy(predicted, target):
    return 0

def train_model(model, criterion, optimizer, scheduler, data_loaders, device, start_epoch=0, num_epochs=10, log_interval=100):
    import torch
    from tqdm import tqdm

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
                #scheduler.step(optimizer)
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_accuracy = 0.0

            progress_bar = tqdm(iterable=data_loaders[phase], desc="{} [{}/{}]".format(phase, epoch, end_epoch))
            for i, data in enumerate(progress_bar, 1):
                images = data['images'].to(device)
                topics = data['topics'].to(device)
                captions = data['captions'].to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(is_train):
                    outputs = model(images, topics, captions)
                    # Need to refine using evaluation function
                    loss = criterion(outputs, captions.view(-1, 1))

                    if is_train:
                        loss.backward()
                        optimizer.step()
                        if i % log_interval == 0:
                            logs[phase][epoch]['loss'][i] = running_loss / log_interval
                            logs[phase][epoch]['accuracy'][i] = running_accuracy / log_interval
                            running_loss = 0.0
                            running_accuracy = 0.0
                running_loss += loss.item()
                running_accuracy += find_accuracy(outputs, targets)
                progress_bar.set_postfix(loss=running_loss/(i % log_interval if i % log_interval != 0 else log_interval), 
                    accuracy=running_accuracy/(i % log_interval if i % log_interval != 0 else log_interval))
            logs[phase][epoch]['loss'][len_train_loader] = running_loss / (i % log_interval if is_train else len_val_loader)
            logs[phase][epoch]['accuracy'][len_train_loader] = running_accuracy / (i % log_interval if is_train else len_val_loader)
        model_data[epoch] = {
            'epoch': epoch,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
        }
    return logs, model_data
