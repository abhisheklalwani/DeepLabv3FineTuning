import copy
import csv
import os
import time

import numpy as np
import torch
from torch.functional import _return_counts
from torch.nn.modules.loss import CrossEntropyLoss
from tqdm import tqdm


def train_model(model, criterion, dataloaders, optimizer, bpath,
                num_epochs,exp_dir,num_classes):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_loss = 1e10
    # Use gpu if available
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Initialize the log file for training and testing loss and metrics
    fieldnames = ['epoch', 'Train_loss', 'Validation_loss']
    with open(os.path.join(bpath, 'log.csv'), 'w', newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

    for epoch in range(1, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        # Each epoch has a training and validation phase
        # Initialize batch summary
        batchsummary = {a: [0] for a in fieldnames}

        for phase in ['Train', 'Validation']:
            if phase == 'Train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_iou_means = []

            # Iterate over data.
            for sample in tqdm(iter(dataloaders[phase])):
                inputs = sample['image'].to(device)
                masks = sample['mask'].to(device)
                # zero the parameter gradients
                optimizer.zero_grad()

                # track history if only in train
                with torch.set_grad_enabled(phase == 'Train'):
                    outputs = model(inputs)['out']
                    loss = criterion(outputs, masks)
                    #print(torch.unique(masks,return_counts=True))
                    _, preds = torch.max(outputs, 1)

                    # backward + optimize only if in training phase
                    if phase == 'Train':
                        loss.backward()
                        optimizer.step()
            batchsummary['epoch'] = epoch
            epoch_loss = loss

            iou_mean = iou(preds, masks, num_classes).mean()
            running_loss += loss.item() * inputs.size(0)
            running_iou_means.append(iou_mean)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            if running_iou_means is not None:
                epoch_acc = np.array(running_iou_means).mean()
            else:
                epoch_acc = 0.
            batchsummary[f'{phase}_loss'] = epoch_loss
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
        for field in fieldnames[3:]:
            batchsummary[field] = np.mean(batchsummary[field])
        with open(os.path.join(bpath, 'log.csv'), 'a', newline='') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writerow(batchsummary)
            # deep copy the model
            if phase == 'Validation' and loss < best_loss:
                best_loss = loss
                best_model_wts = copy.deepcopy(model.state_dict())
        if 0 == epoch%5:
            current_model_path = os.path.join(exp_dir, f"checkpoint_{epoch:04}_DeepLabV3_SmallObject.pt")
            print(f"Save current model : {current_model_path}")
            torch.save(model, current_model_path)

    current_model_path = os.path.join(exp_dir, f"best_weights.pt")
    print(f"Saving Best Model")
    model.load_state_dict(best_model_wts)
    torch.save(model, current_model_path)
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Lowest Validation Loss: {:4f}'.format(best_loss))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

def iou(pred, target, n_classes = 3):
  ious = []
  pred = pred.view(-1)
  target = target.view(-1)

  # Ignore IoU for background class ("0")
  for cls in range(1, n_classes):  # This goes from 1:n_classes-1 -> class "0" is ignored
    pred_inds = pred == cls
    target_inds = target == cls
    intersection = (pred_inds[target_inds]).long().sum().data.cpu().item()  # Cast to long to prevent overflows
    union = pred_inds.long().sum().data.cpu().item() + target_inds.long().sum().data.cpu().item() - intersection
    if union > 0:
        ious.append(float(intersection) / float(max(union, 1)))

  return np.array(ious)
