import torch
import argparse
import os
import numpy as np
import yaml
import random
from tqdm import tqdm
from model.ssd import SSD
from dataset.voc import VOCDataset
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import MultiStepLR

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
if torch.backends.mps.is_available():
    device = torch.device('mps')
    print('Using mps')


PATIENCE = 50

def collate_function(data):
    return tuple(zip(*data))


def train(args):
    with open(args.config_path, 'r') as file:
        try:
            config = yaml.safe_load(file)
        except yaml.YAMLError as exc:
            print(exc)
    print(config)

    dataset_config = config['dataset_params']
    train_config = config['train_params']

    seed = train_config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == 'cuda':
        torch.cuda.manual_seed_all(seed)

    voc_train = VOCDataset('train',
                     im_sets=dataset_config['train_im_sets'],
                     im_size=dataset_config['im_size'])
    train_dataset = DataLoader(voc_train,
                               batch_size=train_config['batch_size'],
                               shuffle=True,
                               collate_fn=collate_function)

    voc_val = VOCDataset('valid',
                         im_sets=dataset_config['valid_im_sets'],
                         im_size=dataset_config['im_size'])
    val_dataset = DataLoader(voc_val,
                             batch_size=train_config['batch_size'],
                             shuffle=False,
                             collate_fn=collate_function)

    model = SSD(config=config['model_params'],
                num_classes=dataset_config['num_classes'])
    model.to(device)
    model.train()

    ckpt_dir = train_config['task_name']
    ckpt_name = train_config['ckpt_name']
    best_ckpt_name = 'best_model.pth'
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)

    ckpt_path = os.path.join(ckpt_dir, ckpt_name)
    if os.path.exists(ckpt_path):
        print('Loading checkpoint as one exists')
        model.load_state_dict(torch.load(ckpt_path, map_location=device))

    optimizer = torch.optim.SGD(lr=train_config['lr'],
                                params=model.parameters(),
                                weight_decay=5E-4, momentum=0.9)
    # optimizer = torch.optim.Adam(lr=train_config['lr'],
    #                             params=model.parameters(),
    #                             weight_decay=5E-4, betas=(0.9, 0.999))
    lr_scheduler = MultiStepLR(optimizer, milestones=train_config['lr_steps'], gamma=0.5)
    acc_steps = train_config['acc_steps']
    num_epochs = train_config['num_epochs']
    steps = 0

    best_val_loss = np.inf
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        ssd_classification_losses = []
        ssd_localization_losses = []

        for idx, (ims, targets, _) in enumerate(tqdm(train_dataset, desc=f"Training Epoch {epoch+1}/{num_epochs}")):
            optimizer.zero_grad()
            for target in targets:
                target['boxes'] = target['bboxes'].float().to(device)
                del target['bboxes']
                target['labels'] = target['labels'].long().to(device)
            images = torch.stack([im.float().to(device) for im in ims], dim=0)
            batch_losses, _ = model(images, targets)
            loss = batch_losses['classification'] + batch_losses['bbox_regression']

            ssd_classification_losses.append(batch_losses['classification'].item())
            ssd_localization_losses.append(batch_losses['bbox_regression'].item())
            loss = loss / acc_steps
            loss.backward()

            if (idx + 1) % acc_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if steps % train_config['log_steps'] == 0:
                print('Step {}: SSD Classification Loss : {:.4f} | SSD Localization Loss : {:.4f}'
                      .format(steps, np.mean(ssd_classification_losses), np.mean(ssd_localization_losses)))

            if torch.isnan(loss):
                print('Loss is becoming nan. Exiting')
                exit(0)
            steps += 1

        optimizer.step()
        optimizer.zero_grad()
        lr_scheduler.step()

        train_class_loss = np.mean(ssd_classification_losses)
        train_loc_loss = np.mean(ssd_localization_losses)
        print('Finished epoch {}'.format(epoch + 1))
        print('Training - SSD Classification Loss : {:.4f} | SSD Localization Loss : {:.4f}'
              .format(train_class_loss, train_loc_loss))

        model.train()
        val_classification_losses = []
        val_localization_losses = []
        with torch.no_grad():
            for idx, (ims, targets, _) in enumerate(tqdm(val_dataset, desc="Validating")):
                for target in targets:
                    target['boxes'] = target['bboxes'].float().to(device)
                    del target['bboxes']
                    target['labels'] = target['labels'].long().to(device)
                images = torch.stack([im.float().to(device) for im in ims], dim=0)

                val_batch_losses, _ = model(images, targets)
                val_classification_losses.append(val_batch_losses['classification'].item())
                val_localization_losses.append(val_batch_losses['bbox_regression'].item())

            val_class_loss = np.mean(val_classification_losses)
            val_loc_loss = np.mean(val_localization_losses)
            val_total_loss = val_class_loss + val_loc_loss
            print('Validation - SSD Classification Loss : {:.4f} | SSD Localization Loss : {:.4f} | Total: {:.4f}'
                  .format(val_class_loss, val_loc_loss, val_total_loss))

        torch.save(model.state_dict(), os.path.join(ckpt_dir, ckpt_name))

        if val_total_loss < best_val_loss:
            best_val_loss = val_total_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), os.path.join(ckpt_dir, best_ckpt_name))
            print("Found better validation loss. Best model updated!")
        else:
            epochs_no_improve += 1
            print(f"No improvement in validation loss for {epochs_no_improve} epoch(s).")
        if epochs_no_improve >= PATIENCE:
            print("Early stopping triggered due to no improvement in validation loss.")
            break

    print('Done Training')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Arguments for ssd training')
    parser.add_argument('--config', dest='config_path',
                        default='config/voc.yaml', type=str)
    args = parser.parse_args()
    train(args)
