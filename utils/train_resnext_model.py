import torch
import torch.nn as nn
import numpy as np
from model.cosinewarmup import CosineAnnealingWarmUpRestarts
from model.resnext import ResNeXt
from data_processing.prepare_data import create_dataloader

def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2


def train_step(train_loader, args, model, epoch, optimizer, loss_fn, device):
    model.train()
    train_loss = 0.0

    for i, (image, label) in enumerate(train_loader):
        optimizer.zero_grad()

        image = image.to(device)
        label = label.to(device)

        r = np.random.rand(1)
        if args.beta > 0 and r < args.cutmix_prob:
            # generate mixed sample
            lam = np.random.beta(args.beta, args.beta)
            rand_index = torch.randperm(image.size()[0]).to(device)
            target_a = label
            target_b = label[rand_index]

            bbx1, bby1, bbx2, bby2 = rand_bbox(image.size(), lam)
            image[:, :, bbx1:bbx2, bby1:bby2] = image[rand_index, :, bbx1:bbx2, bby1:bby2]
            # adjust lambda to exactly match pixel ratio
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (image.size()[-1] * image.size()[-2]))
            # compute output
            output = model(image)
            loss = loss_fn(output, target_a) * lam + loss_fn(output, target_b) * (1. - lam)
            
        else:
            # compute output
            output = model(image)
            loss = loss_fn(output, label)
        
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

        if i % args.log_print_interval == 0:
            print(f"Epoch : {epoch} \t Iteration : {i} \t Loss : {loss.item():.4f}")

    total_train_loss = train_loss / len(train_loader)
    return total_train_loss


def val_step(val_loader, model, loss_fn, device):
    model.eval()
    val_loss = 0.0
    correct, correct_num = 0, 0

    with torch.no_grad():
        for i, (image, label) in enumerate(val_loader):
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            loss = loss_fn(output, label)
            val_loss += loss.item()

            pred = torch.argmax(output, dim = 1)
            correct += torch.sum(torch.eq(pred, label))
            correct_num += label.size()[0]

    total_val_loss = val_loss / len(val_loader)
    accuracy = correct / correct_num

    return total_val_loss, accuracy




def save_checkpoint(args, epoch, model, optimizer, val_loss):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_loss': val_loss,
        },
        f = args.check_dir / f"model_epoch_{epoch}.pt"
    )

def save_best_checkpoint(args, epoch, model, optimizer, val_loss):
    torch.save(
        {
            'epoch': epoch,
            'args': args,
            'model': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'val_loss': val_loss,
        },
        f = args.check_dir / 'best_model.pt'
    )


def train(args):
    train_loader = create_dataloader(args, 'train')
    val_loader = create_dataloader(args, 'val')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # torch.cuda.set_device(device)

    best_val_loss = 10.0
    start_epoch = 0

    model = ResNeXt()

    if args.use_pretrained :
        print("... Loading Checkpoint ...")
        ckp = torch.load(args.check_dir / 'best_model.pt', map_location='cpu')
        best_val_loss = ckp['val_loss']
        start_epoch = ckp['epoch'] + 1
        model.load_state_dict(ckp['model'])
        print(f"... best_val_loss : {best_val_loss} \t start_epoch : {start_epoch} ...")


    model.to(device=device)

    loss_fn = nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr = args.learning_rate,
                                weight_decay = args.weight_decay)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[20, 30, 40, 50], gamma=0.1)
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.001)
    # scheduler = CosineAnnealingWarmUpRestarts(optimizer, T_0 = 30, T_mult = 1, eta_max=0.1, T_up=10, gamma=0.5)

    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch} ------------------------------------------ ")

        train_loss = train_step(train_loader, args, model, epoch, optimizer, loss_fn, device)
        val_loss, val_acc = val_step(val_loader, model, loss_fn, device)

        print(f"Epoch : [{epoch:4d}/{args.epochs:4d}] \t Train Loss : {train_loss:.4f} \t Validation Loss : {val_loss:.4f} \t accuracy : {val_acc * 100:.2f} %")
        
        if epoch % args.save_interval == 0:
            save_checkpoint(args, epoch, model, optimizer, val_loss)

        if val_loss < best_val_loss:
            print("@@@@@@@@@@@@@@@@@@@@@@@@@@@ New Record @@@@@@@@@@@@@@@@@@@@@@@@@@@\n")
            save_best_checkpoint(args, epoch, model, optimizer, val_loss)
            best_val_loss = val_loss

        scheduler.step()
        