import torch
import torch.nn as nn
from model.densenet import DenseNet
from model.SEdense import SEDenseNet
from model.resnext import ResNeXt
from data_processing.prepare_data import create_dataloader
from tqdm import tqdm
import pathlib
temp = pathlib.PosixPath
pathlib.PosixPath = pathlib.WindowsPath

def val_step(val_loader, model_1, model_2, model_3, loss_fn, device):
    model_1.eval()
    model_2.eval()
    model_3.eval()
    val_loss = 0.0
    correct, correct_num = 0, 0

    with torch.no_grad():
        for (image, label) in tqdm(val_loader):
            image = image.to(device)
            label = label.to(device)
            output_1 = model_1(image)
            output_2 = model_2(image)
            output_3 = model_3(image)
            output =  output_1 * 0.5 + output_2 * 0.2 + output_3 * 0.3
            loss = loss_fn(output, label)
            val_loss += loss.item()

            pred = torch.argmax(output, dim = 1)
            correct += torch.sum(torch.eq(pred, label))
            correct_num += label.size()[0]

    total_val_loss = val_loss / len(val_loader)
    accuracy = correct / correct_num

    return total_val_loss, accuracy


def evaluate(args):
    val_loader = create_dataloader(args, 'val')
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # torch.cuda.set_device(device)
    model_1 = DenseNet()
    model_2 = SEDenseNet(growth_rate=32, block_config=(6, 12, 32, 32), 
                num_init_features=64, bn_size=4,
                drop_rate=0.1, num_classes=args.num_label,
                memory_efficient=False)
    model_3 = ResNeXt()
    
    if args.use_pretrained :
        ckp_1 = torch.load(args.check_dir_1 / 'best_model.pt', map_location='cpu')
        model_1.load_state_dict(ckp_1['model'])
        ckp_2 = torch.load(args.check_dir_2 / 'best_model.pt', map_location='cpu')
        model_2.load_state_dict(ckp_2['model'])
        ckp_3 = torch.load(args.check_dir_3 / 'best_model.pt', map_location='cpu')
        model_3.load_state_dict(ckp_3['model'])

    model_1.to(device=device)
    model_2.to(device=device)
    model_3.to(device=device)
    loss_fn = nn.CrossEntropyLoss().to(device)
    val_loss, val_acc = val_step(val_loader, model_1, model_2, model_3, loss_fn, device)
    print(f"Loss : {val_loss:.4f}, \t Accuracy : {val_acc * 100 :.2f} %")
