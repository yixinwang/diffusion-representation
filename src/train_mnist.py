import torch
import torch.nn as nn
from torchvision.datasets import MNIST
from torchvision import transforms 
from torchvision.utils import save_image
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR
from model import MNISTDiffusion
from utils import ExponentialMovingAverage
import os
import math
import argparse


def save_checkpoint(output_dir, global_steps, epoch, model, model_ema, optimizer, scheduler, args):
    """Persist diffusion (train) and score (EMA) models plus trainer state."""
    os.makedirs(output_dir, exist_ok=True)
    ckpt_path = os.path.join(output_dir, "steps_{:0>8}.pt".format(global_steps))
    state = {
        "model": model.state_dict(),
        "model_ema": model_ema.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "epoch": epoch,
        "global_steps": global_steps,
        "args": vars(args),
    }
    torch.save(state, ckpt_path)
    return ckpt_path


def load_checkpoint(path, model, model_ema, device, optimizer=None, scheduler=None):
    if not os.path.isfile(path):
        raise FileNotFoundError("Checkpoint {} does not exist".format(path))
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    model_ema.load_state_dict(checkpoint["model_ema"])
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    start_epoch = checkpoint.get("epoch", 0)
    global_steps = checkpoint.get("global_steps", 0)
    return start_epoch, global_steps

def create_mnist_dataloaders(batch_size,image_size=28,num_workers=4):
    
    preprocess=transforms.Compose([transforms.Resize(image_size),\
                                    transforms.ToTensor(),\
                                    transforms.Normalize([0.5],[0.5])]) #[0,1] to [-1,1]

    train_dataset=MNIST(root="./mnist_data",\
                        train=True,\
                        download=True,\
                        transform=preprocess
                        )
    test_dataset=MNIST(root="./mnist_data",\
                        train=False,\
                        download=True,\
                        transform=preprocess
                        )

    return DataLoader(train_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers),\
            DataLoader(test_dataset,batch_size=batch_size,shuffle=True,num_workers=num_workers)



def parse_args():
    parser = argparse.ArgumentParser(description="Training MNISTDiffusion")
    parser.add_argument('--lr',type = float ,default=0.001)
    parser.add_argument('--batch_size',type = int ,default=128)    
    parser.add_argument('--epochs',type = int,default=50)
    parser.add_argument('--ckpt',type = str,help = 'define checkpoint path',default='')
    parser.add_argument('--n_samples',type = int,help = 'define sampling amounts after every epoch trained',default=36)
    parser.add_argument('--model_base_dim',type = int,help = 'base dim of Unet',default=64)
    parser.add_argument('--timesteps',type = int,help = 'sampling steps of DDPM',default=1000)
    parser.add_argument('--model_ema_steps',type = int,help = 'ema model evaluation interval',default=10)
    parser.add_argument('--model_ema_decay',type = float,help = 'ema model decay',default=0.995)
    parser.add_argument('--log_freq',type = int,help = 'training log message printing frequence',default=10)
    parser.add_argument('--no_clip',action='store_true',help = 'set to normal sampling method without clip x_0 which could yield unstable samples')
    parser.add_argument('--cpu',action='store_true',help = 'cpu training')
    parser.add_argument('--output_dir',type = str,help = 'directory for checkpoints and samples',default='results')

    args = parser.parse_args()

    return args


def main(args):
    device="cpu" if args.cpu else "cuda"
    train_dataloader,test_dataloader=create_mnist_dataloaders(batch_size=args.batch_size,image_size=28)
    model=MNISTDiffusion(timesteps=args.timesteps,
                image_size=28,
                in_channels=1,
                base_dim=args.model_base_dim,
                dim_mults=[2,4]).to(device)

    #torchvision ema setting
    #https://github.com/pytorch/vision/blob/main/references/classification/train.py#L317
    adjust = 1* args.batch_size * args.model_ema_steps / args.epochs
    alpha = 1.0 - args.model_ema_decay
    alpha = min(1.0, alpha * adjust)
    model_ema = ExponentialMovingAverage(model, device=device, decay=1.0 - alpha)

    optimizer=AdamW(model.parameters(),lr=args.lr)
    scheduler=OneCycleLR(optimizer,args.lr,total_steps=args.epochs*len(train_dataloader),pct_start=0.25,anneal_strategy='cos')
    loss_fn=nn.MSELoss(reduction='mean')

    start_epoch=0
    global_steps=0
    if args.ckpt:
        start_epoch,global_steps=load_checkpoint(args.ckpt,model,model_ema,device,optimizer,scheduler)
        print("Loaded checkpoint {} (epoch={}, global_steps={})".format(args.ckpt,start_epoch,global_steps))

    for i in range(start_epoch,args.epochs):
        model.train()
        for j,(image,target) in enumerate(train_dataloader):
            noise=torch.randn_like(image).to(device)
            image=image.to(device)
            pred=model(image,noise)
            loss=loss_fn(pred,noise)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            scheduler.step()
            if global_steps%args.model_ema_steps==0:
                model_ema.update_parameters(model)
            global_steps+=1
            if j%args.log_freq==0:
                print("Epoch[{}/{}],Step[{}/{}],loss:{:.5f},lr:{:.5f}".format(i+1,args.epochs,j,len(train_dataloader),
                                                                    loss.detach().cpu().item(),scheduler.get_last_lr()[0]))
        ckpt_path=save_checkpoint(args.output_dir,global_steps,i+1,model,model_ema,optimizer,scheduler,args)
        print("Saved checkpoint to {}".format(ckpt_path))

        model_ema.eval()
        samples=model_ema.module.sampling(args.n_samples,clipped_reverse_diffusion=not args.no_clip,device=device)
        os.makedirs(args.output_dir,exist_ok=True)
        save_image(samples,os.path.join(args.output_dir,"steps_{:0>8}.png".format(global_steps)),nrow=int(math.sqrt(args.n_samples)))

if __name__=="__main__":
    args=parse_args()
    main(args)