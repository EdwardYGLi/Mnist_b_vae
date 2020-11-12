"""
Created by Edward Li at 11/11/20
"""
from __future__ import print_function

import argparse
import datetime
import os

import torch
import torch.nn.functional as F
import torch.utils.data
import tqdm
import wandb
from b_vae import BetaVAE
from torch import nn
from torch.utils.tensorboard import SummaryWriter
from torchvision import datasets, transforms
from torchvision.utils import make_grid

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_run_name():
    """
     Function to return wandb run name.
     Currently this is the only way to get the run name.
     In a future bugfix wandb.run.name will then be available.
    :return: run name or dryrun when dryrun.
    """
    try:
        wandb.run.save()
        run_name = wandb.run.name
        return run_name
    except BaseException:
        # when WANDB_MODE=dryrun the above will throw an exception. At that
        # stage we except and return "dryrun"
        return "dryrun"


def init_wandb(args, tag):
    # initialize weights and biases.
    wandb.init(project="mnist-VAE", dir="../.wandb/", tags=[tag])
    wandb.tensorboard.patch(save=False, tensorboardX=False)
    wandb.config.update(args)


# Reconstruction + KL divergence losses summed over all elements and batch
def loss_function(recon_x, x, mu, log_var, loss_fn="mse"):
    if loss_fn == "mse":
        recon_loss = nn.MSELoss()(torch.sigmoid(recon_x), x.view(-1, 784))
    elif loss_fn == "bce":
        recon_loss = F.binary_cross_entropy_with_logits(recon_x, x.view(-1, 784))
    elif loss_fn == "mae":
        recon_loss = torch.mean(torch.abs(torch.sigmoid(recon_x) - x.view(-1, 784)))
    else:
        raise NotImplementedError("loss function {} not implemented".format(loss_fn))
    # see Appendix B from VAE paper:
    # Kingma and Welling. Auto-Encoding Variational Bayes. ICLR, 2014
    # https://arxiv.org/abs/1312.6114
    # 0.5 * sum(1 + log(sigma^2) - mu^2 - sigma^2)
    kld_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())

    return recon_loss, kld_loss


def train(model, train_loader, optimizer, beta, loss_fn):
    model.train()
    total_loss = 0
    total_bce = 0
    total_kld = 0
    for batch_idx, (data, _) in enumerate(train_loader):
        data = data.to(device)
        optimizer.zero_grad()
        recon_batch, mu, log_var = model(data)
        loss_bce, loss_kld = loss_function(recon_batch, data, mu, log_var, loss_fn)
        loss = loss_bce + beta * loss_kld
        loss.backward()
        total_loss += loss.item()
        total_bce += loss_bce.item()
        total_kld += loss_kld.item()

        optimizer.step()

    return {"total_loss": total_loss / len(train_loader.dataset),
            "epoch_recon": total_bce / len(train_loader.dataset),
            "epoch_kld": total_kld / len(train_loader.dataset)}


def test(model, epoch, test_loader, batch_size, metric_logger, beta, loss_fn):
    model.eval()
    total_loss = 0
    total_bce = 0
    total_kld = 0

    with torch.no_grad():
        for i, (data, _) in tqdm.tqdm(enumerate(test_loader)):
            data = data.to(device)
            recon_batch, mu, log_var = model(data)
            loss_bce, loss_kld = loss_function(recon_batch, data, mu, log_var, loss_fn)
            loss = loss_bce + beta * loss_kld
            total_loss += loss.item()
            total_bce += loss_bce.item()
            total_kld += loss_kld.item()

            if i < 5:
                n = min(data.size(0), 8)
                comparison = torch.cat([data[:n],
                                        torch.sigmoid(recon_batch).view(batch_size, 1, 28, 28)[:n]])
                grid = make_grid(comparison, nrow=n, padding=2, pad_value=0,
                                 normalize=False, range=None, scale_each=False)
                # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
                image = grid.mul_(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8) # permute(1, 2, 0).cpu().numpy().astype(np.uint8)
                metric_logger.add_image("recon_b_{}_comp".format(i), image, epoch)

    return {"total_loss": total_loss / len(test_loader.dataset),
            "epoch_recon": total_bce / len(test_loader.dataset),
            "epoch_kld": total_kld / len(test_loader.dataset)}


def log_loss_dictionary(loss_dict, phase, step, metric_logger):
    for key, value in loss_dict.items():
        metric_logger.add_scalar("train_{}".format(key), value, step)
        pass


def main():
    parser = argparse.ArgumentParser(description='VAE MNIST Example')
    parser.add_argument('--batch-size', type=int, default=128, metavar='N',
                        help='input batch size for training (default: 128)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--lr', type=float, default=0.003,
                        help='learning rate')
    parser.add_argument('--wd', type=float, default=0.03,
                        help='weight decay')
    parser.add_argument("--activation", default="relu")
    parser.add_argument('--output_dir', type=str, help="output directory", default="./output/")
    parser.add_argument("--layers", type=str,
                        help="layer sizes in comma separated format default(400,20,400), "
                             "middle number will be the size of the latent space"
                             "The first and last layers are always 784 due to (28x28) size of mnist",
                        default="400,20,400")
    parser.add_argument("-b", "--beta", type=float, help="the beta regularization in a b-vae, b==1 implies vae",
                        default=1.0)
    parser.add_argument("--loss_fn", help="loss functions (bce, mse,mae)", default="mse")

    args = parser.parse_args()
    now = datetime.datetime.now()
    init_wandb(args, "beta vae,mnist digits")
    output_dir = os.path.join(args.output_dir, now.strftime("%Y-%m-%d_%H_%M") + "_vae/")
    os.makedirs(output_dir, exist_ok=True)
    args.cuda = torch.cuda.is_available()
    layers = [int(x) for x in args.layers.split(",")]
    print("building vae with layers : " + args.layers)
    assert (len(layers) % 2) == 1, "VAE layer sizes must be mirrored"
    torch.manual_seed(args.seed)

    kwargs = {'num_workers': 1, 'pin_memory': True} if args.cuda else {}
    train_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=True, download=True,
                       transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        datasets.MNIST('../data', train=False, transform=transforms.ToTensor()),
        batch_size=args.batch_size, shuffle=False, **kwargs)

    model = BetaVAE(layers, activation = args.activation).to(device)
    wandb.watch(model)

    print(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)

    metric_logger = SummaryWriter("../.runs/")

    best_loss = 1E10
    for epoch in tqdm.tqdm(range(1, args.epochs + 1)):
        train_loss_dict = train(model, train_loader, optimizer, args.beta, args.loss_fn)
        log_loss_dictionary(train_loss_dict, "train", epoch, metric_logger)
        test_loss_dict = test(model, epoch, test_loader, args.batch_size, metric_logger, args.beta, args.loss_fn)
        log_loss_dictionary(test_loss_dict, "test", epoch, metric_logger)
        if test_loss_dict["total_loss"] < best_loss:
            best_loss = test_loss_dict["total_loss"]
            wandb.run.summary.update({"best_loss": best_loss, "best_recon": test_loss_dict["epoch_recon"],
                                      "best_kld": test_loss_dict["epoch_kld"]})

            metric_logger.add_scalar("best_loss", best_loss,epoch)
            metric_logger.add_scalar("best_recon", test_loss_dict["epoch_recon"],epoch)
            metric_logger.add_scalar("best_kld", test_loss_dict["epoch_kld"],epoch)
            torch.save(model.state_dict(), output_dir + "/best_model.pt")

        with torch.no_grad():
            sample = torch.randn(64, layers[len(layers) // 2]).to(device)
            sample = torch.sigmoid(model.decode(sample)).cpu()
            grid = make_grid(sample.view(64, 1, 28, 28))
            # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
            image = grid.mul_(255).add_(0.5).clamp_(0, 255).to("cpu", torch.uint8)  # permute(1, 2, 0).cpu().numpy().astype(np.uint8)
            metric_logger.add_image("sample", image, epoch)

    torch.save(model.state_dict(), output_dir + "/final_model.pt")


if __name__ == "__main__":
    main()
