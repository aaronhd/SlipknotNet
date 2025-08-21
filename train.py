import datetime
import os
import sys
import argparse
import logging

# import cv2
import torch
import torch.utils.data
import torch.optim as optim
import torch.nn as nn
from tqdm import tqdm
import json
from torch.utils.tensorboard import SummaryWriter
import glob
import random
from utils.data import get_dataset
from models import get_network

logging.basicConfig(level=logging.INFO)
# CURRENT_FILE, _ = os.path.basename(__file__).split('.')
CURRENT_FILE = sys.argv[0]


# 打印模型结构和梯度状态
def print_model_and_grad(model, output_dir=None, verbose=True):
    dict_param = {k: bool(v.requires_grad) for k, v in model.named_parameters()}
    if verbose:
        print(dict_param)

    if output_dir is not None:
        json.dump(
            dict_param,
            open(os.path.join(output_dir, "param.json"), "w"),
        )


def parse_args():
    parser = argparse.ArgumentParser(description="Train slipknotnet")

    # Network
    parser.add_argument(
        "--network", type=str, default="suturenet", help="Network Name in .models"
    )

    # Dataset & Data & Training
    parser.add_argument(
        "--dataset",
        type=str,
        default="suture_line",
        help='Dataset Name ("suture_line")',
    )
    parser.add_argument(
        "--dataset-path",
        type=str,
        default="Davinci-sliputure_Dataset",
        help="Path to dataset",
    )
    parser.add_argument("--model", type=str, default=False, help="path to model")
    parser.add_argument("--gpu", type=str, default="0", help="gpu used")

    parser.add_argument(
        "--light_aug",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--split",
        type=float,
        default=0.9,
        help="Fraction of data for training (remainder is validation)",
    )
    parser.add_argument("--num-workers", type=int, default=10, help="Dataset workers")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--epochs", type=int, default=60, help="Training epochs")
    parser.add_argument(
        "--batches-per-epoch", type=int, default=1000, help="Batches per Epoch"
    )
    parser.add_argument(
        "--val-batches", type=int, default=250, help="Validation Batches"
    )

    # Logging etc.
    parser.add_argument(
        "--description", type=str, default="_", help="Training description"
    )
    parser.add_argument(
        "--outdir",
        type=str,
        default="checkpoint",
        help="Training Output Directory",
    )
    parser.add_argument("--logdir", type=str, default="./logging", help="Log directory")
    parser.add_argument(
        "--vis",
        action="store_true",
        default=False,
        help="Visualise the training process",
    )

    args = parser.parse_args()
    return args


def validate(epoch, net, device, val_data, maxepoch):
    """
    Run validation.
    :param net: Network
    :param device: Torch device
    :param val_data: Validation Dataset
    :param batches_per_epoch: Number of batches to run
    :return: Successes, Failures and Losses
    """
    net.eval()

    results = {"loss": 0, "losses": {}}

    with torch.no_grad():
        batch_idx = 0
        # while batch_idx < batches_per_epoch:
        loop = tqdm(enumerate(val_data), total=len(val_data), smoothing=0.9)
        for i, (x_image, y_mask) in loop:
            # for x_image, y_mask in val_data:
            batch_idx += 1
            # if batches_per_epoch is not None and batch_idx >= batches_per_epoch:
            #     break
            xc = x_image.to(device)
            yc = y_mask.to(device)

            lossd = net.compute_loss(xc, yc)
            loss = lossd["loss"]

            results["loss"] += loss.item()

            for ln, l in lossd["losses"].items():
                if ln not in results["losses"]:
                    results["losses"][ln] = 0
                results["losses"][ln] += l.item()
            if maxepoch:
                loop.set_description(f"Test Epoch [{epoch}/{maxepoch}]")
    results["loss"] /= batch_idx
    for l in results["losses"]:
        results["losses"][l] /= batch_idx
    return results


def train(epoch, net, device, train_data, optimizer, maxepoch, vis=False):
    """
    Run one training epoch
    :param epoch: Current epoch
    :param net: Network
    :param device: Torch device
    :param train_data: Training Dataset
    :param optimizer: Optimizer
    :param batches_per_epoch:  Data batches to train on
    :param vis:  Visualise training progress
    :return:  Average Losses for Epoch
    """
    results = {"loss": 0, "losses": {}}

    net.train()

    batch_idx = 0

    # while batch_idx < batches_per_epoch:
    loop = tqdm(enumerate(train_data), total=len(train_data), smoothing=0.9)

    for i, (x_image, y_mask) in loop:
        # for x_image, y_mask in train_data:
        batch_idx += 1
        # if batch_idx >= batches_per_epoch:
        #     break

        xc = x_image.to(device)
        yc = y_mask.to(device)

        lossd = net.compute_loss(xc, yc)

        loss = lossd["loss"]

        # if batch_idx % 100 == 0:
        #     logging.info(
        #         "Epoch: {}, Batch: {}, Loss: {:0.4f}".format(
        #             epoch, batch_idx, loss.item()
        #         )
        #     )
        loop.set_description(f"Train Epoch [{epoch}/{maxepoch}]")
        loop.set_postfix(loss=loss.item())
        results["loss"] += loss.item()
        for ln, l in lossd["losses"].items():
            if ln not in results["losses"]:
                results["losses"][ln] = 0
            results["losses"][ln] += l.item()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    results["loss"] /= batch_idx
    for l in results["losses"]:
        results["losses"][l] /= batch_idx

    return results, xc


def run():
    args = parse_args()
    print(args)
    # Set-up output directories
    # dt = datetime.datetime.now().strftime('%y%m%d_%H%M')
    timestr = str(datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))

    net_desc = "{}_{}".format(timestr, args.network) + "_".join(
        args.description.split()
    )
    pre_model = args.model
    NETWORK_NOTES = """
        Des: use suture_data containing 3 labelled packages
        Input: input RGB image. 
        Output: suture line mask
        Args: %s
        Dataset: %s
        Pre_model: %s
        Training detail:
        Train script: %s
    """ % (repr(args), args.dataset_path, pre_model, CURRENT_FILE)
    save_folder = os.path.join(args.outdir, net_desc)
    print(save_folder)
    if not os.path.exists(save_folder):
        os.makedirs(save_folder)
    writer = SummaryWriter(os.path.join(args.logdir, net_desc))

    # Load Dataset
    logging.info("Loading {} Dataset...".format(args.dataset.title()))
    Dataset = get_dataset(args.dataset)

    image_list = glob.glob(os.path.join(args.dataset_path, "input_data", "*.jpg"))
    random.shuffle(image_list)

    train_dataset = Dataset(
        args.dataset_path, start=0.0, end=args.split, image_list=image_list
    )
    train_data = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_dataset = Dataset(
        args.dataset_path,
        start=args.split,
        end=1.0,
        image_list=image_list,
        light_aug=args.light_aug,
    )
    val_data = torch.utils.data.DataLoader(
        val_dataset, batch_size=1, shuffle=False, num_workers=args.num_workers
    )
    logging.info("Done")

    # Load the network
    logging.info("Loading Network...")
    input_channels = 3
    slipknotnet = get_network(args.network)
    print("input_channels", input_channels)
    net = slipknotnet(input_channels=input_channels, output_channels=1)

    print_model_and_grad(net, save_folder, False)

    if args.model:
        print(args.model)
        net.load_state_dict(torch.load(pre_model))

    device = torch.device("cuda:" + args.gpu)
    net = net.to(device)
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    logging.info("Done")

    # Print model architecture.
    f = open(os.path.join(save_folder, "arch.txt"), "w")
    sys.stdout = f
    f.write(NETWORK_NOTES)
    f.write("\n\n")
    sys.stdout = sys.__stdout__
    f.close()

    for epoch in range(args.epochs):
        logging.info("Beginning Epoch {:02d}".format(epoch))
        train_results, xc = train(
            epoch,
            net,
            device,
            train_data,
            optimizer,
            maxepoch=args.epochs,
            vis=args.vis,
        )
        # Log training losses to tensorboard

        for n, l in train_results["losses"].items():
            writer.add_scalar("train_loss/" + n, l, epoch)
        writer.add_scalar("train_loss/loss", train_results["loss"], epoch)

        if epoch == 0:
            writer.add_graph(net, (xc))

        # Run Validation
        logging.info("Validating...")
        print("save_folder", save_folder)
        test_results = validate(epoch, net, device, val_data, args.epochs)
        print(test_results)

        for n, l in test_results["losses"].items():
            writer.add_scalar("val_loss/" + n, l, epoch)
        writer.add_scalar("val_loss/loss", test_results["loss"], epoch)

        # Save best performing network
        if epoch % 2 == 0:
            torch.save(
                net.state_dict(), os.path.join(save_folder, "epoch_%02d" % (epoch))
            )


if __name__ == "__main__":
    run()
