#!/usr/bin/env python3
import logging
import numpy as np
import os
import torch
import torch.autograd as grad
import torch.utils.data as Data
import toml_config
import model_library
from datetime import datetime
import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def initialize_CNN(x_data, y_data, tconfig):
    """ creates a model for learning temperature (without noise)

    :input_size: (int) dim of the input
    :output_size: (int) dim of the output
    :returns: a model architecture from model_library, ready to train
    """
    model = getattr(model_library, tconfig.model_name)(x_data, y_data)
    return model

def train_CNN(model, x_data, y_data, batch_size, epochs, lr, valid_split=0.2):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    loss_func = torch.nn.MSELoss(reduction="none")
    par_ct = y_data.shape[-1]
    x, y = grad.Variable(torch.from_numpy(x_data)), grad.Variable(torch.from_numpy(y_data))
    torch_dataset = Data.TensorDataset(x, y)
    dataset_size = len(torch_dataset)
    valid_size = int(valid_split * dataset_size)
    train_size = dataset_size - valid_size
    train_dataset, valid_dataset = torch.utils.data.random_split(
        torch_dataset, [train_size, valid_size])
    loader = Data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )

    valid_loader = Data.DataLoader(
        dataset=valid_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
    )
    epoch_ste = np.zeros((epochs, 2, par_ct))
    epoch_loss = np.zeros((epochs,2, par_ct))
    for epoch in range(epochs):
        for step, (batch_x, batch_y) in enumerate(loader):  # for each training step
            b_x = grad.Variable(batch_x)
            b_y = grad.Variable(batch_y)

            # input x and predict based on x
            prediction = model(b_x.float())

            # must be (1. nn output, 2. target)
            loss = loss_func(prediction, b_y.float())       # loss is already normalized with repect to batch size
            for i in range(par_ct):
                epoch_loss[epoch,0,i] += torch.mean(loss[:,:,i])
                epoch_ste[epoch, 0,i] += torch.mean(torch.abs(prediction[:,:,i] - b_y.float()[:,:,i]))
            loss = torch.mean(loss)
            optimizer.zero_grad()   # clear gradients for next train
            loss.backward()         # backpropagation, compute gradients
            optimizer.step()        # apply gradients
        epoch_loss[epoch, 0, :] /= len(list(enumerate(loader)))
        epoch_ste[epoch, 0, :] /= len(list(enumerate(loader)))

        for step, (batch_x, batch_y) in enumerate(valid_loader):
            # Validate at the end of each training epoch
            vb_x = grad.Variable(batch_x)
            vb_y = grad.Variable(batch_y)
            # input x and predict based on x
            valid_prediction = model(vb_x.float())
            # must be (1. nn output, 2. target)
            valid_loss = loss_func(valid_prediction, vb_y.float())
            for i in range(par_ct):
                epoch_loss[epoch,1,i] += torch.mean(valid_loss[:,:,i])
                epoch_ste[epoch, 1,i] += torch.mean(torch.abs(valid_prediction[:,:,i] - vb_y.float()[:,:,i]))
        epoch_loss[epoch, 1, :] /= len(list(enumerate(valid_loader)))
        epoch_ste[epoch, 1,  :] /= len(list(enumerate(valid_loader)))

        if epoch == 0:
            logger.info(f" Training on {len(list(enumerate(loader)))} batches")
            logger.info(f" Validate on {len(list(enumerate(valid_loader)))} batches")
        precision = 4
        logger.info(f"epoch {epoch}: train loss {np.round(epoch_loss[epoch, 0,:], precision)} :"
            +f"valid loss {np.round(epoch_loss[epoch, 1,:], precision)}")
        logger.info(f"          std_err train {np.round(epoch_ste[epoch, 0, :], precision)} :"
        +f" std_err valid {np.round(epoch_ste[epoch, 1, :], precision)}")
    return epoch_loss, epoch_ste

def quickload(dirname):
    x_data = np.load("./" + dirname + "/xdata.npz")
    x_data = x_data[(x_data.__dict__["files"])[0]]
    y_data = np.load("./"+ dirname + "/ydata.npy")
    while np.ndim(y_data) < 3:
        y_data = np.expand_dims(y_data, -1)
    logger.info("      Quickload successful")
    logger.info("__________________________")
    logger.info(f" x_data.shape {x_data.shape}")
    logger.info(f" y_data.shape {y_data.shape}")
    logger.info(f" __________________________\n")
    return x_data, y_data

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-v", type=int, default=3,
        help=("set logging level: 0 critical, 1 error, "
                "2 warning, 3 info, 4 debug, default=info"))
    parser.add_argument("-i", default=None, type=str,
        help="training/validation dataset")
    parser.add_argument("-e", default=None, type=int, 
        help='epochs')
    parser.add_argument("-lr", type=float, default=None, 
        help=" -log10 of learning rate (ex. 2 -> 0.01)")
    parser.add_argument("-t", type=str, default=None,
                        help="See toml_config.py")

    config = toml_config.TConfig()
    args = parser.parse_args()

    logging_translate = [logging.CRITICAL, logging.ERROR, logging.WARNING,
                         logging.INFO, logging.DEBUG]
    logger = None
    logging.basicConfig(format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                        level=logging_translate[args.v])
    logger = logging.getLogger(__name__)

    if args.t:
        config = toml_config.TConfig(args.t)
    config.model_name = "starter"
    config.translate_args(parser)
    x_data, y_data = quickload("n25_16d01m_09H44M")
    model = initialize_CNN(x_data, y_data, config)
    e_std, e_ste = train_CNN(model, x_data, y_data, 4, 20, 0.01)
if __name__ == "__main__":
    main()

"""
        now = datetime.now()
        dt_string = now.strftime("_%dd%mm_%HH%MM")
        while (os.path.isdir(outdir)):
            dt_string = now.strftime("_%dd%mm_%HH%MM")
        outdir += dt_string
"""
