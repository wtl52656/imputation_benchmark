import numpy as np
import torch
from torch.optim import Adam
import nni
import time


def train(
    model,
    config,
    train_loader,
    valid_loader=None,
    valid_epoch_interval=5,
):
    optimizer = Adam(model.parameters(), lr=float(config["lr"]), weight_decay=1e-6)

    p1 = int(0.75 * int(config["epochs"]))
    p2 = int(0.9 * int(config["epochs"]))
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[p1, p2], gamma=0.1
    )

    best_valid_loss = 1e10
    train_start = time.time()
    print("start...")
    for epoch_no in range(int(config["epochs"])):
        avg_loss = 0
        model.train()
        for batch_no, train_batch in enumerate(train_loader):
            optimizer.zero_grad()

            loss = model(train_batch)
            loss.backward()
            avg_loss += loss.item()
            optimizer.step()

            if batch_no % 20 == 0:
                print(
                    "Train Epoch: ",
                    epoch_no,
                    "Batch: ",
                    batch_no,
                    "Loss: ",
                    loss.item(),
                )

        lr_scheduler.step()
        if valid_loader is not None and (epoch_no + 1) % valid_epoch_interval == 0:
            model.eval()
            avg_loss_valid = 0
            with torch.no_grad():
                for batch_no, valid_batch in enumerate(valid_loader):
                    loss = model(valid_batch, is_train=0)
                    avg_loss_valid += loss.item()
                    if (batch_no % 20 == 0):
                        print(
                            "Valid Epoch: ",
                            epoch_no,
                            "Batch: ",
                            batch_no,
                            "Loss: ",
                            loss.item(),
                        )
            if best_valid_loss > avg_loss_valid:
                best_valid_loss = avg_loss_valid
                print(
                    "\n best loss is updated to ",
                    avg_loss_valid / (batch_no+1),
                    "at",
                    epoch_no,
                )
    train_end_time = time.time()
    print("Training time:", train_end_time - train_start)

def evaluate(model, test_loader, _std,_mean,use_nni, nsample=10):

    test_start = time.time()
    with torch.no_grad():
        model.eval()

        all_imputed = []
        all_gt = []
        print("START TEST...")
        for batch_no, test_batch in enumerate(test_loader):
            output = model.evaluate(test_batch, nsample)

            samples, c_target, eval_points, observed_points, observed_time = output
            samples = samples.permute(0, 1, 3, 2)  # (B,nsample,L,K)
            c_target = c_target.permute(0, 2, 1)  # (B,L,K)
            eval_points = eval_points.permute(0, 2, 1).long()
            observed_points = observed_points.permute(0, 2, 1)
            samples_median = samples.median(dim=1).values

            all_gt.append(c_target[eval_points].view(-1).cpu())
            all_imputed.append(samples_median[eval_points].view(-1).cpu())

            if batch_no %50 == 0:
                print("\ntest batch:",batch_no,"/",len(test_loader),"completed", 
                      "\ntarget:",c_target[eval_points].view(-1)[:10]*_std+_mean,
                      "\nimputation:",samples_median[eval_points].view(-1)[:10]*_std+_mean)
        
        imputed = torch.cat(all_imputed, dim=0).view(-1)*_std+_mean
        truth = torch.cat(all_gt, dim=0).view(-1)*_std+_mean

        print(_std,_mean)
        print(imputed,truth)

        MAE = torch.abs(truth - imputed).mean()
        RMSE = torch.sqrt(((truth - imputed)**2).mean())
        MAPE = torch.divide(torch.abs(truth - imputed),truth).nan_to_num(posinf=0).mean()

        
        if use_nni:
            nni.report_final_result(MAE.cpu().numpy().item())
        print(f"MAE:{MAE.__format__('.6f')}      RMSE:{RMSE.__format__('.6f')}      MAPE:{MAPE.__format__('.6f')}")
    test_end_time = time.time()
    print("Testing time:", test_end_time - test_start)
