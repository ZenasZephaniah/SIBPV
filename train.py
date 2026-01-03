import os, math, argparse
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score
from src.config import cfg
from src.data_fetcher import download
from src.datasets import HouseDataset
from src.model import FusionModel
from src.gradcam import GradCAM
import matplotlib.pyplot as plt

def rmse(pred, true):
    return math.sqrt(mean_squared_error(true, pred))

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    total = 0

    for img, tab, y in loader:
        img = img.to(cfg.device)
        tab = tab.to(cfg.device)
        y = y.to(cfg.device)

        optimizer.zero_grad()
        pred = model(img, tab)
        loss = criterion(pred, y)
        loss.backward()
        optimizer.step()

        total += loss.item() * len(y)

    return total / len(loader.dataset)

@torch.no_grad()
def eval_model(model, loader, criterion):
    model.eval()
    ys, ps = [], []
    total = 0
    for img, tab, y in loader:
        img = img.to(cfg.device)
        tab = tab.to(cfg.device)
        y = y.to(cfg.device)
        pred = model(img, tab)
        loss = criterion(pred, y)
        total += loss.item() * len(y)
        ys.append(y.cpu().numpy()); ps.append(pred.cpu().numpy())

    ys = np.concatenate(ys); ps = np.concatenate(ps)
    return total / len(loader.dataset), rmse(ps, ys), r2_score(ys, ps)

def run_gradcam(model, val_ds):
    gc = GradCAM(model)
    os.makedirs(os.path.join(cfg.output_dir, "gradcam"), exist_ok=True)

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    for i in range(min(cfg.grad_cam_samples, len(val_ds))):
        img, tab, y = val_ds[i]
        cam = gc(img, tab)

        orig = img.permute(1, 2, 0).numpy()
        orig = (orig * std + mean) * 255
        orig = np.clip(orig, 0, 255).astype(np.uint8)

        cam_resized = cv2.resize(
            cam,
            (orig.shape[1], orig.shape[0]),
            interpolation=cv2.INTER_LINEAR
        )

        heatmap = np.uint8(cam_resized * 255)
        heatmap_color = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
        heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)

        overlay = cv2.addWeighted(orig, 0.6, heatmap_color, 0.4, 0)
        overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

        cv2.imwrite(os.path.join(cfg.output_dir, "gradcam", f"sample_{i}_gt_{y:.0f}.png"), overlay_bgr)

def main():
    print("Device from config:", cfg.device)
    print("Torch CUDA available:", torch.cuda.is_available())

    if torch.cuda.is_available():
        print("GPU name:", torch.cuda.get_device_name(0))

    os.makedirs(cfg.output_dir, exist_ok=True)
    os.makedirs(cfg.model_dir, exist_ok=True)

    # Load data
    train_df = pd.read_excel(cfg.train_xlsx)
    test_df = pd.read_excel(cfg.test_xlsx)

    # Download images
    img_paths = download(pd.concat([train_df, test_df], axis=0))

    # Filter rows with images
    train_df = train_df[train_df["id"].isin(img_paths.keys())]
    test_df = test_df[test_df["id"].isin(img_paths.keys())]

    # Scale tabular data
    scaler = StandardScaler()
    scaler.fit(train_df[cfg.tab_feats].astype(float))

    tr_df, val_df = train_test_split(train_df,test_size=cfg.val_split,random_state=cfg.seed)
    
    tr_ds = HouseDataset(tr_df, img_paths, scaler, train=True)
    val_ds = HouseDataset(val_df, img_paths, scaler, train=True)
    te_ds = HouseDataset(test_df, img_paths, scaler, train=False)

    tr_loader = DataLoader(tr_ds,batch_size=cfg.batch_size,shuffle=True,num_workers=cfg.num_workers)
    val_loader = DataLoader(val_ds,batch_size=cfg.batch_size,shuffle=False,num_workers=cfg.num_workers)
    te_loader = DataLoader(te_ds,batch_size=cfg.batch_size,shuffle=False,num_workers=cfg.num_workers)

    # Model
    model = FusionModel(tab_in=len(cfg.tab_feats)).to(cfg.device)
    optimizer = torch.optim.Adam(model.parameters(),lr=cfg.lr,weight_decay=cfg.weight_decay)
    criterion = nn.MSELoss()

    best_rmse = float("inf")

    print(f"Starting training for {cfg.epochs} epochs...")

    for epoch in range(cfg.epochs):
        tr_loss = train_one_epoch(model, tr_loader, optimizer, criterion)
        val_loss, val_rmse, val_r2 = eval_model(model, val_loader, criterion)

        print(
            f"Epoch [{epoch+1}/{cfg.epochs}] | "
            f"tr_loss {tr_loss:.4f} | "
            f"val_loss {val_loss:.4f} | "
            f"rmse {val_rmse:.4f} | "
            f"r2 {val_r2:.4f}"
        )

        if val_rmse < best_rmse:
            best_rmse = val_rmse
            torch.save({
                    "model": model.state_dict(),
                    "scaler_mean": scaler.mean_,
                    "scaler_scale": scaler.scale_
                },
                os.path.join(cfg.model_dir, "best_model.pt")
            )

    # Reload best model - FIX IS HERE: added weights_only=False
    ckpt = torch.load(os.path.join(cfg.model_dir, "best_model.pt"), map_location=cfg.device, weights_only=False)
    model.load_state_dict(ckpt["model"])

    # Predict test
    preds, ids = [], []
    model.eval()

    with torch.no_grad():
        for img, tab, pid in te_loader:
            img = img.to(cfg.device)
            tab = tab.to(cfg.device)

            pred = model(img, tab).cpu().numpy()
            preds.extend(pred.tolist())
            ids.extend(pid.tolist())
    sub = pd.DataFrame({"id": ids,"predicted_price": preds})
    sub.to_csv(os.path.join(cfg.output_dir, "submission.csv"),index=False)
    print("Saved outputs/submission.csv")

    # Grad-CAM
    run_gradcam(model, val_ds)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.parse_args()
    main()
