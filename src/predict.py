import time

import hydra
import pytorch_lightning as pl
from omegaconf import open_dict
from argparse import ArgumentParser

from model.lcdpnet import LitModel as ModelClass
from tqdm import tqdm
from data.img_dataset import ColorCorrectionDataset
from globalenv import *
from utils.util import parse_config

pl.seed_everything(GLOBAL_SEED)


def parse_args():
    args = ArgumentParser()
    args.add_argument("--ckpt", type=str, default=None)
    args.add_argument("--img_dir", type=str, default=None)
    args.add_argument("--out_dir", type=str, default=None)
    return args.parse_args()


@hydra.main(config_path="conf", config_name="config")
def main(opt):
    args = parse_args()
    opt = parse_config(opt, TEST)
    print("Running config:", opt)

    model = ModelClass.load_from_checkpoint(args.ckpt, opt=opt)

    print(f"Loading model from: {args.ckpt}")

    predict_ds = ColorCorrectionDataset(
        image_dir=args.image_dir, image_size=args.image_size
    )
    predict_dl = torch.utils.data.DataLoader(
        predict_ds, batch_size=1, shuffle=False, num_workers=20
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    pbar = tqdm(enumerate(predict_dl), total=len(predict_dl))
    for bx, data in pbar:
        model.eval()

        x, image_id = data
        (x,) = x.to(device, dtype=torch.float)
        y_pred = model(x)
        y_pred = y_pred.unsqueeze(0)

        torchvision.utils.save_image(y_pred, f"{args.out_dir}/{image_id}")


if __name__ == "__main__":
    main()
