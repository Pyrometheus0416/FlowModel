from dataclasses import dataclass
from pathlib import Path
from statistics import mean, pstdev
#--------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.optim as optim
from torch import Tensor
from torch.utils.data import Dataset, DataLoader, Subset
from torch.amp import autocast, GradScaler

import torchvision.transforms.v2 as transforms
from torchvision.io import decode_image, write_jpeg

from torchmetrics.image.fid import FrechetInceptionDistance as FID

from tqdm import tqdm

#--------------------------------------------------------------------
from CFM_model import ConditionFlowMatching
from CFM_model import ARCH, TIMESTEP, TIME_DIM
from img_dataset import AnimeFaceDataset

from utils import EMA
#--------------------------------------------------------------------
torch.set_default_device('cpu') # 'cuda:0  [IMPORTANT!!!]
DEVICE = torch.device('cpu') # 'cuda:0'
torch.set_default_dtype(torch.float32)

#--------------------------------------------------------------------
CONTINUE = False # 是否从上次中断的地方继续训练

EPOCH = 16
BATCH_SIZE = 8
LR = 0.00005

BETAS = (0.9, 0.999)

#--------------------------------------------------------------------
IMG_FLODER = Path(r"E:\CodeHub\Mydata\AnimeFace") # [IMPORTANT!!!]
SAVE_PTH_PATH  = Path(__file__).parent / 'cnf_model.pth'
SAVE_IMG_PATH = Path(__file__).parent / 'cnf_samples'

assert IMG_FLODER.exists(), f"Image folder {IMG_FLODER} does not exist. Please check the path."
if not SAVE_PTH_PATH.exists():
    CONTINUE = False  # No checkpoint to continue
    print(f"Warning: Checkpoint {SAVE_PTH_PATH} already exists. "
          "It will be overwritten since CONTINUE is set to False.")
if not SAVE_IMG_PATH.exists():
    SAVE_IMG_PATH.mkdir(parents=True, exist_ok=True)

#--------------------------------------------------------------------

face_dataset = AnimeFaceDataset(IMG_FLODER)
# mini_face_dataset = Subset(face_dataset, list(range(128)))
print("▤The dataset capability is",len(face_dataset))

cnf_fid = FID(feature=64, reset_real_features=False)

if not CONTINUE:
    # init the fid with real dataset
    face_dataset.transform = fid_transform = transforms.Compose([
        transforms.Resize(face_dataset.size),
        transforms.Normalize(face_dataset.mean, face_dataset.std)
    ])  # temporary transform for FID evaluation (without data augmentation)
    fid_dataloader = DataLoader(
        face_dataset,
        shuffle=True,
        batch_size=BATCH_SIZE,
        drop_last=True,
        generator=torch.Generator(device=DEVICE) # [IMPORTANT!!!]
    )
    cnf_fid.reset()
    for img in tqdm(fid_dataloader, "Evaluating FID of real data"):
        cnf_fid.update(img, real=True)
    face_dataset.reset()

#--------------------------------------------------------------------

cnf = ConditionFlowMatching(ARCH, TIME_DIM, TIMESTEP)
cnf_optim = optim.Adam(cnf.parameters(), lr=LR, betas=BETAS)
scaler = GradScaler(DEVICE)
loss_logger = EMA()

dataloader = DataLoader(
        face_dataset,
        shuffle=True,
        batch_size=BATCH_SIZE,
        drop_last=True,
        generator=torch.Generator(device=DEVICE) # [IMPORTANT!!!]
    )

if CONTINUE:
    assert SAVE_PTH_PATH.exists(), "No pre-trained model detected, cannot continue training."

    print('Loading pre-trained model...')
    checkpoint: dict = torch.load(SAVE_PTH_PATH)
    cnf.load_state_dict(checkpoint['cnf'])
    cnf_optim.load_state_dict(checkpoint['cnf_optim'])
    cnf_fid.load_state_dict(checkpoint['cnf_fid'])
    print('Start training from loaded model...')


for epoch in range(EPOCH):

    cnf.train()
    for x1 in tqdm(dataloader, "Train"):
        x1: Tensor = x1.to(DEVICE)
        
        t = torch.rand(x1.shape[0], device=DEVICE)
        x0 = torch.randn_like(x1)  

        x_t = torch.lerp(x0, x1, t.view(-1, 1, 1, 1))  # path interpolation
        v_target = x1 - x0

        # Predict velocity
        with autocast(DEVICE, dtype= torch.float16):
            v_pred = cnf.velocity_predicter(x_t, t)
            loss = nn.functional.mse_loss(v_pred, v_target)
        
        loss_logger.update(loss.item())

        scaler.scale(loss).backward()
        scaler.step(cnf_optim)
        scaler.update()
        cnf_optim.zero_grad()

    cnf.eval()
    with torch.inference_mode():
        h = w = face_dataset.size
        x0_prod = cnf.sample((1,3,h,w), DEVICE)
        image = AnimeFaceDataset.inv_trans(x0_prod[0])
        # image = image.cpu() [IMPORTANT!!!]

        write_jpeg(image, SAVE_IMG_PATH/'test.jpg')

        for batch in tqdm(range(len(face_dataset) // BATCH_SIZE), "Evaluating FID of generated data"):
            x0_prod = cnf.sample((BATCH_SIZE,3,h,w), DEVICE)
            cnf_fid.update(x0_prod, real=False)
        fid_score = cnf_fid.compute().item()
        cnf_fid.reset()  # reset FID generator features for the next epoch

    checkpoint = {
        'epoch': epoch,
        'cnf': cnf.state_dict(),
        'cnf_optim': cnf_optim.state_dict(),
        'loss': loss,
        'cnf_fid': cnf_fid.state_dict(),
        # 'scheduler_state_dict': scheduler.state_dict(),
        # 'rng_state': torch.get_rng_state(),  # 可选但推荐
    }

    torch.save(checkpoint, SAVE_PTH_PATH)

    m, s = loss_logger.value, loss_logger.deviation**0.5  # mean and std of training loss
    best = loss_logger.best

    print("═══════════════════════════════════════════════════════════════════════")
    print(f"EPOCH {epoch:>3d} COMPLETE")
    print(f"|Train Loss: {m:.4f} ± {s:.4f} (Best: {best:.4f}) | Valid Loss: ---")
    print(f"|BatchSize: {BATCH_SIZE} | LR: {LR} | Checkpoint: saved")
    print("───────────────────────────────────────────────────────────────────────")
    print(f"Preview: {SAVE_IMG_PATH/'test.jpg'} | FID: {fid_score:.4f}")
    print("═══════════════════════════════════════════════════════════════════════\n")