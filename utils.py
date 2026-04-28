from dataclasses import dataclass
from typing import Iterable, TypeAlias
from pathlib import Path

from PIL import Image
import numpy as np
import torch
from torch import Tensor
from tqdm import tqdm

#--------------------------------------------------------------------
ImgLoader: TypeAlias = Iterable[Tensor]

#--------------------------------------------------------------------
@dataclass
class WelfordStats:
    n: int = 0
    mean: Tensor = 0.0
    M2: Tensor = 0.0

    def calculate(self, loader: ImgLoader):
        """
        ```mean_AB = n*mean_A + B*mean_B
        = mean_A + (mean_B - mean_A) * B / (n+B)
        = mean_B + (mean_A - mean_B) * n / (n+B)
        = mean_A + (mean_B - mean_AB) * B / n
        = mean_B + (mean_A - mean_AB) * n / B
        ```
        """
        
        for images in tqdm(loader, "online calculate"):
            B, C = images.size(0), images.size(1)
            images = images.permute((1,0,2,3)).flatten(1)
            mean_B = images.mean(1)
            m2_B = (B-1) * images.var(1)

            new_n = self.n + B
            delta = mean_B - self.mean
            self.mean += delta * B / new_n  # update mean
            self.M2 += m2_B + (delta * delta) * self.n * B / new_n  # update m2
            self.n = new_n  # update counter
            
    @property
    def variance(self):
        if self.n < 2:
            return float('nan')
        return self.M2 / (self.n - 1)
    
    @property
    def pvariance(self):
        if self.n < 2:
            return float('nan')
        return self.M2 / self.n

    @property
    def stdev(self):
        return self.variance.sqrt()
    
    @property
    def pstdev(self):
        return self.pvariance.sqrt()


@dataclass
class EMA:
    t: int = 0
    decay: float = 0.999
    value: float = 0.0
    deviation: float = 0.0
    best: float = float('inf')

    def update(self, new_value: Tensor):
        new_value = new_value.detach()  # Ensure the new value is detached from the computation graph
        self.value = torch.lerp(self.value, new_value, 1 - self.decay)  # equivalent to the above line but more numerically stable
        self.deviation = (self.value - new_value)**2  # MSE error
        self.best = min(self.best, self.value)
        self.t += 1

    def reset(self):
        self.value = 0.0


def compute_average_image(image_folder: Path):

    image_files = image_folder.glob("[0-9]*-[0-9]*.jpg")

    base_array = np.zeros((256,256,3), dtype=np.float32)
    cnt = 0

    for img_path in image_files:
        width = int(img_path.name[0])
        if 4 <= width <= 7:
            continue

        img = Image.open(img_path)
        img = img.resize((256,256))
        base_array += np.array(img, dtype=np.float32)
        cnt += 1
        print(cnt, img_path)
            
    average_array = base_array / cnt

    average_image = Image.fromarray(np.uint8(average_array))
    
    return average_image


#--------------------------------------------------------------------
if __name__ == "__main__":
    src = Path(r"E:\CodeHub\Mydata\AnimeFace")


    avg_img = compute_average_image(src)
    if avg_img:
        avg_img.show()