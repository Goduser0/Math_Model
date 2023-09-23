import sys
sys.path.append("2023F/Dataloader")
from data_loader import get_loader

import os
import torch
from torch import nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision.transforms as T
import matplotlib.pyplot as plt
import pandas as pd

# df_rainfall = pd.read_csv("Dataset/rainfall_path.csv")
# rain_id_unique_list = df_rainfall["rain_id"].unique().tolist()
# print(len(rain_id_unique_list))

# rain_id_num = []
# for rain_id in rain_id_unique_list:
#     df_rain_id = df_rainfall[df_rainfall["rain_id"] == rain_id]
#     rain_id_num.append(len(df_rain_id))
# df = pd.DataFrame({"rain_id": rain_id_unique_list, "num": rain_id_num})
# df.to_csv("帧数统计.csv")

