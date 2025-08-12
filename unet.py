#%%
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchview import draw_graph  # pip install torchview
from torch.fx import symbolic_trace
from torch._dynamo import explain
from torchinfo import summary
import numpy as np
import sys
import random
import matplotlib.pyplot as plt
import os
from support import view_us_gt_pred_plot
from calculations import compute_centroid, calculate_f_score
print("Main module:", sys.modules['__main__'].__file__)

"""
U-Net model: one built from scratch
One in the style of nnU-Net

U-Net paper: https://arxiv.org/abs/1505.04597
nnU-Net paper: https://arxiv.org/abs/1809.10486
"""

class UNet3D_Scratch(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, features=[32, 64, 128, 256, 320]):
        super().__init__()

        self.encoder1 = nn.Sequential(
            nn.Conv3d(in_channels, features[0], kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(features[0], affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(features[0], features[0], kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(features[0], affine=True),
            nn.LeakyReLU(inplace=True)
        )
        self.pool1 = nn.MaxPool3d(kernel_size=2)

        self.encoder2 = nn.Sequential(
            nn.Conv3d(features[0], features[1], kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(features[1], affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(features[1], features[1], kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(features[1], affine=True),
            nn.LeakyReLU(inplace=True)
        )
        self.pool2 = nn.MaxPool3d(kernel_size=2)

        self.encoder3 = nn.Sequential(
            nn.Conv3d(features[1], features[2], kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(features[2], affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(features[2], features[2], kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(features[2], affine=True),
            nn.LeakyReLU(inplace=True)
        )
        self.pool3 = nn.MaxPool3d(kernel_size=2)

        self.encoder4 = nn.Sequential(
            nn.Conv3d(features[2], features[3], kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(features[3], affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(features[3], features[3], kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(features[3], affine=True),
            nn.LeakyReLU(inplace=True)
        )
        self.pool4 = nn.MaxPool3d(kernel_size=2)

        self.bottleneck = nn.Sequential(
            nn.Conv3d(features[3], features[4], kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(features[4], affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(features[4], features[4], kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(features[4], affine=True),
            nn.LeakyReLU(inplace=True)
        )

        self.upconv1 = nn.ConvTranspose3d(features[4], features[3], kernel_size=2, stride=2)
        self.decoder1 = nn.Sequential(
            nn.Conv3d(features[4], features[3], kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(features[3], affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(features[3], features[3], kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(features[3], affine=True),
            nn.LeakyReLU(inplace=True)
        )

        self.upconv2 = nn.ConvTranspose3d(features[3], features[2], kernel_size=2, stride=2)
        self.decoder2 = nn.Sequential(
            nn.Conv3d(features[3], features[2], kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(features[2], affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(features[2], features[2], kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(features[2], affine=True),
            nn.LeakyReLU(inplace=True)
        )

        self.upconv3 = nn.ConvTranspose3d(features[2], features[1], kernel_size=2, stride=2)
        self.decoder3 = nn.Sequential(
            nn.Conv3d(features[2], features[1], kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(features[1], affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(features[1], features[1], kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(features[1], affine=True),
            nn.LeakyReLU(inplace=True)
        )

        self.upconv4 = nn.ConvTranspose3d(features[1], features[0], kernel_size=2, stride=2)
        self.decoder4 = nn.Sequential(
            nn.Conv3d(features[1], features[0], kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(features[0], affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(features[0], features[0], kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(features[0], affine=True),
            nn.LeakyReLU(inplace=True)
        )

        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip1 = self.encoder1(x)

        skip2 = self.encoder2(self.pool1(skip1))

        skip3 = self.encoder3(self.pool2(skip2))

        skip4 = self.encoder4(self.pool3(skip3))
        
        x = self.bottleneck(self.pool4(skip4))

        x = self.upconv1(x)
        x = self.decoder1(torch.cat((x, skip4), dim=1))

        x = self.upconv2(x)
        x = self.decoder2(torch.cat((x, skip3), dim=1))

        x = self.upconv3(x)
        x = self.decoder3(torch.cat((x, skip2), dim=1))

        x = self.upconv4(x)
        x = self.decoder4(torch.cat((x, skip1), dim=1))

        return torch.sigmoid(self.final_conv(x))

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True),
            nn.Conv3d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.InstanceNorm3d(out_channels, affine=True),
            nn.LeakyReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)

class ConcatBlock(nn.Module):
    def __init__(self, skip_channels, up_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(skip_channels + up_channels, out_channels)

    def forward(self, skip, up):
        x = torch.cat((skip, up), dim=1)
        return self.conv(x)

class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()
        self.up = nn.ConvTranspose3d(in_channels, out_channels, kernel_size=2, stride=2)
        self.concat_block = ConcatBlock(skip_channels, out_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        if x.shape[2:] != skip.shape[2:]:
            x = F.interpolate(x, size=skip.shape[2:], mode='trilinear', align_corners=False)
        return self.concat_block(skip, x)

class DownBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.pool = nn.MaxPool3d(kernel_size=2)
        self.conv = ConvBlock(in_channels, out_channels)

    def forward(self, x):
        return self.conv(self.pool(x))

class UNet3D(nn.Module):
    def __init__(self, in_channels=1, out_channels=2, features=[32, 64, 128, 256, 320]):
        super().__init__()
        self.encoder1 = ConvBlock(in_channels, features[0])
        self.encoder2 = DownBlock(features[0], features[1])
        self.encoder3 = DownBlock(features[1], features[2])
        self.encoder4 = DownBlock(features[2], features[3])
        self.bottleneck = DownBlock(features[3], features[4])

        self.up1 = UpBlock(features[4], features[3], features[3])
        self.up2 = UpBlock(features[3], features[2], features[2])
        self.up3 = UpBlock(features[2], features[1], features[1])
        self.up4 = UpBlock(features[1], features[0], features[0])

        self.final_conv = nn.Conv3d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip1 = self.encoder1(x)
        skip2 = self.encoder2(skip1)
        skip3 = self.encoder3(skip2)
        skip4 = self.encoder4(skip3)
        bottleneck = self.bottleneck(skip4)

        x = self.up1(bottleneck, skip4)
        x = self.up2(x, skip3)
        x = self.up3(x, skip2)
        x = self.up4(x, skip1)

        x = self.final_conv(x)
        return torch.sigmoid(x)



#%%
if __name__ == "__main__":
    model = UNet3D()

    print(type(model))
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.compile(model)
    model = model.to("cuda")
    print("Type of model", type(model))
    assert hasattr(model, 'forward'), "Model is missing forward()"
    print("Model type:", type(model))
    print("Does model have forward?", hasattr(model, 'forward'))
    print("Is forward callable?", callable(getattr(model, 'forward', None)))

    # original_model = model._orig_mod
    # # for name, module in original_model.named_modules():
    # #     if isinstance(module, torch.nn.InstanceNorm3d):
    # #         print(f"{name}: affine={module.affine}, eps={module.eps}")
    # #     if isinstance(module, torch.nn.LeakyReLU):
    # #         print(f"{name}: inplace={module.inplace}, negative_slope={module.negative_slope}")

    # dummy_input = torch.randn(1, 1, 128, 128, 128)  # adjust size


    # ADAM
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    # BCE
    criterion = nn.CrossEntropyLoss()

    # LOSS VALUES
    loss_values = []

    # Create input image directories
    base = "/mnt/data2/joe/experiments/dataset0001/
    imagesTr = sorted(os.listdir(base + "input images npy/")) # folder of 3d volumes
    labelsTr = sorted(os.listdir(base + "segmentations npy (edited)/")) # folder of 3d binary masks
    print(imagesTr)
    print(labelsTr)
    print(type(torch.randn(1, 1, 128, 128, 128)))



    excluded = "2023-11-21-rec06-frame038.npy"
    imageTs = np.load(base + f"input images npy/{excluded}")
    imageTs = imageTs / np.max(imageTs) # Normalization min max
    labelTs = np.load(base + f"segmentations npy (edited)/{excluded}")



    epochs = 10

    # Training
    for epoch in range(epochs):

        
        # validate on one input image for sanity check
        choice = True
        while choice:
            file = random.choice(labelsTr)
            if file != excluded:
                choice = False

        # Load input volume
        input = np.load(base + f"input_images/{file}")
        input = input / np.max(input)
        input = torch.from_numpy(input).unsqueeze(0).unsqueeze(0).float()

        # Load binary mask
        binary_mask_np = np.load(base + f"segmentations npy (edited)/{file}").astype(bool)
        binary_mask = torch.from_numpy(binary_mask_np).unsqueeze(0).unsqueeze(0).long()
        binary_mask = binary_mask.squeeze(1)  # (1, 129, 124, 128)

        # Move to device if needed (optional). Change cuda device (default == 0)
        # input = input.to(device)
        # binary_mask = binary_mask.to(device)

        
        # Forward propagation
        optimizer.zero_grad()
        outputs = model(input)
        # print("output.shape", outputs.shape)
        loss = criterion(outputs, binary_mask)

        # Back Propagation
        loss.backward()
        optimizer.step()

        # LOGGING
        loss_values.append(loss.item())
        print(f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.4f}")
        # print(f"Output shape: {outputs.shape}")

        # Over time, see the model perform better and better on the test image
        # Validation / Test Dataset is not part of training


        predTs = model(torch.from_numpy(imageTs).unsqueeze(0).unsqueeze(0).float())
        predTs = torch.argmax(predTs, dim=1)
        predTs = predTs.squeeze(0).cpu().numpy()
        # print(np.unique(predTs))
        # print(predTs.shape)

        # View in matplotlib
        view_us_gt_pred_plot(imageTs, 
                             labelTs, 
                             predTs, 
                             center=compute_centroid(labelTs),
                             default_title_score="F-Score", 
                             verbose=True,
                             title=calculate_f_score(labelTs, predTs) # F-Score calculation between GT and Prediction
                             )

# #%%

# LOG METRICS
loss_values = np.array(loss_values)
plt.figure()
plt.plot(np.arange(len(loss_values)), loss_values, "r--o")
plt.title("Loss")
plt.show()
print("All loss values:", loss_values)
# %%
