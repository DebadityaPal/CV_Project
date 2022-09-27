import cv2
import numpy as np
import torch
import torchvision.models as models
from torch.nn import Module, Conv2d, LeakyReLU, BatchNorm2d, MaxPool2d, Upsample, Linear
from torchvision.models import VGG16_Weights
from torchvision import transforms
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class ConvBlock(Module):
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.conv1 = Conv2d(in_channels, out_channels,
                            kernel_size=3, padding='same')
        self.conv2 = Conv2d(out_channels, out_channels,
                            kernel_size=3, padding='same')
        self.relu = LeakyReLU()
        self.bn = BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.bn(x)
        return x


class GAN_Generator(Module):
    def __init__(self):
        super().__init__()
        # Downsampling
        self.conv1 = ConvBlock(3, 64)
        self.maxpool1 = MaxPool2d(2)
        self.conv2 = ConvBlock(64, 128)
        self.maxpool2 = MaxPool2d(2)
        self.conv3 = ConvBlock(128, 256)
        self.maxpool3 = MaxPool2d(2)
        self.conv4 = ConvBlock(256, 512)
        self.maxpool4 = MaxPool2d(2)

        # Upsampling
        self.upsample1 = Upsample(scale_factor=2)
        self.conv5 = Conv2d(512, 256, kernel_size=3, padding='same')
        self.upsample2 = Upsample(scale_factor=2)
        self.conv6 = Conv2d(256, 128, kernel_size=3, padding='same')
        self.upsample3 = Upsample(scale_factor=2)
        self.conv7 = Conv2d(128, 64, kernel_size=3, padding='same')
        self.upsample4 = Upsample(scale_factor=2)
        self.conv8 = Conv2d(64, 3, kernel_size=3, padding='same')

    def generate_attention_mask(self, image):
        # Extract illumination channel from RGB
        illumination_channel = torch.amax(image, axis=1)
        # Normalize illumination channel
        illumination_channel = illumination_channel / \
            torch.max(illumination_channel)
        # Generate attention mask (1 - illumination channel)
        attention_mask = 1 - illumination_channel
        return attention_mask

    def PILtoTensor(self, image):
        # Convert PIL image to tensor
        image = transforms.ToTensor()(image)
        image = image.unsqueeze(0)
        return image

    def TensortoPIL(self, image):
        # Convert tensor to PIL image
        image = transforms.ToPILImage()(image)
        return image

    def resize(self, img, width, height):
        # Resize attension mask
        output = transforms.Resize((width, height))(img)
        return output

    def forward(self, inputs):
        outputs = []
        for image in inputs:
            input_image = self.PILtoTensor(image).to(device)
            # Generating attention mask
            att_mask = self.generate_attention_mask(input_image)

            # Downsampling
            x = self.conv1(input_image)
            x = self.maxpool1(x)
            x = self.conv2(x)
            x = self.maxpool2(x)
            x = self.conv3(x)
            x = self.maxpool3(x)
            x = self.conv4(x)
            x = self.maxpool4(x)

            # Upsampling
            x = x * self.resize(att_mask, x.shape[2], x.shape[3])
            x = self.upsample1(x)
            x = self.conv5(x)
            x = x * self.resize(att_mask, x.shape[2], x.shape[3])
            x = self.upsample2(x)
            x = self.conv6(x)
            x = x * self.resize(att_mask, x.shape[2], x.shape[3])
            x = self.upsample3(x)
            x = self.conv7(x)
            x = x * self.resize(att_mask, x.shape[2], x.shape[3])
            x = self.upsample4(x)
            x = self.conv8(x)

            # Remove Batch Dimension
            x = torch.squeeze(x, 0)
            x = self.resize(x, input_image.shape[2], input_image.shape[3])
            # Adding input image to the output
            x = x * att_mask
            x = x + torch.squeeze(input_image, 0)

            # Converting tensor to PIL image
            x = self.TensortoPIL(x)
            outputs.append(x)

        return outputs


class GAN_Discriminator(Module):
    def __init__(self):
        super().__init__()
        self.weights = VGG16_Weights.IMAGENET1K_V1
        self.vgg16 = models.vgg16(weights=self.weights)
        self.preprocess = self.weights.transforms()
        # Set requires_grad to False
        for param in self.vgg16.parameters():
            param.requires_grad = False
        # Set classifier requires_grad to True
        for param in self.vgg16.classifier.parameters():
            param.requires_grad = True
        # Set Last layer to 1 class
        self.vgg16.classifier[6] = Linear(4096, 1)

    def process(self, image):
        image = self.preprocess(image)
        return image

    def forward(self, x):
        images = torch.stack([self.process(image) for image in x]).to(device)
        outputs = self.vgg16(images)
        return outputs
