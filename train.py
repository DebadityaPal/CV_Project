import argparse
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataloader import create_train_test_eval_sets, GANDataset
from modelling import GAN_Generator, GAN_Discriminator
import cv2
from torchvision import transforms


def collate(batch):
    # Load images and create batch
    images = []
    ground_truths = []
    shapes = []
    for image, ground_truth in batch:
        image = cv2.imread(image)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        ground_truth = cv2.imread(ground_truth)
        ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2RGB)
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        ground_truth = torch.from_numpy(ground_truth).permute(2, 0, 1).float()
        images.append(image)
        ground_truths.append(ground_truth)
        shapes.append(image.shape)
    return images, ground_truths, shapes


def preprocess_vgg16(images):
    image_tensors = torch.zeros((len(images), 3, 224, 224))
    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])
    for i, image in enumerate(images):
        image = preprocess(image)
        image_tensors[i] = image
    return image_tensors


def remove_padding(images, shapes):
    output = []
    for i, image in enumerate(images):
        output.append(image[:, :shapes[i][1], :shapes[i][2]])
    return output


def pad_images(images, ground_truths):
    max_height = max([image.shape[1] for image in images])
    max_width = max([image.shape[2] for image in images])
    # Create tensor of zeros
    padded_images = torch.zeros((len(images), 3, max_height, max_width))
    padded_ground_truths = torch.zeros(
        (len(ground_truths), 3, max_height, max_width))
    for i, (image, ground_truth) in enumerate(zip(images, ground_truths)):
        padded_images[i, :, :image.shape[1], :image.shape[2]] = image
        padded_ground_truths[i, :, :ground_truth.shape[1],
                             :ground_truth.shape[2]] = ground_truth
    # Create masks
    masks = torch.zeros((len(images), 1, max_height, max_width))
    for i, (image, ground_truth) in enumerate(zip(images, ground_truths)):
        masks[i, :, :image.shape[1], :image.shape[2]] = 1
    return padded_images, padded_ground_truths, masks


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    train_dataset, _, val_dataset, train_labels, _, val_labels = create_train_test_eval_sets(
        exposure_errors_path=r'D:\Academics\CV_project\datasets\Exposure-Errors',
        lol_path=r'D:\Academics\CV_project\datasets\LOL',
        uieb_path=r'D:\Academics\CV_project\datasets\UIEB'
    )
    # Print length of datasets
    print("Train dataset length: {}".format(len(train_dataset)))
    print("Validation dataset length: {}".format(len(val_dataset)))

    # Print length of labels
    print("Train labels length: {}".format(len(train_labels)))
    print("Validation labels length: {}".format(len(val_labels)))

    train_dataset = GANDataset(train_dataset, train_labels)
    val_dataset = GANDataset(val_dataset, val_labels)

    # train_dataloader = DataLoader(
    #     train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    # val_dataloader = DataLoader(
    #     val_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    generator = GAN_Generator().to(device)
    discriminator = GAN_Discriminator().to(device)

    # Optimizers
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr)
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=args.lr)

    # Loss history
    train_generator_loss_history = []
    train_discriminator_loss_history = []
    # Training Loop begins here
    for epoch in range(args.epochs):
        generator.train()
        discriminator.train()
        avg_train_generator_loss = 0
        avg_train_discriminator_loss = 0
        with tqdm(train_dataset, unit='batch', leave=True, position=0) as pbar:
            for image, ground_truth in pbar:
                # Load images
                image = cv2.imread(image)
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                ground_truth = cv2.imread(ground_truth)
                ground_truth = cv2.cvtColor(ground_truth, cv2.COLOR_BGR2RGB)
                image = torch.from_numpy(image).permute(2, 0, 1).float()
                ground_truth = torch.from_numpy(ground_truth).permute(
                    2, 0, 1).float()
                image = image.unsqueeze(0)
                ground_truth = ground_truth.unsqueeze(0)
                ##################################
                # (1) Update Discriminator network
                ##################################
                # Pass real images through discriminator
                discriminator_optimizer.zero_grad()
                discriminator_inputs = preprocess_vgg16(
                    ground_truth).to(device)
                real_output = discriminator(discriminator_inputs)
                real_labels = torch.full(
                    (1, 1), 1, dtype=torch.float, device=device)
                loss_fct = torch.nn.BCEWithLogitsLoss()
                real_loss = loss_fct(real_output, real_labels)

                # Create fake images
                fake_image = generator(image.to(device))
                fake_output = discriminator(fake_image.detach())
                fake_labels = torch.full(
                    (1, 1), 0, dtype=torch.float, device=device)
                loss_fct = torch.nn.BCEWithLogitsLoss()
                fake_loss = loss_fct(fake_output, fake_labels)

                # Update discriminator
                discriminator_loss = real_loss + fake_loss
                discriminator_loss.backward()
                discriminator_optimizer.step()

                ##################################
                # (2) Update Generator network
                ##################################
                generator_optimizer.zero_grad()
                fake_output = discriminator(fake_image)
                fake_labels = torch.full(
                    (1, 1), 1, dtype=torch.float, device=device)
                loss_fct = torch.nn.BCEWithLogitsLoss()
                generator_discriminator_loss = loss_fct(
                    fake_output, fake_labels)
                generator_pixelwise_loss = torch.nn.functional.mse_loss(
                    fake_image, ground_truth.to(device))
                generator_loss = generator_discriminator_loss + \
                    generator_pixelwise_loss
                generator_loss.backward()
                generator_optimizer.step()

                avg_train_generator_loss += generator_loss.item() / len(train_dataset)
                avg_train_discriminator_loss += discriminator_loss.item() / len(train_dataset)

                pbar.set_description(f'Train Epoch {epoch}')
                pbar.set_postfix(generator_loss=generator_loss.item(
                ), discriminator_loss=discriminator_loss.item())

        train_generator_loss_history.append(avg_train_generator_loss)
        train_discriminator_loss_history.append(avg_train_discriminator_loss)

        generator.eval()
        discriminator.eval()
        with torch.no_grad():
            with tqdm(val_dataset, unit='batch', leave=True, position=0) as pbar:
                for image, ground_truth in pbar:
                    # Load images
                    image = cv2.imread(image)
                    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                    ground_truth = cv2.imread(ground_truth)
                    ground_truth = cv2.cvtColor(
                        ground_truth, cv2.COLOR_BGR2RGB)
                    image = torch.from_numpy(image).permute(2, 0, 1).float()
                    ground_truth = torch.from_numpy(ground_truth).permute(
                        2, 0, 1).float()
                    image = image.unsqueeze(0).to(device)
                    ground_truth = ground_truth.unsqueeze(0).to(device)
                    fake_image = generator(image)
                    fake_labels = torch.full(
                        (1, 1), 0, dtype=torch.float, device=device)
                    real_labels = torch.full(
                        (1, 1), 1, dtype=torch.float, device=device)
                    # Combine real and fake images
                    combined_images = torch.cat((ground_truth, fake_image))
                    discriminator_labels = torch.cat(
                        (real_labels, fake_labels))
                    generator_labels = torch.cat((fake_labels, real_labels))
                    # Pass through discriminator
                    discriminator_inputs = preprocess_vgg16(
                        combined_images).to(device)
                    combined_output = discriminator(discriminator_inputs)
                    loss_fct = torch.nn.BCEWithLogitsLoss()
                    discriminator_loss = loss_fct(
                        combined_output, discriminator_labels)
                    generator_discriminator_loss = loss_fct(
                        combined_output, generator_labels)
                    generator_pixelwise_loss = torch.nn.functional.mse_loss(
                        fake_image, ground_truth)
                    generator_loss = generator_discriminator_loss + \
                        generator_pixelwise_loss
                    pbar.set_description(f'Val Epoch {epoch}')
                    pbar.set_postfix(generator_loss=generator_loss.item(),
                                     discriminator_loss=discriminator_loss.item())


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    main(args)
