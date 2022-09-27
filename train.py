import argparse
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from dataloader import create_train_test_eval_sets, GANDataset
from modelling import GAN_Generator, GAN_Discriminator
import PIL


def collate(batch):
    # Load images and create batch
    images = []
    ground_truths = []
    for image, ground_truth in batch:
        image = PIL.Image.open(image)
        ground_truth = PIL.Image.open(ground_truth)
        images.append(image)
        ground_truths.append(ground_truth)
    return images, ground_truths


def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device: {}".format(device))

    train_dataset, _, val_dataset, train_labels, _, val_labels = create_train_test_eval_sets(
        exposure_errors_path=r'D:\Academics\CV_project\datasets\Exposure-Errors',
        # lol_path=r'D:\Academics\CV_project\datasets\LOL',
        # uieb_path=r'D:\Academics\CV_project\datasets\UIEB'
    )
    # Print length of datasets
    print("Train dataset length: {}".format(len(train_dataset)))
    print("Validation dataset length: {}".format(len(val_dataset)))

    # Print length of labels
    print("Train labels length: {}".format(len(train_labels)))
    print("Validation labels length: {}".format(len(val_labels)))

    train_dataset = GANDataset(train_dataset, train_labels)
    val_dataset = GANDataset(val_dataset, val_labels)

    train_dataloader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)
    val_dataloader = DataLoader(
        val_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    generator = GAN_Generator().to(device)
    discriminator = GAN_Discriminator().to(device)

    # Optimizers
    generator_optimizer = torch.optim.Adam(generator.parameters(), lr=args.lr)
    discriminator_optimizer = torch.optim.Adam(
        discriminator.parameters(), lr=args.lr)

    # Training Loop begins here
    for epoch in range(args.epochs):
        generator.train()
        discriminator.train()
        with tqdm(train_dataloader, unit='batch', leave=True, position=0) as pbar:
            for images, ground_truths in pbar:
                ##################################
                # (1) Update Discriminator network
                ##################################
                # Pass real images through discriminator
                discriminator_optimizer.zero_grad()
                real_output = discriminator(ground_truths)
                real_labels = torch.full(
                    (args.batch_size, 1), 1, dtype=torch.float, device=device)
                loss_fct = torch.nn.BCEWithLogitsLoss()
                real_loss = loss_fct(real_output, real_labels)
                real_loss.backward()

                # Create fake images
                fake_images = generator(images)
                fake_output = discriminator(fake_images)
                fake_labels = torch.full(
                    (args.batch_size, 1), 0, dtype=torch.float, device=device)
                loss_fct = torch.nn.BCEWithLogitsLoss()
                fake_loss = loss_fct(fake_output, fake_labels)
                fake_loss.backward()

                # Update discriminator
                discriminator_loss = real_loss.item() + fake_loss.item()
                discriminator_optimizer.step()

                ##################################
                # (2) Update Generator network
                ##################################
                generator_optimizer.zero_grad()
                fake_output = discriminator(fake_images)
                fake_labels = torch.full(
                    (args.batch_size, 1), 1, dtype=torch.float, device=device)
                loss_fct = torch.nn.BCEWithLogitsLoss()
                generator_discriminator_loss = loss_fct(
                    fake_output, fake_labels)
                generator_discriminator_loss.backward()
                # generator_pixelwise_loss = torch.nn.functional.mse_loss(
                #     fake_images, ground_truths)
                # generator_pixelwise_loss.backward()
                # generator_loss = generator_discriminator_loss.item() + \
                #     generator_pixelwise_loss.item()
                generator_optimizer.step()

                pbar.set_description(f'Train Epoch {epoch}')
                # pbar.set_postfix(generator_loss=generator_loss.item(
                # ), discriminator_loss=discriminator_loss.item())
                torch.cuda.empty_cache()

        generator.eval()
        discriminator.eval()
        with torch.no_grad():
            with tqdm(val_dataloader, unit='batch', leave=True, position=0) as pbar:
                for images, ground_truths in pbar:
                    fake_images = generator(images)
                    fake_labels = torch.full(
                        (args.batch_size, 1), 0, dtype=torch.float, device=device)
                    real_labels = torch.full(
                        (args.batch_size, 1), 1, dtype=torch.float, device=device)
                    # Combine real and fake images
                    combined_images = ground_truths + fake_images
                    discriminator_labels = torch.cat(
                        (real_labels, fake_labels))
                    generator_labels = torch.cat((fake_labels, real_labels))
                    # Pass through discriminator
                    combined_output = discriminator(combined_images)
                    loss_fct = torch.nn.BCEWithLogitsLoss()
                    discriminator_loss = loss_fct(
                        combined_output, discriminator_labels)
                    generator_discriminator_loss = loss_fct(
                        combined_output, generator_labels)
                    # generator_pixelwise_loss = torch.nn.functional.mse_loss(
                    #     fake_images, ground_truths)
                    # generator_loss = generator_discriminator_loss.item() + \
                    #     generator_pixelwise_loss.item()
                    pbar.set_description(f'Val Epoch {epoch}')
                    # pbar.set_postfix(generator_loss=generator_loss.item(),
                    #                  discriminator_loss=discriminator_loss.item())
                    torch.cuda.empty_cache()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()

    main(args)
