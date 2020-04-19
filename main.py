import torch
import numpy as np
from torch import nn, optim
from srresnet import SRResNet
from load_data_gan import load_data_gan
from time import time, strftime, gmtime
from srgan import Generator, Discriminator
from load_data_resnet import load_data_resnet


def train_resnet(model=None, loader=None, criterion=None, optimizer=None, mod=500, epoch=0, device='cuda'):
    losses = []
    model.train()
    for idx, (img, label) in enumerate(loader):
        optimizer.zero_grad()
        img, label = img.to(device), label.to(device)

        # forward pass
        output = model(img)
        loss = criterion(output, label)

        # backward pass
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
        if idx % mod == 0 and idx != 0:
            print('Epoch = {} | Batch = {} | Loss = {}'.format(epoch, idx, loss))

        del img, label, output

    return losses


def train_gan(gen=None,
              disc=None,
              loader=None,
              gen_criterion=None,
              disc_criterion=None,
              gen_optimizer=None,
              disc_optimizer=None,
              mod=500,
              epoch=0,
              device='cuda'):
    gen_losses, disc_losses = [], []
    imagenet_mean = torch.FloatTensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
    imagenet_std = torch.FloatTensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)
    imagenet_mean_cuda = torch.FloatTensor([0.485, 0.456, 0.406]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
    imagenet_std_cuda = torch.FloatTensor([0.229, 0.224, 0.225]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
    gen.train()
    disc.train()
    for idx, (img, label) in enumerate(loader):
        gen_optimizer.zero_grad()
        disc_optimizer.zero_grad()
        img, label = img.to(device), label.to(device)

        # training generator

        # forward pass
        gen_output_img = gen(img)
        gen_output_img = (gen_output_img + 1.) / 2.
        if gen_output_img.ndimension() == 3:
            gen_output_img = (gen_output_img - imagenet_mean) / imagenet_std
        elif gen_output_img.ndimension() == 4:
            gen_output_img = (gen_output_img - imagenet_mean_cuda) / imagenet_std_cuda
        disc_output_img = disc(gen_output_img)
        content_loss = gen_criterion(gen_output_img, label)
        adversarial_loss = disc_criterion(disc_output_img, torch.ones_like(disc_output_img))
        gen_loss = content_loss + adversarial_loss

        # backward pass
        gen_loss.backward()
        gen_optimizer.step()

        # training discriminator

        # forward pass
        disc_output_img = disc(img.detach())
        disc_output_label = disc(label)
        img_loss = disc_criterion(disc_output_img, torch.zeros_like(disc_output_img))
        label_loss = disc_criterion(disc_output_label, torch.ones_like(disc_output_label))
        disc_loss = img_loss + label_loss

        # backward pass
        disc_loss.backward()
        disc_optimizer.step()

        gen_losses.append(gen_loss)
        disc_losses.append(disc_loss)
        if idx % mod == 0 and idx != 0:
            print('Epoch = {} | Batch = {} | Generator Loss = {} | Discriminator Loss = {}'.format(epoch, idx, gen_loss, disc_loss))

        del img, label, gen_output_img, disc_output_img, disc_output_label

    return gen_losses, disc_losses


def save_checkpoint_loss(arch, model, optimizer, path, losses, epoch):
    ckpt = {'model': arch, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(ckpt, path.format(epoch))
    np.save('losses_{}'.format(epoch), losses)


def load_checkpoint(path, epoch):
    ckpt = torch.load(path.format(epoch))
    model = ckpt['model']
    model.load_state_dict(ckpt['state_dict'])
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model


def main():
    print('Training on {}'.format(torch.cuda.get_device_name(0)))
    root = 'data/img_align_celeba'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    crop_size = 96  # crop size of target HR images
    scaling_factor = 4  # the scaling factor for the generator; the input LR images will be downsampled from the target HR images by this factor
    batch_size = 8
    iterations = 2e5
    mod = 500
    lr = 1e-4
    beta = 1e-3

    mode = 'gan'  # train SRResNet or SRGAN

    if mode == 'resnet':
        losses = []
        model_path = 'model/srresnet_{}.pt'
        model = SRResNet()
        optimizer = optim.Adam(model.parameters(), lr=lr)
        model = model.to(device)
        criterion = nn.MSELoss()
        loader = load_data_resnet(root, crop_size=crop_size, scaling_factor=scaling_factor, batch_size=batch_size)
        num_epoch = int(iterations // len(loader) + 1)
        print('Number of epochs = {}'.format(num_epoch))
        print('Size of data loader = {}'.format(len(loader)))
        start = time()
        for epoch in range(1, num_epoch + 1):
            epoch_start = time()
            losses_per_epoch = train_resnet(model=model, loader=loader, criterion=criterion, optimizer=optimizer, mod=mod, epoch=epoch, device=device)
            losses.extend(losses_per_epoch)
            save_checkpoint_loss(SRResNet(), model, optimizer, model_path, losses, epoch)
            epoch_end = time()
            est_time_left = (epoch_end - epoch_start) * num_epoch - (time() - start)
            print('Time taken in Epoch {} = {}'.format(epoch, strftime("%H:%M:%S", gmtime(epoch_end - epoch_start))))
            print('Estimated time left in Training = {}'.format(strftime("%H:%M:%S", gmtime(est_time_left))))
        end = time()
        print('Training done | Time Elapsed --> {}'.format(strftime("%H:%M:%S", gmtime(end - start))))

    elif mode == 'gan':
        gen_losses, disc_losses = [], []
        gen_path = 'model/srgan_gen_{}.pt'
        disc_path = 'model/srgan_disc_{}.pt'
        gen = Generator()
        disc = Discriminator()
        gen.load_model()
        gen_optimizer = optim.Adam(gen.parameters(), lr=lr)
        disc_optimizer = optim.Adam(disc.parameters(), lr=lr)
        gen = gen.to(device)
        disc = disc.to(device)
        gen_criterion = nn.MSELoss()
        disc_criterion = nn.BCEWithLogitsLoss()
        loader = load_data_gan(root, crop_size=crop_size, scaling_factor=scaling_factor, batch_size=batch_size)
        num_epoch = int(iterations // len(loader) + 1)
        print('Number of epochs = {}'.format(num_epoch))
        print('Size of data loader = {}'.format(len(loader)))
        start = time()
        for epoch in range(1, num_epoch + 1):
            epoch_start = time()
            gen_losses_per_epoch, disc_losses_per_epoch = train_gan(gen=gen,
                                                                    disc=disc,
                                                                    loader=loader,
                                                                    gen_criterion=gen_criterion,
                                                                    disc_criterion=disc_criterion,
                                                                    gen_optimizer=gen_optimizer,
                                                                    disc_optimizer=disc_optimizer,
                                                                    mod=mod,
                                                                    epoch=epoch,
                                                                    device=device)
            gen_losses.extend(gen_losses_per_epoch)
            disc_losses.extend(disc_losses_per_epoch)
            save_checkpoint_loss(Generator(), gen, gen_optimizer, gen_path, gen_losses, epoch)
            save_checkpoint_loss(Discriminator(), disc, disc_optimizer, disc_path, disc_losses, epoch)
            epoch_end = time()
            est_time_left = (epoch_end - epoch_start) * num_epoch - (time() - start)
            print('Time taken in Epoch {} = {}'.format(epoch, strftime("%H:%M:%S", gmtime(epoch_end - epoch_start))))
            print('Estimated time left in Training = {}'.format(strftime("%H:%M:%S", gmtime(est_time_left))))
        end = time()
        print('Training done | Time Elapsed --> {}'.format(strftime("%H:%M:%S", gmtime(end - start))))


if __name__ == '__main__':
    main()
