import torch
from PIL import Image
from srresnet import SRResNet
from torchvision.transforms import ToTensor, ToPILImage


def load_checkpoint(path, device):
    ckpt = torch.load(path)
    model = ckpt['model']
    model.load_state_dict(ckpt['state_dict'])
    model.to(device)
    for parameter in model.parameters():
        parameter.requires_grad = False
    model.eval()
    return model


def main():
    device = 'cuda'
    imagenet_mean = torch.FloatTensor([0.485, 0.456, 0.406]).unsqueeze(1).unsqueeze(2)
    imagenet_std = torch.FloatTensor([0.229, 0.224, 0.225]).unsqueeze(1).unsqueeze(2)
    imagenet_mean_cuda = torch.FloatTensor([0.485, 0.456, 0.406]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
    imagenet_std_cuda = torch.FloatTensor([0.229, 0.224, 0.225]).to(device).unsqueeze(0).unsqueeze(2).unsqueeze(3)
    model_resnet = load_checkpoint('model/srresnet_14.pt', device)
    model_gan = load_checkpoint('model/srgan_gen.pt', device)
    img = Image.open('data/test_data/dmcv.jpg', mode='r')
    # model.eval()
    img = img.convert('RGB')
    if (img.width >= 1920 and img.width < 3840) or (img.height >= 1080 and img.height < 2160):
        img = img.resize((img.width // 2, img.height // 2), Image.BICUBIC)
    elif img.width >= 3840 or img.height >= 2160:
        img = img.resize((img.width // 4, img.height // 4), Image.BICUBIC)
    output = img.resize((img.width // 4, img.height // 4), Image.BICUBIC)
    resized_output = output.resize((img.width, img.height), Image.BICUBIC)
    transform = ToTensor()
    lr_img = transform(output)
    if lr_img.ndimension() == 3:
        lr_img = (lr_img - imagenet_mean) / imagenet_std
    elif lr_img.ndimension() == 4:
        lr_img = (lr_img - imagenet_mean_cuda) / imagenet_std_cuda
    # with torch.no_grad():
    sr_resnet_output = model_resnet(lr_img.unsqueeze(0).to(device))
    # sr_gan_output = model_gan(lr_img.unsqueeze(0).to(device))

    transform_1 = ToPILImage()

    sr_resnet_output = sr_resnet_output.squeeze(0).cpu().detach()
    sr_resnet_output = (sr_resnet_output + 1.) / 2.
    sr_resnet_output = transform_1(sr_resnet_output)

    '''
    sr_gan_output = sr_gan_output.squeeze(0).cpu().detach()
    sr_gan_output = (sr_gan_output + 1.) / 2.
    sr_gan_output = transform_1(sr_gan_output)
    '''

    grid_image = Image.new('RGB', (2 * img.width, img.height))
    x_offset = 0
    grid_image.paste(resized_output, (x_offset, 0))
    x_offset += img.width
    grid_image.paste(sr_resnet_output, (x_offset, 0))
    # x_offset += img.width
    # grid_image.paste(sr_gan_output, (x_offset, 0))
    grid_image.save('result/output.jpg')


if __name__ == '__main__':
    main()
