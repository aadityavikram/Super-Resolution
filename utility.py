import os
import cv2
import time
import shutil
import zipfile


def extract(source="data/img_align_celeba.zip"):
    if not os.path.exists(source):
        print('Dataset not found')
    else:
        print('Extracting....')
        zip_file = source
        zip_ref = zipfile.ZipFile(zip_file, 'r')
        start = time.time()
        zip_ref.extractall(path="data")
        zip_ref.close()
        os.remove(zip_file)
        end = time.time()
        print('Extracted | Time Elapsed --> {} seconds'.format(end - start))


def frames_to_video(src=''):
    frame_list = []
    fps = 24
    output_path = 'data/test/video1.mp4'
    size = 0
    start = time()
    print('Creating video from frames....')
    for frame in sorted(os.listdir(src), key=len):
        img = cv2.imread(os.path.join(src, frame))
        height, width, layers = img.shape
        size = (width, height)
        frame_list.append(img)
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    for images in frame_list:
        out.write(images)
    out.release()
    end = time()
    print('{} seconds'.format(end - start))


def shorten_dataset(source='data/img_align_celeba_full', dest='data/img_align_celeba'):
    if not os.path.exists(dest):
        os.makedirs(dest)
    keep_num = 120000
    start = time.time()
    for i, files in enumerate(os.listdir(source)):
        # i starts from 0
        if i < keep_num:
            src = os.path.join(source, files)
            dst = os.path.join(dest, files)
            shutil.copyfile(src, dst)
            print('Copied {} files'.format(i + 1))
    end = time.time()
    print('Shortened Dataset | Time Elapsed --> {} seconds'.format(end - start))
