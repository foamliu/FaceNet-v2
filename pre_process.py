import bz2
import os
import tarfile
from multiprocessing import Pool

import cv2 as cv
import mtcnn
from tqdm import tqdm

from config import img_size
from utils import ensure_folder


def extract(filename):
    print('Extracting {}...'.format(filename))
    with tarfile.open(filename) as tar:
        tar.extractall('data')


def check_one_image(filename):
    img = cv.imread(filename)
    img = img[:, :, ::-1]
    dets = detector(img, 1)

    num_faces = len(dets)
    if num_faces == 0:
        return filename

    # Find the 5 face landmarks we need to do the alignment.
    # faces = dlib.full_object_detections()
    # for detection in dets:
    #     faces.append(sp(img, detection))
    #
    # # It is also possible to get a single chip
    # image = dlib.get_face_chip(img, faces[0], size=img_size)
    # image = image[:, :, ::-1]


def check_images(usage):
    folder = os.path.join('data', usage)
    dirs = [d for d in os.listdir(folder)]
    fileset = []
    for d in dirs:
        dir = os.path.join(folder, d)
        files = [os.path.join(dir, f) for f in os.listdir(dir) if f.lower().endswith('.jpg')]
        fileset += files
    print('usage:{}, files:{}'.format(usage, len(fileset)))

    pool = Pool(12)
    results = []
    for item in tqdm(pool.imap_unordered(check_one_image, fileset), total=len(fileset)):
        results.append(item)
    pool.close()
    pool.join()

    results = [r for r in results if r is not None]
    print(len(results))
    with open('data/exclude.txt', 'w') as file:
        file.write('\n'.join(results))


if __name__ == '__main__':
    ensure_folder('data')
    ensure_folder('models')

    extract('data/vggface2_test.tar.gz')
    extract('data/vggface2_train.tar.gz')



    check_images('train')
    check_images('test')
