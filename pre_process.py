import os
import tarfile
from multiprocessing import Pool

import cv2 as cv
import dlib
from tqdm import tqdm

from config import predictor_path
from utils import ensure_folder


def extract(folder):
    filename = '{}.tar.gz'.format(folder)
    print('Extracting {}...'.format(filename))
    with tarfile.open(filename) as tar:
        tar.extractall('data')


def ensure_dlib_model():
    if not os.path.isfile(predictor_path):
        import urllib.request
        urllib.request.urlretrieve("http://dlib.net/files/shape_predictor_5_face_landmarks.dat.bz2",
                                   filename="models/shape_predictor_5_face_landmarks.dat.bz2")


def check_one_image(line):
    line = line.strip()
    if len(line) > 0:
        tokens = line.split(' ')
        image_name = tokens[0].strip()
        # print(image_name)
        filename = os.path.join(image_folder, image_name)
        # print(filename)
        img = cv.imread(filename)
        img = img[:, :, ::-1]
        dets = detector(img, 1)

        num_faces = len(dets)
        if num_faces == 0:
            return image_name

        # Find the 5 face landmarks we need to do the alignment.
        faces = dlib.full_object_detections()
        for detection in dets:
            faces.append(sp(img, detection))

        # It is also possible to get a single chip
        image = dlib.get_face_chip(img, faces[0], size=img_size)
        image = image[:, :, ::-1]


def check_images(usage):
    folder = os.path.join('data', usage)
    dirs = [d for d in os.listdir(folder)]
    fileset = []
    for d in dirs:
        dir = os.path.join(folder, d)
        files = [os.path.join(dir, f) for f in os.listdir(dir) if f.lower().endswith('.jpg')]
        fileset += files
    print('usage:{}, files:{}'.format(usage, len(fileset)))
    #
    # pool = Pool(24)
    # results = []
    # for item in tqdm(pool.imap_unordered(check_one_image, fileset), total=len(fileset)):
    #     results.append(item)
    # pool.close()
    # pool.join()
    #
    # results = [r for r in results if r is not None]
    # print(len(results))
    # with open('data/exclude.txt', 'w') as file:
    #     file.write('\n'.join(results))


if __name__ == '__main__':
    # parameters
    ensure_folder('data')
    if not os.path.isdir('data/vggface2_test'):
        extract('data/vggface2_test')
    if not os.path.isdir('data/vggface2_train'):
        extract('data/vggface2_train')
    ensure_dlib_model()

    # Load all the models we need: a detector to find the faces, a shape predictor
    # to find face landmarks so we can precisely localize the face
    detector = dlib.get_frontal_face_detector()
    sp = dlib.shape_predictor(predictor_path)

    check_images('train')
    check_images('test')
