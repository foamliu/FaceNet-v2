import os
import tarfile
from multiprocessing import Pool

from PIL import Image
from tqdm import tqdm

from mtcnn.detector import detect_faces
from utils import ensure_folder


def extract(filename):
    print('Extracting {}...'.format(filename))
    with tarfile.open(filename) as tar:
        def is_within_directory(directory, target):
            
            abs_directory = os.path.abspath(directory)
            abs_target = os.path.abspath(target)
        
            prefix = os.path.commonprefix([abs_directory, abs_target])
            
            return prefix == abs_directory
        
        def safe_extract(tar, path=".", members=None, *, numeric_owner=False):
        
            for member in tar.getmembers():
                member_path = os.path.join(path, member.name)
                if not is_within_directory(path, member_path):
                    raise Exception("Attempted Path Traversal in Tar File")
        
            tar.extractall(path, members, numeric_owner=numeric_owner) 
            
        
        safe_extract(tar, "data")


def check_one_image(filename):
    img = Image.open(filename)
    bounding_boxes, landmarks = detect_faces(img)
    num_faces = len(bounding_boxes)
    if num_faces == 0:
        return filename


def check_images(usage):
    folder = os.path.join('data', usage)
    dirs = [d for d in os.listdir(folder)]
    fileset = []
    for d in dirs:
        dir = os.path.join(folder, d)
        files = [os.path.join(dir, f) for f in os.listdir(dir) if f.lower().endswith('.jpg')]
        fileset += files
    print('usage:{}, files:{}'.format(usage, len(fileset)))

    results = []
    # pool = Pool(12)
    # for item in tqdm(pool.imap_unordered(check_one_image, fileset), total=len(fileset)):
    #     results.append(item)
    # pool.close()
    # pool.join()
    # results = [r for r in results if r is not None]

    for item in tqdm(fileset):
        ret = check_one_image(item)
        if ret is not None:
            results.append(ret)

    print(len(results))
    with open('data/exclude_{}.txt'.format(usage), 'w') as file:
        file.write('\n'.join(results))


if __name__ == '__main__':
    ensure_folder('data')
    ensure_folder('models')

    extract('data/vggface2_test.tar.gz')
    extract('data/vggface2_train.tar.gz')

    check_images('train')
    check_images('test')
