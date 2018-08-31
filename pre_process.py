import os
import tarfile

from utils import ensure_folder


def extract(folder):
    filename = 'data/{}.tar.gz'.format(folder)
    print('Extracting {}...'.format(filename))
    with tarfile.open(filename) as tar:
        tar.extractall('data')


if __name__ == '__main__':
    # parameters
    ensure_folder('data')

    if not os.path.isdir('vggface2_test'):
        extract('vggface2_test')

    if not os.path.isdir('vggface2_train'):
        extract('vggface2_train')
