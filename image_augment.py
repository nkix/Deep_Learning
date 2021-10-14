import cv2
import pandas as pd
from tqdm import tqdm
import shutil


train = pd.read_csv("C:/Users/Nrx03/Desktop/deep_learning/proj/ISIC_2019_Training_set.csv")
train['image'] += '.jpg'

test = pd.read_csv("C:/Users/Nrx03/Desktop/deep_learning/proj/Test_set.csv")
test['image'] += '.jpg'

test_path = 'C:/Users/Nrx03/Desktop/deep_learning/proj/augmented_data/test/'
train_path = 'C:/Users/Nrx03/Desktop/deep_learning/proj/augmented_data/train/'


def process_img(path):
    img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
    h, w, d = img.shape
    img = cv2.resize(img, (int(w / 2), int(h / 2)), interpolation=cv2.INTER_AREA)

    return img


"""for file in tqdm(train['image']):
    file_path = train_path + file
    img = process_img("C:/Users/Nrx03/Desktop/deep_learning/proj/ISIC_2019_Training_Input/" + file)
    cv2.imwrite(file_path, img)

for file in tqdm(test['image']):
    file_path = test_path + file
    img = process_img("C:/Users/Nrx03/Desktop/deep_learning/proj/Test_set/" + file)
    cv2.imwrite(file_path, img)"""


def train_file_repack():
    # 'MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC'
    MEL = []
    NV = []
    BCC = []
    AK = []
    BKL = []
    DF = []
    VASC = []
    SCC = []

    for i in tqdm(range(len(train['image']))):
        img = train['image'][i]
        if train['MEL'][i] == 1:
            MEL.append(img)
        elif train['NV'][i] == 1:
            NV.append(img)
        elif train['BCC'][i] == 1:
            BCC.append(img)
        elif train['AK'][i] == 1:
            AK.append(img)
        elif train['BKL'][i] == 1:
            BKL.append(img)
        elif train['DF'][i] == 1:
            DF.append(img)
        elif train['VASC'][i] == 1:
            VASC.append(img)
        else:
            SCC.append(img)

    file_move(MEL, train_path+'/MEL', train_path)
    file_move(NV, train_path + '/NV', train_path)
    file_move(BCC, train_path + '/BCC', train_path)
    file_move(AK, train_path + '/AK', train_path)
    file_move(BKL, train_path + '/BKL', train_path)
    file_move(DF, train_path + '/DF', train_path)
    file_move(VASC, train_path + '/VASC', train_path)
    file_move(SCC, train_path + '/SCC', train_path)


def test_file_repack():
    # 'MEL', 'NV', 'BCC', 'AK', 'BKL', 'DF', 'VASC', 'SCC'
    MEL = []
    NV = []
    BCC = []
    AK = []
    BKL = []
    DF = []
    VASC = []
    SCC = []

    for i in tqdm(range(len(test['image']))):
        img = test['image'][i]
        if test['MEL'][i] == 1:
            MEL.append(img)
        elif test['NV'][i] == 1:
            NV.append(img)
        elif test['BCC'][i] == 1:
            BCC.append(img)
        elif test['AK'][i] == 1:
            AK.append(img)
        elif test['BKL'][i] == 1:
            BKL.append(img)
        elif test['DF'][i] == 1:
            DF.append(img)
        elif test['VASC'][i] == 1:
            VASC.append(img)
        else:
            SCC.append(img)

    file_move(MEL, test_path + '/MEL', test_path)
    file_move(NV, test_path + '/NV', test_path)
    file_move(BCC, test_path + '/BCC', test_path)
    file_move(AK, test_path + '/AK', test_path)
    file_move(BKL, test_path + '/BKL', test_path)
    file_move(DF, test_path + '/DF', test_path)
    file_move(VASC, test_path + '/VASC', test_path)
    file_move(SCC, test_path + '/SCC', test_path)


def file_move(file_list, despath, rootpath):
    for file in file_list:
        full_path = rootpath + '/' + file
        shutil.move(full_path, despath)


test_file_repack()

