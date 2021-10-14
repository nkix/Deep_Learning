import os
import cv2
import shutil
import random
import tqdm


def oversampling(num, path):
    file_list = os.listdir(path)
    length = len(file_list)

    for i in tqdm(range(num)):
        rad_index = random.randint(0, length-1)

        file_name = file_list[rad_index]

        new_filename = 'ISIC_copy_' + i + '.jpg'
        shutil.copyfile(file_name, new_filename)




def image_augmentate(img_path):
    img = cv2.imread(img_path)

