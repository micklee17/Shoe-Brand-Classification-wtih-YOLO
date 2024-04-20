import shutil
import glob
import random

"""script for splitting a validation set from the training set"""


if __name__ == '__main__':

    nike_to_be_moved = random.sample(glob.glob("Nike_Adidas_converse_Shoes_image_dataset/train/nike/*.jpg"), 33)

    for f in enumerate(nike_to_be_moved, 1):
        dest = "Nike_Adidas_converse_Shoes_image_dataset/val/nike"
        shutil.move(f[1], dest)


    adidas_to_be_moved = random.sample(glob.glob("Nike_Adidas_converse_Shoes_image_dataset/train/adidas/*.jpg"), 33)

    for f in enumerate(adidas_to_be_moved, 1):
        dest = "Nike_Adidas_converse_Shoes_image_dataset/val/adidas"
        shutil.move(f[1], dest)
    
    
    converse_to_be_moved = random.sample(glob.glob("Nike_Adidas_converse_Shoes_image_dataset/train/converse/*.jpg"), 33)

    for f in enumerate(converse_to_be_moved, 1):
        dest = "Nike_Adidas_converse_Shoes_image_dataset/val/converse"
        shutil.move(f[1], dest)