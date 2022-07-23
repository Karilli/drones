import os

from PIL import Image


def convert_images_to_png(dir_path):
    for file_name in os.listdir(dir_path):
        try:
            im = Image.open(dir_path + "\\" + file_name)
            new_name = file_name.rsplit('.', 1)[0]
            im.save(dir_path + "\\" + new_name + '.png')
            print(f"Succesfully converted image: '{file_name}'.")
        except:
            print(f"'{file_name}' couldn't be converted.")


def main(dir_path):
    convert_images_to_png(dir_path)


if __name__ == "__main__":
    main("data\\images")
