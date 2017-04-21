import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def plot_compare_two_images(image1, image2, suptitle="Heading", subtitle1="One", subtitle2="Two", is_save=False,
                            save_path="./imageCombined.png"):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 4))

    fig.suptitle(suptitle)
    # print("setting image1")
    ax1.set_title(subtitle1)
    ax1.imshow(image1)

    # print("setting image2")
    ax2.set_title(subtitle2)
    ax2.imshow(image2)

    if is_save is True:
        # print("Saving image")
        plt.savefig(save_path, bbox_inches='tight')


def plot_compare_two_image_paths(image_path1, image_path2, suptitle="Heading", subtitle1="One", subtitle2="Two",
                                 is_save=False, save_path="./imageCombined.png", use_filename_for_subtitle=False):
    image1 = mpimg.imread(image_path1)
    image2 = mpimg.imread(image_path2)

    if use_filename_for_subtitle is True:
        subtitle1 = image_path1.split("/")[-1]
        subtitle2 = image_path2.split("/")[-1]

    plot_compare_two_images(image1, image2, suptitle=suptitle, subtitle1=subtitle1, subtitle2=subtitle2,
                            is_save=is_save, save_path=save_path)


def plot_compare_three_images(image1, image2, image3, suptitle="Heading", subtitle1="One", subtitle2="Two",
                              subtitle3="Three", is_save=False,
                              save_path="./imageCombined.png"):
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(15, 4))

    fig.suptitle(suptitle)

    # print("setting image1")
    ax1.set_title(subtitle1)
    ax1.imshow(image1)

    # print("setting image2")
    ax2.set_title(subtitle2)
    ax2.imshow(image2)

    # print("setting image2")
    ax3.set_title(subtitle3)
    ax3.imshow(image3)

    if is_save is True:
        # print("Saving image")
        plt.savefig(save_path, bbox_inches='tight')


def plot_compare_three_images_from_n(image1, image2, image3, suptitle="Heading", subtitle1="One", subtitle2="Two",
                                     subtitle3="Three", is_save=False,
                                     save_path="./imageCombined.png"):
    images = [image1, image2, image3]
    subtitles = [subtitle1, subtitle2, subtitle3]
    plot_compare_n_images(images, suptitle=suptitle, subtitles=subtitles, is_save=is_save, save_path=save_path)


def plot_compare_n_images(images, suptitle="Heading", subtitles=[], is_save=False,
                          save_path="./imageCombined.png"):
    n = len(images)
    fig, tuple1 = plt.subplots(1, n, figsize=(15, 4))
    i = 0

    use_index_as_subtitle = False
    if (len(subtitles) == 0):
        use_index_as_subtitle = True

    for tup in tuple1:
        if use_index_as_subtitle is True:
            tup.set_title(str(i))
        else:
            tup.set_title(subtitles[i])

        tup.imshow(images[i])
        i += 1

    if is_save is True:
        # print("Saving image")
        plt.savefig(save_path, bbox_inches='tight')


def plot_compare_n_images_grayed(images, suptitle="Heading", subtitles=[], grayed_array=[], is_save=False,
                                 save_path="./imageCombined.png"):
    n = len(images)
    fig, tuple1 = plt.subplots(1, n, figsize=(15, 4))
    i = 0

    dont_use_subtitles_array_but_iterator = False
    if (len(subtitles) == 0):
        dont_use_subtitles_array_but_iterator = True

    for tup in tuple1:
        if dont_use_subtitles_array_but_iterator is True:
            tup.set_title(str(i))
        else:
            tup.set_title(subtitles[i])
        if grayed_array[i] is True:
            tup.imshow(images[i], cmap='gray')
        else:
            tup.imshow(images[i])

        i += 1

    if is_save is True:
        # print("Saving image")
        plt.savefig(save_path, bbox_inches='tight')
