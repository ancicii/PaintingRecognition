from tkinter import *
import cv2
import numpy as np
import os
import csv
import pandas

if __name__ == "__main__":
    database_dir = 'D:\\ANIN KOMP\\Soft\\train\\'
    images = {}
    img_name = None
    match = None

    # ucitanje svih sika u "bazu"
    def load_images_to_database():
        print("Loading images...")
        for img_name in os.listdir(database_dir):
            img_path = os.path.join(database_dir, img_name)
            image = load_image(img_path)
            image = cv2.resize(image, (int(10), int(10)))
            images[img_name] = image
        print("Images successfully loaded...")

    # ucitavanje slike
    def load_image(path):
        return cv2.imread(path)

    # prikaz slike
    def display_image(image):
        cv2.namedWindow('Image preview', cv2.WINDOW_NORMAL)
        cv2.resizeWindow('Image preview', 600, 600)
        cv2.imshow('Image preview', image)
        cv2.waitKey(0)

    def find_best_match(image):
        print("Matching...")
        image = cv2.resize(image, (int(100), int(100)))
        # inicijalizujemo SIFT detektor
        sift = cv2.xfeatures2d.SIFT_create()

        # nadjemo keypointe i deskriptore pomocu SIFTa
        kp, des = sift.detectAndCompute(image, None)
        # kreiramo BFMatcher objekat

        bf = cv2.BFMatcher()
        length = 0
        best_matching_image = None

        for image_db in images:
            img_path1 = os.path.join(database_dir, image_db)
            image_from_db = load_image(img_path1)

            # nadjemo keypointe i deskriptore pomocu ORBa za sliku iz baze
            kp1, des1 = sift.detectAndCompute(image_from_db, None)
            matches = bf.knnMatch(des, des1, k=2)

            matches = [m for m, n in matches if m.distance < 0.80 * n.distance]
            if length < len(matches):
                length = len(matches)
                best_matching_image = image_from_db
                match = os.path.join(database_dir, image_db)
                img_name = image_db
        print("finished")
        print()
        get_painting_data(img_name)
        return best_matching_image


    def draw_match_points(best_match_image, my_image):
        image_height, image_width, image_depth = best_match_image.shape

        # best_match_image = cv2.resize(best_match_image, (int(300), int(300)))
        my_image = cv2.resize(my_image, (int(image_width), int(image_height)))
        sift = cv2.xfeatures2d.SIFT_create()
        kp, des = sift.detectAndCompute(my_image, None)
        kp1, des1 = sift.detectAndCompute(best_match_image, None)
        bf = cv2.BFMatcher()
        matches = bf.knnMatch(des, des1, k=2)
        matches = [m for m, n in matches if m.distance < 0.80 * n.distance]
        img_show = cv2.drawMatches(my_image, kp, best_match_image, kp1, matches, None, flags=2)
        # display_image(img_show)

    def prepare_image_to_test(path):
        img = load_image(path)
        img_gs = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # konvert u grayscale

        canny = cv2.Canny(img_gs, 200, 100)
        kernel = np.ones((9, 9), np.uint8)
        dilation = cv2.dilate(canny, kernel, iterations=1)

        _, contours, hierarchy = cv2.findContours(dilation.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        contour_to_draw = 0
        width = 0
        height = 0
        x1 = 0
        y1 = 0
        for contour in contours:  # za svaku konturu
            x, y, w, h = cv2.boundingRect(contour)
            if width * height < w * h:  # nadjemo najvecu konturu tj najveci pravougaonik
                width = w
                height = h
                x1 = x
                y1 = y
                contour_to_draw = contour

        # isolated_painting = cv2.rectangle(img.copy(), (x1, y1), (x1 + width, y1 + height), (0, 255, 0), 20)
        # display_image(isolated_painting)

        crop_img = img[y1:y1 + height, x1:x1 + width]

        best_match = find_best_match(crop_img)
        draw_match_points(best_match, crop_img)

    def get_painting_data(image):
        df = pandas.read_csv('all_data_info.csv')
        i = 0
        while i < len(df):
            if df['new_filename'][i] == image:
                print("Title of image: " + df['title'][i])
                print("Artist of image: " + df['artist'][i])
                print("Style of image: " + df['style'][i])
                break
            else:
                i = i+1

    load_images_to_database()

    getImage = input('Enter name of image from test folder: ')
    prepare_image_to_test("Test/" + getImage)
    display_image(load_image("Test/" + getImage))
    # display_image(load_image(match))
