import argparse
import cv2
import pytesseract
import os
import random

def show_image(image):
    cv2.imshow("frame", image)
    cv2.waitKey(0)

CONFIG = r'--oem 3 --psm 8 -c tessedit_char_whitelist=0123456789'
def detect_digits(image):
    digits = pytesseract.image_to_string(image, config=CONFIG)
    return digits.strip() if digits else "N/A"
    

def main():
    random_frame = random.choice(os.listdir("data/frames/"))
    image = cv2.imread("data/frames/" + random_frame)
    box = cv2.selectROI("frame", image)

    cropped = image[int(box[1]):int(box[1]+box[3]), 
                      int(box[0]):int(box[0]+box[2])]

    # random_box = random.choice(os.listdir("data/boxes/"))
    # cropped = cv2.imread("data/boxes/" + random_box)
    
    show_image(cropped)
    
    # GRAY-SCALE
    gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)

    show_image(gray)

    # DENOISING
    blur = cv2.GaussianBlur(gray,(5, 5),0)

    show_image(blur)

    # BINARIZATION
    _, binarized = cv2.threshold(blur,127,255,cv2.THRESH_OTSU)

    # binarized = cv2.adaptiveThreshold(
    # gray, 255,
    # cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
    # cv2.THRESH_BINARY_INV,
    # 21, 0)

    show_image(binarized)

    digits_og = detect_digits(cropped)
    digits_gray = detect_digits(gray)
    digits_blur = detect_digits(blur)
    digits_bin = detect_digits(binarized)
    print("Detected Number - og image:", digits_og)
    print("Detected Number - gray scale image:", digits_gray)
    print("Detected Number - blurred image:", digits_blur)
    print("Detected Number - binarized image:", digits_bin) 

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()