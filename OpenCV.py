import os
import numpy as np
import os.path
import cv2
import glob
import imutils

CAPTCHA_IMAGE_FOLDER = "generated_captcha_images"
OUTPUT_FOLDER = "extracted_letter_images"


captchaImageFiles = glob.glob(os.path.join(CAPTCHA_IMAGE_FOLDER, "*"))
counts = {}

def drawRectOnContours(img,contours):
    for contour in contours:
        [x,y,w,h] = cv2.boundingRect(contour)
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,255),1)
    img = cv2.bitwise_not(img)

    cv2.imshow('contours', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

def exportLetters(img,letterRegions,correctCaptcha):
    for boundingBox, correctLetter in zip(letterRegions, correctCaptcha):
        x,y,w,h = boundingBox
        letterImg = img[y - 2:y + h + 2, x - 2:x + w + 2]
        letterImg = cv2.bitwise_not(letterImg)

        save_path = os.path.join(OUTPUT_FOLDER, correctLetter)
         # if the output directory does not exist, create it
        if not os.path.exists(save_path):
            os.makedirs(save_path)

         # write the letter image to a file
        count = counts.get(correctLetter, 1)
        p = os.path.join(save_path, "{}.png".format(str(count).zfill(6)))
        cv2.imwrite(p, letterImg)

        # increment the count for the current key
        counts[correctLetter] = count + 1

if __name__ == "__main__":
    for i, captchaImgFile in enumerate(captchaImageFiles):
        print("{}: {}".format(i+1, captchaImgFile))

        fileName = os.path.basename(captchaImgFile)
        correctAns = os.path.splitext(fileName)[0]

        readInImage = cv2.imread(captchaImgFile)

        convertToGrey = cv2.cvtColor(readInImage, cv2.COLOR_BGR2GRAY)
        convertToGrey = cv2.copyMakeBorder(convertToGrey, 8, 8, 8, 8, cv2.BORDER_REPLICATE)
        
        thresholded = cv2.threshold(convertToGrey, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
        
        findContours = cv2.findContours(thresholded.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        findContours = findContours[1] if imutils.is_cv3() else findContours[0]

        # drawRectOnContours(thresholded.copy(),findContours)

        letterRegions = []
        for contour in findContours:
            [x,y,w,h] = cv2.boundingRect(contour)
            if w / h > 1.25:
                # This contour is too wide to be a single letter!
                # Split it in half into two letter regions!
                half_width = int(w / 2)
                letterRegions.append((x, y, half_width, h))
                letterRegions.append((x + half_width, y, half_width, h))
            else:
                # This is a normal letter by itself
                letterRegions.append((x, y, w, h))
        if len(letterRegions) != 4:
            continue
        # Sort by x value so we read in captcha left to right
        letterRegions = sorted(letterRegions, key=lambda x: x[0])
        exportLetters(thresholded,letterRegions,correctAns)







    





