import time
from PIL import Image
import cv2
import imutils
import numpy as np
import pickle
import csv
from keras.models import load_model


class Monitor:

    def __init__(self):

        # NOTE: Camera index passed into cv2.VideoCapture() may vary based on how many
        # cameras are connected to your computer. If using a laptop with a built in camera
        # and no other cameras but the webcam, index will most likely be 1.
        self.capture = cv2.VideoCapture('monitor.m4v')

        # Configuration parameters
        self.config_img = None
        self.top_left = None
        self.bottom_right = None
        self.rois = dict() # set during calibration -> {label : [roi, number of digits expected]}
        self.roi = list()

        # Runtime parameters
        self.model = load_model('models/char74/model_1.h5') # DIGIT RECOGNITION MODEL
        self.first = True

        # Configuration parameters
        self.data_name = 'test_data'
        self.config_name = 'test_config'
        self.save_file = 'results/' + self.data_name + '.csv' # DATA FILE
        self.save_config = [True, 'configuration/' + self.config_name + '.p'] # LABEL CONFIGURATION OPTIONS

    def track(self):

        print('PRESS Q TO EXIT')
        print('\n')

        while(True):

            # Capture frame-by-frame
            received, img = self.capture.read()
            img = Image.fromarray(img)

            # Go through your ROI's and identify digits
            data = list() # all the data in the frame
            for roi in self.rois:

                # Exctract ROI information
                top_left, bottom_right = self.rois[roi][0] # roi boundary
                num_expected = self.rois[roi][1] # number of digits expected
                
                # Crop and threshold
                cropped = np.array(img.crop((top_left[0], top_left[1], bottom_right[0], bottom_right[1])))
                gray = cv2.cvtColor(cropped, cv2.COLOR_BGR2GRAY)
                thresh, bw = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

                if bw[0][0] == 255: # black on white background -> white on black background
                    bw = 255-bw

                # Contours
                contours, hierarchy = cv2.findContours(bw, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
                cv2.drawContours(cropped, contours, -1, (0,255,0), 1) # DRAW CONTOURS

                # Bounding Boxes
                sub_images = list()
                contours.sort(key=len)

                for c in contours[:num_expected]:
                    x, y, w, h = cv2.boundingRect(c)
                    cv2.rectangle(cropped, (x, y), (x+w, y+h), (0, 0, 255), 1) # DRAW BOUNDING BOXES
                    sub_images.append([x, bw[y:y+h, x:x+w].copy()]) # PREDICT ON BLACK AND WHITE IMAGES
                
                # Sort digits by left to right spatial position
                sub_images.sort()
                sub_images = [_[1] for _ in sub_images]

                # Extract individual digit values
                digits = list()
                for si in sub_images:
                    padded = cv2.copyMakeBorder(si, 10, 10, 5, 5, cv2.BORDER_CONSTANT, 0) # pad images
                    resized1 = cv2.resize(padded, (28, 28))
                    resized2 = resized1.reshape(1, 28, 28, 1)

                    # value = pytesseract.image_to_string(bw, config='outputbase digits')
                    one_hot = self.model.predict(resized2, batch_size=1) # should return a one hot array
                    one_hot = np.around(one_hot)
                    value = np.where(one_hot[0]==1)[0][0]
                    digits.append(value)
                
                # Join digits to form a number value
                joined_number = int(''.join(map(str, digits)))
                print('DETECTED: ', joined_number)
                data.append(joined_number)
                cv2.imshow('Stream', cropped)

            if self.first:
                with open(self.save_file, mode='w') as file:
                    writer = csv.writer(file, delimiter=',')
                    writer.writerow(list(self.rois.keys()))
                    self.first = False

            else:
                with open(self.save_file, mode='a') as file:
                    writer = csv.writer(file, delimiter=',')
                    writer.writerow(data)

            # Display tracking results
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # When everything done, release the capture
        self.capture.release()
        cv2.destroyAllWindows()

    def set_rois(self):

        while(True):

            # Capture frame-by-frame
            received, img = self.capture.read()
            self.config_img = img
            cv2.rectangle(self.config_img, self.top_left, self.bottom_right, (0,255, 0), 1)
            cv2.imshow('Stream', self.config_img)
            cv2.setMouseCallback('Stream', self.crop_callback)
    
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print('FINISHED LABEL ASSIGNMENTS')
                pickle.dump(self.rois, open(self.save_config[1], 'wb'))
                break

        # When everything done, release the capture
        self.capture.release()
        cv2.destroyAllWindows()

    def crop_callback(self, event, x, y, flags, param):
     
        # left mouse down
        if event == cv2.EVENT_LBUTTONDOWN:
            self.roi.append((x, y)) # top left point
            self.top_left = (x, y) # store this for visuals
            self.bottom_right = (x, y)

        # mouse is moving
        elif event == cv2.EVENT_MOUSEMOVE:
            if self.top_left != None:
                self.bottom_right = (x, y)

        # left mouse released
        elif event == cv2.EVENT_LBUTTONUP:
            self.roi.append((x, y)) # bottom right point
            label = input('Enter associated label: ')
            num_expected = int(input('Enter expected number of digits: '))
            self.rois[label] = [self.roi, num_expected]
            self.roi = list()
            self.top_left = None
            self.bottom_right = None


if __name__ == '__main__':
    M = Monitor()
    if M.save_config[0]:
        M.set_rois()
    else:
        M.rois = pickle.load(open(M.save_config[1], 'rb'))
    M.capture = cv2.VideoCapture('monitor.m4v')
    M.track()












