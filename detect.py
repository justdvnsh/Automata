import cv2
import numpy as np
import copy
from keras.models import load_model


cap = cv2.VideoCapture(0)
cap.set(10, 200)
# parameters
cap_region_x_begin = 0.5  # start point/total width
cap_region_y_end = 0.8  # start point/total width
threshold = 60  # binary threshold
blurValue = 41  # GaussianBlur parameter
bgSubThreshold = 50
learningRate = 0

reverse_gesture_names = {0: '02_l',
 1: '01_palm',
 2: '05_thumb',
 3: '07_ok',
 4: '04_fist_moved',
 5: '06_index',
 6: '08_palm_moved',
 7: '09_c',
 8: '10_down',
 9: '03_fist'}

model = load_model('saved_model.hdf5')

def predict(image):
    image = cv2.resize(image, (224,224))
    cv2.imshow('resized', image)
    image = np.array(image, dtype='float32')
    image = np.stack((image,)*3, axis=-1)
    image = image.reshape(1,224,224,3)
    image /= 255
    pred_array = model.predict(image)
    print(f'pred_array: {pred_array}')
    result = reverse_gesture_names[np.argmax(pred_array)]
    print(f'Result: {result}')
    print(max(pred_array[0]))
    score = float("%0.2f" % (max(pred_array[0]) * 100))
    print(result)
    return result, score

while True:
    ret, frame = cap.read()

    cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                  (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)

    img = frame[0:int(cap_region_y_end * frame.shape[0]),
          int(cap_region_x_begin * frame.shape[1]):frame.shape[1]]

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (blurValue, blurValue), 0)
    # cv2.imshow('blur', blur)
    ret, thresh = cv2.threshold(blur, threshold, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    # Add prediction and action text to thresholded image
    # cv2.putText(thresh, f"Prediction: {prediction} ({score}%)", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))
    # cv2.putText(thresh, f"Action: {action}", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))  # Draw the text
    # Draw the text
    prediction, score = predict(thresh)
    cv2.putText(frame, f"Prediction: {prediction} ({score}%)", (50, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                (255, 255, 255))
    # cv2.putText(thresh, f"Action: {action}", (50, 80), cv2.FONT_HERSHEY_SIMPLEX, 1,
    #             (255, 255, 255))  # Draw the text
    cv2.imshow('ori', thresh)

    thresh1 = copy.deepcopy(thresh)
    _, contours, hierarchy = cv2.findContours(thresh1, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    length = len(contours)
    maxArea = -1
    if length > 0:
        for i in range(length):  # find the biggest contour (according to area)
            temp = contours[i]
            area = cv2.contourArea(temp)
            if area > maxArea:
                maxArea = area
                ci = i

        res = contours[ci]
        hull = cv2.convexHull(res)
        drawing = np.zeros(img.shape, np.uint8)
        cv2.drawContours(drawing, [res], 0, (0, 255, 0), 2)
        cv2.drawContours(drawing, [hull], 0, (0, 0, 255), 3)

    cv2.imshow('output', drawing)

    frame = cv2.flip(frame, 1)
    cv2.imshow('frame',frame)
    cv2.imshow('img', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

