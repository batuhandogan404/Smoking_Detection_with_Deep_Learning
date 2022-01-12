import cv2
import numpy as np
import time

# load Yolo
net = cv2.dnn.readNet('yolov4_plus2_last.weights', 'yolov4_plus2.cfg')

classes = []

#images_path = glob.glob(r"C:\Users\batuh\PycharmProjects\smokingOpenCv\testing_data\\*.jpg")
with open("obj.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

layer_names = net.getLayerNames()
output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
colors = np.random.uniform(0, 255, size=(len(classes), 3))

#random.shuffle(images_path)
cap = cv2.VideoCapture(0)

font = cv2.FONT_HERSHEY_PLAIN
starting_time = time.time()
frame_id = 0
while True:
    _, img = cap.read()
    frame_id += 1
    height, width, channels = img.shape

    blob = cv2.dnn.blobFromImage(img, 0.00392, (320, 320), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    class_ids = []
    confidences = []
    boxes = []

    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.8:
                print(class_id)
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)
    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
    font = cv2.FONT_HERSHEY_SIMPLEX

    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            color = colors[class_ids[i]]

            # cv2.rectangle(img, (x,y), (x+ w, y + h), color, 2)
            if label == 'smoking_detection' and float(confidences[-1]) > 0.90:
                # img[y:y + h, x:x + w] = cv2.GaussianBlur(img[y:y + h, x:x + w], (51, 51), cv2.BORDER_DEFAULT)
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), -1)
            # cv2.putText(img, "{} [{:.2f}]".format(label, float(confidences[-1])), (x, y + 30), font, 0.5, color, 2)
    elapsed_time = time.time() - starting_time
    fps = frame_id / elapsed_time

    cv2.putText(img, "FPS: " + str(round(fps, 2)), (10, 50), font, 3, (1, 1, 1), 3)
    cv2.imshow("Image", img)
    cv2.waitKey(1)
cap.release()
cv2.destroyAllWindows()
