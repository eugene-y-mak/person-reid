from ultralytics import YOLO
import cv2
import math
from single_query import SingleQueryModel

# Change this depending on system (i.e. which port webcam is in). Probably 0 or 1
CAMERA_NUM = 0
# start webcam
cap = cv2.VideoCapture(CAMERA_NUM)
cap.set(3, 640)
cap.set(4, 480)

# model
model = YOLO("yolo-Weights/yolov8n.pt")  # v8x
classNames = ["Person"]

# reid
reid_model_name = "osnet_x0_5"
reid_weights = "pretrained/osnet_x0_5_imagenet.pth"
query_path = "query_highres.JPG"
sq_model = SingleQueryModel(reid_model_name, reid_weights, query_path)
THRESHOLD = 0.02


def main():
    i = 0
    box_color = (255, 0, 255)
    while True:
        i = i % 5
        success, img = cap.read()
        results = model(img, stream=True, classes=0, verbose=False)

        # coordinates
        for r in results:
            boxes = r.boxes
            for box in boxes:
                # bounding box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)  # convert to int values

                # do reid
               # if i == 0:
                subimage = img[y1:y2, x1:x2]
                #print(subimage.shape)
                dist = sq_model.test_matrix(subimage)
                print(dist)
               # print(f"=============================similarity: {dist}============================")
                if dist < THRESHOLD:
                    box_color = (0, 255, 0)
                else:
                    box_color = (255, 0, 255)
                # put box in cam
                cv2.rectangle(img, (x1, y1), (x2, y2), box_color, 3)

                # confidence
                confidence = math.ceil((box.conf[0] * 100)) / 100
                # print("Confidence --->", confidence)

                # class name
                cls = int(box.cls[0])
                # print("Class name -->", classNames[cls])

                # object details
                org = [x1, y1]
                font = cv2.FONT_HERSHEY_SIMPLEX
                fontScale = 1
                color = (255, 0, 0)
                thickness = 2

                cv2.putText(img, f"{classNames[cls]}, {dist:.3f}", org, font, fontScale, color, thickness)

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) == ord('q'):
            break
        i += 1

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
