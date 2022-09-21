import cv2
import numpy as np


def apply_yolo_object_detection(image_to_process):

    #Поиск и определение координат объектов на изображении


    height, width, _ = image_to_process.shape
    blob = cv2.dnn.blobFromImage(image_to_process, 1 / 255, (608, 608),
                                 (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(out_layers)
    class_indexes, class_scores, boxes = ([] for i in range(3))
    objects_count = 0

    # Начало поиска объектов на изображении
    for out in outs:
        for obj in out:
            scores = obj[5:]
            class_index = np.argmax(scores)
            class_score = scores[class_index]
            if class_score > 0:
                center_x = int(obj[0] * width)
                center_y = int(obj[1] * height)
                obj_width = int(obj[2] * width)
                obj_height = int(obj[3] * height)
                box = [center_x - obj_width // 2, center_y - obj_height // 2,
                       obj_width, obj_height]
                boxes.append(box)
                class_indexes.append(class_index)
                class_scores.append(float(class_score))

    # Выбор
    chosen_boxes = cv2.dnn.NMSBoxes(boxes, class_scores, 0.0, 0.4)
    for box_index in chosen_boxes:
        box_index = box_index
        box = boxes[box_index]
        class_index = class_indexes[box_index]

        if classes[class_index] in classes_to_look_for:
            objects_count += 1
            image_to_process = draw_object_bounding_box(image_to_process,
                                                        class_index, box)

    final_image = draw_object_count(image_to_process, objects_count)
    return final_image


def draw_object_bounding_box(image_to_process, index, box):

    #Отображение рамки вокруг объекта


    x, y, w, h = box
    start = (x, y)
    end = (x + w, y + h)
    color = (0, 0, 255)
    width = 2
    final_image = cv2.rectangle(image_to_process, start, end, color, width)

    return final_image


def draw_object_count(image_to_process, objects_count):

    #Отображение тревоги в случае обнаружения рюкзака


    start = (10, 120)
    font_size = 1.5
    font = cv2.FONT_HERSHEY_SIMPLEX
    width = 3
    text = ''

    if objects_count > 0:
        text = text + " ALARM"

    red_color = (0, 0, 255)
    black_outline_color = (0, 0, 0)
    final_image = cv2.putText(image_to_process, text, start, font, font_size,
                              black_outline_color, width * 3, cv2.LINE_AA)
    final_image = cv2.putText(final_image, text, start, font, font_size,
                              red_color, width, cv2.LINE_AA)

    return final_image


def start_video_object_detection(video: str):

    #Захват и анализ видео в режиме реального времени

    while True:
        try:

            video_camera_capture = cv2.VideoCapture(video)
            #video_camera_capture.set(cv2.cv.CV_CAP_PROP_FPS, 1)
            while video_camera_capture.isOpened():
                ret, frame = video_camera_capture.read()
                if not ret:
                    break

                frame = apply_yolo_object_detection(frame)

                frame = cv2.resize(frame, (1920 // 2, 1080 // 2))
                cv2.imshow("Video Capture", frame)
                cv2.waitKey(1)


            video_camera_capture.release()
            cv2.destroyAllWindows()

        except KeyboardInterrupt:
            pass


if __name__ == '__main__':

    net = cv2.dnn.readNetFromDarknet("Resources/yolov7.cfg",
                                     "Resources/yolov7.weights")
    layer_names = net.getLayerNames()
    out_layers_indexes = net.getUnconnectedOutLayers()
    out_layers = [layer_names[index - 1] for index in out_layers_indexes]

    with open("Resources/coco.names.txt") as file:
        classes = file.read().split("\n")

    video = ("video.mp4")
    look_for = ("backpack, handbag, suitcase").split(',')

    list_look_for = []
    for look in look_for:
        list_look_for.append(look.strip())

    classes_to_look_for = list_look_for

    start_video_object_detection(video)