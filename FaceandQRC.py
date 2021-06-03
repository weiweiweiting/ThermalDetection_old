# -*- coding:utf-8 -*-
import cv2
import time
import argparse
import numpy as np
from PIL import Image
#from keras.models import model_from_json
from utils.anchor_generator import generate_anchors
from utils.anchor_decode import decode_bbox
from utils.nms import single_class_non_max_suppression
from load_model.tensorflow_loader import load_tf_model, tf_inference

# QRcode
#from imutils.video import VideoStream
from pyzbar import pyzbar # QRCode
import datetime
import imutils
import cv2

# 多執行緒
import threading

sess, graph = load_tf_model('models/face_mask_detection.pb')
# anchor configuration
feature_map_sizes = [[33, 33], [17, 17], [9, 9], [5, 5], [3, 3]]
anchor_sizes = [[0.04, 0.056], [0.08, 0.11], [0.16, 0.22], [0.32, 0.45], [0.64, 0.72]]
anchor_ratios = [[1, 0.62, 0.42]] * 5

# generate anchors
anchors = generate_anchors(feature_map_sizes, anchor_sizes, anchor_ratios)

# for inference , the batch size is 1, the model output shape is [1, N, 4],
# so we expand dim for anchors to [1, anchor_num, 4]
anchors_exp = np.expand_dims(anchors, axis=0)

id2class = {0: 'Mask', 1: 'NoMask'}

def clear_QRCTemp(self): # 清空QRCode暫存
    global QRCTemp
    print("clear temp")
    QRCTemp = ""
    #return

def inference(image,
              conf_thresh=0.5,
              iou_thresh=0.4,
              target_shape=(160, 160),
              draw_result=True,
              show_result=True
              ):
    '''
    Main function of detection inference
    :param image: 3D numpy array of image
    :param conf_thresh: the min threshold of classification probabity.
    :param iou_thresh: the IOU threshold of NMS
    :param target_shape: the model input size.
    :param draw_result: whether to daw bounding box to the image.
    :param show_result: whether to display the image.
    :return:
    '''
    # image = np.copy(image)
    output_info = []
    height, width, _ = image.shape
    image_resized = cv2.resize(image, target_shape)
    image_np = image_resized / 255.0  # 標準化0-1
    image_exp = np.expand_dims(image_np, axis=0)
    y_bboxes_output, y_cls_output = tf_inference(sess, graph, image_exp)

    # remove the batch dimension, for batch is always 1 for inference.
    y_bboxes = decode_bbox(anchors_exp, y_bboxes_output)[0]
    y_cls = y_cls_output[0]
    # To speed up, do single class NMS, not multiple classes NMS.
    bbox_max_scores = np.max(y_cls, axis=1)
    bbox_max_score_classes = np.argmax(y_cls, axis=1)

    # keep_idx is the alive bounding box after nms.
    keep_idxs = single_class_non_max_suppression(y_bboxes,
                                                 bbox_max_scores,
                                                 conf_thresh=conf_thresh,
                                                 iou_thresh=iou_thresh,
                                                 )

    for idx in keep_idxs:
        conf = float(bbox_max_scores[idx])
        class_id = bbox_max_score_classes[idx]
        bbox = y_bboxes[idx]
        # clip the coordinate, avoid the value exceed the image boundary.
        xmin = max(0, int(bbox[0] * width))
        ymin = max(0, int(bbox[1] * height))
        xmax = min(int(bbox[2] * width), width)
        ymax = min(int(bbox[3] * height), height)

        if draw_result:
            if class_id == 0:
                color = (0, 255, 0)
            else:
                color = (255, 0, 0)
            cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color, 2)
            cv2.putText(image, "%s: %.2f" % (id2class[class_id], conf), (xmin + 2, ymin - 2),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color)
        output_info.append([class_id, conf, xmin, ymin, xmax, ymax])

    if show_result:
        Image.fromarray(image).show()
    return output_info

# 即時影像
def run_on_realtime(conf_thresh):
    # 讀入即時影像（從camera）
    cap = cv2.VideoCapture(0)
    
    # 影像設定
    height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT) # 影像高度
    width = cap.get(cv2.CAP_PROP_FRAME_WIDTH) # 影像寬度
    fps = cap.get(cv2.CAP_PROP_FPS) # 每秒顯示幾張
    fourcc = cv2.VideoWriter_fourcc(*'XVID') # 設定影片編碼方式
    total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    
    
    if not cap.isOpened(): # 影片讀取失敗
        raise ValueError("Video open failed.")
        return
    status = True
#    idx = 0
    
    # 影像讀取成功
    while status:
        # 人臉
#        start_stamp = time.time()  # 紀錄開始時間
        status, img_raw = cap.read() # 讀取的影像狀態、影像本身
        img_raw = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB) # 將影像BGR轉換成RGB
#        read_frame_stamp = time.time()
        if (status):
            inference(img_raw,
                      conf_thresh,
                      iou_thresh=0.5,
                      target_shape=(260, 260),
                      draw_result=True,
                      show_result=False)
#            cv2.imshow('image', img_raw[:, :, ::-1])
#            cv2.waitKey(1)
#            inference_stamp = time.time()
#            write_frame_stamp = time.time()
#            idx += 1
            
            
        # QRCode
        barcodes = pyzbar.decode(img_raw) # 抓取QRC
        for barcode in barcodes:
            # 得到剛剛找的QRC輪廓並繪製
            (x, y, w, h) = barcode.rect
            cv2.rectangle(img_raw, (x, y), (x + w, y + h), (0, 0, 255), 2)

            # QRC 的data是bytes object，如果要顯示網址，則要把他轉換成string
            barcodeData = barcode.data.decode("utf-8")
            barcodeType = barcode.type
            
            # 將QRC data 繪製在畫面中
            text = "{} ({})".format(barcodeData, barcodeType)
            cv2.putText(img_raw, text, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # show the output frame
#        cv2.imshow("Barcode Scanner", img_raw)

        # 顯示結果
        cv2.imshow('Face and QRC', img_raw[:, :, ::-1])
        
        # if the `q` key was pressed, break from the loop
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        
            

if __name__ == "__main__":
    run_on_realtime(conf_thresh=0.5)



