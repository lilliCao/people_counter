"""People Counter."""
"""
 Copyright (c) 2018 Intel Corporation.
 Permission is hereby granted, free of charge, to any person obtaining
 a copy of this software and associated documentation files (the
 "Software"), to deal in the Software without restriction, including
 without limitation the rights to use, copy, modify, merge, publish,
 distribute, sublicense, and/or sell copies of the Software, and to
 permit person to whom the Software is furnished to do so, subject to
 the following conditions:
 The above copyright notice and this permission notice shall be
 included in all copies or substantial portions of the Software.
 THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
 NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE
 LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION
 OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION
 WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
"""


import os
import sys
import time
import socket
import json
import cv2
import numpy as np

import logging as log
import paho.mqtt.client as mqtt

from argparse import ArgumentParser
from inference import Network

# MQTT server environment variables
HOSTNAME = socket.gethostname()
IPADDRESS = socket.gethostbyname(HOSTNAME)
MQTT_HOST = IPADDRESS
MQTT_PORT = 3001
MQTT_KEEPALIVE_INTERVAL = 60

# classes from MobileNet
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
    "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
    "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
    "sofa", "train", "tvmonitor"]
COLOR_BB = (255, 0, 0)

def build_argparser():
    """
    Parse command line arguments.

    :return: command line arguments
    """
    parser = ArgumentParser()
    parser.add_argument("-m", "--model", required=True, type=str,
                        help="Path to an xml file with a trained model.")
    parser.add_argument("-i", "--input", required=True, type=str,
                        help="Path to image or video file")
    parser.add_argument("-l", "--cpu_extension", required=False, type=str,
                        default=None,
                        help="MKLDNN (CPU)-targeted custom layers."
                             "Absolute path to a shared library with the"
                             "kernels impl.")
    parser.add_argument("-d", "--device", type=str, default="CPU",
                        help="Specify the target device to infer on: "
                             "CPU, GPU, FPGA or MYRIAD is acceptable. Sample "
                             "will look for a suitable plugin for device "
                             "specified (CPU by default)")
    parser.add_argument("-pt", "--prob_threshold", type=float, default=0.3,
                        help="Probability threshold for detections filtering"
                        "(0.3 by default)")
    return parser


def connect_mqtt():
    '''
    Connect to mqtt server
    '''
    client = mqtt.Client()
    client.connect(MQTT_HOST, MQTT_PORT, MQTT_KEEPALIVE_INTERVAL)

    return client

def preprocessing(image, h, w):
    '''
    Preprocess the input frame to model input

    :image : frame to process
    :h : height of model input
    :w : width of model input
    :return prepo: the preprocessed frame to feed to the IE
    '''
    prepo = np.copy(image)
    prepo = cv2.resize(prepo, (w, h))
    prepo = prepo.transpose((2,0,1))
    prepo = prepo.reshape(1, 3, h, w)
    return prepo

def postprocessing(image, output, height, width, confidence_threshold):
    '''
    Postprocess result from IE to get the person in frame and the detected bounding box

    :image : frame to process
    :output : output from IE
    :height : frame height
    :width : frame width
    :confidence_threshold: threshold to filter bad detection results
    :return personCounter: number of people in current frame
            box: detected bounding box
    '''
    detections = output['detection_out']
    personCounter = 0
    box = None
    for i in np.arange(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        # filter out weak detections by ensuring the `confidence` is
        # greater than the minimum confidence
        if confidence > confidence_threshold:
            # extract the index of the class label from the `detections`,
            # then compute the (x, y)-coordinates of the bounding box for
            # the object
            idx = int(detections[0, 0, i, 1])
            box = detections[0, 0, i, 3:7] * np.array([width, height, width, height])
            (startX, startY, endX, endY) = box.astype("int")

            # display the prediction
            classStr = CLASSES[idx]
            if (classStr == "person"):
                label = "{}: {:.2f}%".format(classStr, confidence * 100)
                cv2.rectangle(image, (startX, startY), (endX, endY),
                    COLOR_BB, 2)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(image, label, (startX, y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BB, 2)
                personCounter = personCounter + 1

    return personCounter, box

def overlap(rect1, rect2):
    '''
    Define if there is overlap between 2 rectangles
    '''
    return not (rect1[0] > rect2[2] or rect1[2] < rect2[0] or rect1[3] < rect2[1] or rect1[1] > rect2[3])

def infer_on_stream(args, client):
    """
    Initialize the inference network, stream video to network,
    and output stats and video.

    :param args: Command line arguments parsed by `build_argparser()`
    :param client: MQTT client
    :return: None
    """
    # Initialise the class
    inference_network = Network()

    # Load the model and get input shape
    n, c, h, w = inference_network.load_model(args.model, args.device, args.cpu_extension)

    # Set Probability threshold for detections
    prob_threshold = args.prob_threshold

    # Handle input stream
    # Get and open video capture
    cap = cv2.VideoCapture(args.input)
    cap.open(args.input)
    # Grab the shape of input
    width = int(cap.get(3))
    height = int(cap.get(4))
    # Prepare writing output video
    fourcc = cv2.VideoWriter_fourcc(*'MPEG')
    out = cv2.VideoWriter(os.getcwd() + '/resources/output.mp4', fourcc, 24, (width,height))

    # Process frames until the video ends, or process is exited
    fCount = 0
    totalPersonCounter = 0
    totalDuration = 0
    lastCount = 0
    lastBox = None
    averageDuration = 0
    # rectangle of 5% height from right of the frame
    right_image = (int(0.95*width), 0, width,height)
    # rectangle of 5% height from bottom of the frame
    bottom_image = (0, int(0.95*height), width, height)

    print ("Starting..............................")
    while cap.isOpened():
        # Read the next frame
        flag, frame = cap.read()
        if not flag:
            break
        key_pressed = cv2.waitKey(60)

        fCount = fCount + 1

        prepo = preprocessing(frame, h, w)
        inference_network.async_inference(prepo)

        if inference_network.wait() == 0:
            result = inference_network.get_output()
            personInFrame, box = postprocessing(frame, result, height, width, prob_threshold)
            if box is not None:
                lastBox = box
            if personInFrame > lastCount and overlap(lastBox, bottom_image):
                start_time = time.time()
                totalPersonCounter = totalPersonCounter + personInFrame - lastCount
                print ("Frame #{} total={} current={} People entered".format(fCount, totalPersonCounter, personInFrame))
                # Publish new number of people in frame/total count
                client.publish("person", json.dumps({"count": personInFrame}))
                client.publish("person", json.dumps({"total": totalPersonCounter}))
                lastCount = personInFrame
            elif personInFrame < lastCount and overlap(lastBox, right_image):
                duration = int(time.time() - start_time)
                totalDuration = totalDuration + duration
                averageDuration = int(totalDuration/totalPersonCounter)
                print ("Frame #{} total={} current={} People left after {}s average={}"
                    .format(fCount, totalPersonCounter, personInFrame, duration, averageDuration))
                # Publish new average duration after a person left
                client.publish("person/duration", json.dumps({"duration": averageDuration}))
                lastCount = personInFrame

            cv2.putText(frame, "Total={} Current={} Duration={}".format(totalPersonCounter, lastCount, averageDuration), (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_BB, 1)
            out.write(frame)

        # Send to ffmpeg server
        sys.stdout.buffer.write(frame)
        sys.stdout.flush()

        # Break if escape key pressed
        if key_pressed == 27:
            break

    # Release the out writer, capture, and destroy any OpenCV windows
    out.release()
    cap.release()
    cv2.destroyAllWindows()
    client.disconnect()
    print ("Finished..............................")
    ### TODO: Write an output image if `single_image_mode` ###


def main():
    """
    Load the network and parse the output.

    :return: None
    """
    # Grab command line args
    args = build_argparser().parse_args()
    # Connect to the MQTT server
    client = connect_mqtt()
    # Perform inference on the input stream
    infer_on_stream(args, client)


if __name__ == '__main__':
    main()
