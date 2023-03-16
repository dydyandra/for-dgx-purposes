import os
import cv2
import numpy as np
import time
import peakutils
import matplotlib.pyplot as plt
from PIL import Image
from google.cloud import storage
from tempfile import NamedTemporaryFile
from pathlib import Path


def scale(img, xScale, yScale):
    res = cv2.resize(img, None, fx=xScale, fy=yScale, interpolation=cv2.INTER_AREA)
    return res


def crop(infile, height, width):
    im = Image.open(infile)
    imgwidth, imgheight = im.size
    for i in range(imgheight // height):
        for j in range(imgwidth // width):
            box = (j * width, i * height, (j + 1) * width, (i + 1) * height)
            yield im.crop(box)


def averagePixels(path):
    r, g, b = 0, 0, 0
    count = 0
    pic = Image.open(path)
    for x in range(pic.size[0]):
        for y in range(pic.size[1]):
            imgData = pic.load()
            tempr, tempg, tempb = imgData[x, y]
            r += tempr
            g += tempg
            b += tempb
            count += 1
    return (r / count), (g / count), (b / count), count

def convert_frame_to_grayscale(frame):
    grayframe = None
    gray = None
    if frame is not None:
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = scale(gray, 1, 1)
        grayframe = scale(gray, 1, 1)
        gray = cv2.GaussianBlur(gray, (9, 9), 0.0)
    return grayframe, gray

# def prepare_dirs(keyframePath, imageGridsPath, csvPath):
#     if not os.path.exists(keyframePath):
#         os.makedirs(keyframePath)
#     if not os.path.exists(imageGridsPath):
#         os.makedirs(imageGridsPath)
#     if not os.path.exists(csvPath):
#         os.makedirs(csvPath)


def plot_metrics(indices, lstfrm, lstdiffMag):
    plt.plot(indices, y[indices], "x")
    l = plt.plot(lstfrm, lstdiffMag, 'r-')
    plt.xlabel('frames')
    plt.ylabel('pixel difference')
    plt.title("Pixel value differences from frame to frame and the peak values")
    plt.show()


def keyframeDetection(filename, source, dest, Thres, plotMetrics=False, verbose=False):
    
    filename = Path(filename)
    print(filename)
    filename_wo_ext = filename.with_suffix('')
    # filename_with_jpg = filename.with_suffix('.jpg')


    # keyframePath = f"/keyFrames/{filename_wo_ext}/" 
    keyframePath = f"/keyFrames/" 


    cap = cv2.VideoCapture(source)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
  
    if (cap.isOpened()== False):
        print("Error opening video file")

    lstfrm = []
    lstdiffMag = []
    timeSpans = []
    images = []
    full_color = []
    lastFrame = None
    Start_time = time.process_time()
    
    # Read until video is completed
    for i in range(length):
        ret, frame = cap.read()
        grayframe, blur_gray = convert_frame_to_grayscale(frame)

        frame_number = cap.get(cv2.CAP_PROP_POS_FRAMES) - 1
        lstfrm.append(frame_number)
        images.append(grayframe)
        full_color.append(frame)
        if frame_number == 0:
            lastFrame = blur_gray

        diff = cv2.subtract(blur_gray, lastFrame)
        diffMag = cv2.countNonZero(diff)
        lstdiffMag.append(diffMag)
        stop_time = time.process_time()
        time_Span = stop_time-Start_time
        timeSpans.append(time_Span)
        lastFrame = blur_gray

    cap.release()
    y = np.array(lstdiffMag)
    base = peakutils.baseline(y, 2)
    indices = peakutils.indexes(y-base, Thres, min_dist=1)
    
    #plot to monitor the selected keyframe
    if (plotMetrics):
        plot_metrics(indices, lstfrm, lstdiffMag)

    cnt = 1
    for x in indices:
        with NamedTemporaryFile() as temp:
            temp_file = "".join([str(temp.name), filename])

            cv2.imwrite(temp_file, full_color[x])
            # cv2.imwrite(os.path.join(keyframePath , 'keyframe'+ str(cnt) +'.jpg'), full_color[x])
            cnt +=1
            log_message = 'keyframe ' + str(cnt) + ' happened at ' + str(timeSpans[x]) + ' sec.'


            keyframePath = keyframePath + f"/{filename_wo_ext}_{cnt}.jpg" 
            dest_blob = dest.blob(keyframePath)
            dest_blob.upload_from_filename(temp_file)

            if(verbose):
                print(log_message)

    cv2.destroyAllWindows()


def main(event, context):
    """Triggered by a change to a Cloud Storage bucket.
    Args:
         event (dict): Event payload.
         context (google.cloud.functions.Context): Metadata for the event.
    """
    file = event
    client = storage.Client()
    source_bucket = client.get_bucket(file['bucket'])
    source_blob = source_bucket.get_blob(file['name'])

    dest_bucket_name = os.environ.get('PROCESSED_BUCKET_NAME', 'Specified environment variable is not set.')
    dest_bucket = client.get_bucket(dest_bucket_name)

    # print("des")
    if source_blob.endswith('.mp4'):
        # print("yes")
        keyframeDetection(file['name'], source_blob, dest_bucket, 0.5)

