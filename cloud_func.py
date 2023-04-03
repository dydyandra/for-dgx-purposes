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


def __scale(img, xScale, yScale):
    res = cv2.resize(img, None, fx=xScale, fy=yScale, interpolation=cv2.INTER_AREA)
    return res


def __crop(infile, height, width):
    im = Image.open(infile)
    imgwidth, imgheight = im.size
    for i in range(imgheight // height):
        for j in range(imgwidth // width):
            box = (j * width, i * height, (j + 1) * width, (i + 1) * height)
            yield im.crop(box)


def __averagePixels(path):
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

def __convert_frame_to_grayscale(frame):
    grayframe = None
    gray = None
    if frame is not None:
        cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = __scale(gray, 1, 1)
        grayframe = __scale(gray, 1, 1)
        gray = cv2.GaussianBlur(gray, (9, 9), 0.0)
    return grayframe, gray


def __keyframeDetection(filename, source, dest, Thres, plotMetrics=False, verbose=False):
    
    filename = Path(str(filename))
    print(filename)
    filename_wo_ext = filename.with_suffix('')
    filename_with_jpg = filename.with_suffix('.jpg')
    print(filename_with_jpg)

    # keyframePath = f"/keyFrames/{filename_wo_ext}/" 
    folderPath = f"{filename_wo_ext}" 

    # source = np.asarray(bytearray(source.download_as_bytes()), dtype="uint8")
    source.download_to_filename('/tmp/video.mp4')
    print(source)
    cap = cv2.VideoCapture('/tmp/video.mp4')
    print(cap)
    length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print(length)
  
    if (cap.isOpened()== False):
        print("Error opening video file")

    lstfrm = []
    lstdiffMag = []
    timeSpans = []
    images = []
    full_color = []
    lastFrame = None
    Start_time = time.process_time()
    
    print("Masuk sini")
    # Read until video is completed
    for i in range(length):
        if i % fps == 0:
            cap.set(cv2.CAP_PROP_POS_FRAMES, i)
            ret, frame = cap.read()
        
            grayframe, blur_gray = __convert_frame_to_grayscale(frame)

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
        else: 
            continue

    cap.release()
    print("Release cap")
    y = np.array(lstdiffMag)
    base = peakutils.baseline(y, 2)
    indices = peakutils.indexes(y-base, Thres, min_dist=1)

    cnt = 1
    print(filename)
    for x in indices:
        with NamedTemporaryFile(suffix = '.jpg') as temp:
            filename_with_jpg = f"/{filename_wo_ext}_{cnt}.jpg"
            # temp_file = "".join([str(temp.name), str(filename_with_jpg)])
            # print(temp_file)
            # print(full_color[x])
            status = cv2.imwrite(temp.name, full_color[x])
            print(status)
            if status is True:
                print('It got uploaded')
            else:
                print("GAMASUK")
            # cv2.imwrite(os.path.join(keyframePath , 'keyframe'+ str(cnt) +'.jpg'), full_color[x])
            cnt +=1
            log_message = 'keyframe ' + str(cnt) + ' happened at ' + str(timeSpans[x]) + ' sec.'
            print(log_message)
            keyframePath = folderPath + f"/{filename_wo_ext}_{cnt}.jpg" 
            dest_blob = dest.blob(keyframePath)
            dest_blob.upload_from_filename(temp.name, content_type='image/jpeg')
            print(dest_blob)

            if(verbose):
                print(log_message)

    if os.path.exists('/tmp/video.mp4'):
        print("True")
        os.remove('/tmp/video.mp4')
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

    print(source_blob)
    if str(source_blob.name).endswith('.mp4'):
        __keyframeDetection(file['name'], source_blob, dest_bucket, 0.7)

