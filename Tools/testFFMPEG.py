
# import subprocess as sp
# command = [ 'C:\\FFMPEG\\ffmpeg.exe',
#             '-i', 'test.avi',
#             '-f', 'image2pipe',
#             '-pix_fmt', 'rgb24',
#             '-vcodec', 'rawvideo', '-']
# pipe = sp.Popen(command, stdout = sp.PIPE, bufsize=10**8)

import cv2
import subprocess

input_file = 'input_file_name.mp4'
output_file = 'output_file_name.mp4'



# Initialize IP cam
username = "admin"
password = "JuLian50210809"
IP_address = "192.168.10.100"
rtsp_port = "554"
channel = "1"
subtype = "0"
# ss = "rtsp://admin:JuLian50210809@" + IP_address + ":554/cam/realmonitor?channel=1&subtype=00authbasic=YWRtaW46SnVMaWFuNTAyMTA4MDk="


doAVI = False

if doAVI:
    # ss = "E:\Electric Bird Caster\Videos\Test1.avi"
    # ss = "E:\Electric Bird Caster\Videos\Testy.avi"
    ss = "E:\Electric Bird Caster\Videos\MoveDetectAndRecognize.avi"
    # cap = cv2.VideoCapture("E:\Electric Bird Caster\Videos\sun and birds.avi")
else:
    ss = "rtsp://" + username + ":" + password + "@" + IP_address + \
         ":554/cam/realmonitor?channel=" + channel + "&subtype=" + subtype + "&unicast=true&proto=Onvif"

cap = cv2.VideoCapture(ss)
print("Video source: " + ss)

ret, frame = cap.read()
if frame is None:
    print('Not able to grab images from IP cam!')
    pass

frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
h, w, _ = frame.shape  # img_pil = img_pil.resize((int(w/3), int(h/3)))

# raw_image_shape = (h, w)  #  (height, width)

height, width, ch = frame.shape

ffmpeg = 'C:\\FFMPEG\\ffmpeg.exe'
dimension = '{}x{}'.format(width, height)
f_format = 'bgr24' # remember OpenCV uses bgr format
fps = str(cap.get(cv2.CAP_PROP_FPS))

command = [ffmpeg,
        '-f', 'rawvideo',
        '-vcodec','rawvideo',
        '-s', dimension,
        '-pix_fmt', 'rgb8',
        '-r', fps,
        '-i', '-',
        '-an',
        '-vcodec', 'mpeg4',
        '-b:v', '5000k',
        'rtmp://a.rtmp.youtube.com/live2/67qv-p2wf-uqd0-7qx6' ]

pipe = subprocess.Popen(command, stdin=subprocess.PIPE)

proc = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)

while True:
    ret, frame = cap.read()
    if not ret:
        break
    proc.stdin.write(frame.tostring())

print("v")
# # https://stackoverflow.com/questions/43991594/piping-pis-opencv-video-to-ffmpeg-for-youtube-streaming
# # #main.py
# #
# import subprocess
# import cv2
#
# cap = cv2.VideoCapture(0)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
#
# ffmpeg_path_and_exe = 'C:\\FFMPEG\\ffmpeg.exe'
#
#
# command = [ffmpeg_path_and_exe,
#             '-f', 'rawvideo',
#             '-pix_fmt', 'bgr24',
#             '-s','640x480',
#             '-i','-',
#             '-ar', '44100',
#             '-ac', '2',
#             '-acodec', 'pcm_s16le',
#             '-f', 's16le',
#             '-ac', '2',
#             '-i','/dev/zero',
#             '-acodec','aac',
#             '-ab','128k',
#             '-strict','experimental',
#             '-vcodec','h264',
#             '-pix_fmt','yuv420p',
#             '-g', '50',
#             '-vb','1000k',
#             '-profile:v', 'baseline',
#             '-preset', 'ultrafast',
#             '-r', '30',
#             '-f', 'flv',
#             'rtmp://a.rtmp.youtube.com/live2/[STREAMKEY]']
#
# pipe = subprocess.Popen(command, stdin=subprocess.PIPE)
#
# while True:
#     _, frame = cap.read()
#     pipe.stdin.write(frame.tostring())
#
# pipe.kill()
# cap.release()
