import cv2
from matplotlib import pyplot as plt
# cap = cv2.VideoCapture("rtsp://admin:JuLian50210809@192.168.1.108:37777")

# The format of the RTSP information will be as follows:
# "rtsp://UserName:Password@IPAddress/cam/realmonitor?channel=1&subtype=0
# be sure to change "UserName:Password@IPAddress".


username = "admin"
password = "JuLian50210809"
IP_address = "192.168.10.100"
rtsp_port = "554"
channel = "1"
subtype = "0"
ss = "rtsp://admin:JuLian50210809@" + IP_address + ":554/cam/realmonitor?channel=1&subtype=00authbasic=YWRtaW46SnVMaWFuNTAyMTA4MDk="

ss= "rtsp://" + username + ":" + password + "@" + IP_address + \
        ":554/cam/realmonitor?channel=" + channel + "&subtype=" + subtype
print(ss)
cap = cv2.VideoCapture(ss)
# rtsp://admin:12345@10.202.189.201:554/h264Preview_01_main
fig = plt.figure(figsize=(18, 8))
# ax1 = fig.add_subplot(1, 1, 1)

while 1 == 1:
    ret, frame=cap.read()

    if frame is None:
        print('Frame was None')
    else:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        cv2.imshow("Mokker", frame)
        # # Plot snippet
        # plt.cla()
        # plt.imshow(frame.astype(int))
        # plt.show()
        # plt.close(fig)
        print("Yes!")

print('Ebd')