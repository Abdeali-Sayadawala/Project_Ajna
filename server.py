"""
Abdeali
02-12-2019, 11:30
"""

"""importing all the important files"""
import cv2
"""Imagezmq is a zmq library to send and recieve images over network"""
from Assets import imagezmq
import numpy as np 
import socket
"""predict is out moduel to detect faces and recognize them"""
import predict

"""initializing the server using imagezmq"""
server_init = imagezmq.ImageHub(open_port='tcp://*:8008')
hostname = socket.gethostname()
ipaddress = socket.gethostbyname(hostname)
print("Ip address of server: "+str(ipaddress))

blank = predict.Play()

"""infinite loop to recieve image from client and show it on screen"""
while True:
    """We recieve image and message from the client connected"""
    (msg, frame) = server_init.recv_image()
    
    faces = blank.give_me_face(frame)
    for (x, y, w, h) in faces:
        name = blank.give_me_names(frame[y-20:y+h+20, x-12:x+w+12])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0,255,0), 2)
        cv2.putText(frame, name, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,255,0), 2)
    """When we stop here we send a reply stop to the client to stop it."""
    if cv2.waitKey(1) & 0xFF == ord("q"):
        server_init.send_reply(b'stop')
        break
    """Send reply to server to keep sending."""
    server_init.send_reply(b'OK')
    cv2.imshow(msg, frame)
    
cv2.destroyAllWindows()        
    
    