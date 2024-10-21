import cv2
import pandas as pd
from ultralytics import YOLO
from tracker import*
import time
import math

model=YOLO('yolov8n.pt')

class_list = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']

dd=['person','bicycle','car','motorcycle','bus','truck']
tracker=Tracker()
count=0

cap=cv2.VideoCapture(r"C:\IDP\4K Video of Highway Traffic!.mp4")
ac_in=dict()
ac_out=dict()
object_data={}
speed=0
ppm=10
speed_limit=15
max1=0
ov_in=set()
ov_out=set()
for i in dd:
    ac_in[i]=set()
    ac_out[i]=set()
down={}
counter_down=set()

dir={}
cu=set()
cd=set()

# Create video writer object
output_path = "output_video_2.avi"
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = int(cap.get(cv2.CAP_PROP_FPS))
fourcc = cv2.VideoWriter_fourcc(*'XVID')
out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))


while True:    
    ret,frame = cap.read()
    
    if not ret:
        break
    # Resize the frame to fit the screen
    #frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
    count += 1

    results=model.predict(frame)

    a=results[0].boxes.data
    a = a.detach().cpu().numpy()  
    px=pd.DataFrame(a).astype("float")
    #print(px)
    
    list=[]
             
    for index,row in px.iterrows():
#        print(row) 
        x1=int(row[0])
        y1=int(row[1])
        x2=int(row[2])
        y2=int(row[3])
        d=int(row[5])
        c=class_list[d]
        if c in dd:
            list.append([x1,y1,x2,y2,d])


    bbox_id=tracker.update(list)
    #print(bbox_id)
    for bbox in bbox_id:
        x3,y3,x4,y4,id,cld=bbox
        cx=int(x3+x4)//2
        cy=int(y3+y4)//2
        centroid1=(cx,cy)
        dist=math.sqrt((centroid1[0] - centroid1[0]) ** 2 + (centroid1[1] - 308) ** 2)
        dist=round(dist/ppm,2)



        # Calculate speed
        if id in object_data:
            # Calculate time difference
            current_time = time.time()  # Current time
            prev_time, prev_position,max1 = object_data[id]
            time_difference = current_time - prev_time
            
            # Check if time difference is greater than or equal to one second
            if time_difference >=1:
                # Calculate distance
                distance = math.sqrt((cx - prev_position[0]) ** 2 + (cy - prev_position[1]) ** 2)
                distance=distance//ppm
                # Calculate speed using distance and time difference
                speed = distance // time_difference
                if max1<speed:
                    max1=speed
                # You can now use 'speed' variable for further processing or display
                #print("Object ID:", id, "Speed:", speed)
                # Update object data with current position and timestamp
                object_data[id] = (current_time, centroid1,max1)
        else:
            # If object is not in the dictionary, add it with current position and timestamp
            object_data[id] = (time.time(), centroid1,0)
            




        
        
        #print("Class ID:", cld)
        cv2.circle(frame,(cx,cy),4,(0,0,255),-1) #draw ceter points of bounding box
        if max1<speed_limit:
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 0), 2)  # normal speed
        else:
            cv2.rectangle(frame, (x3, y3), (x4, y4), (0, 255, 255), 2)  # overspeed
            if id in dir and dir[id]==0:
                ov_in.add(id)
            elif id in dir and dir[id]==1:
                ov_out.add(id)
        #cv2.putText(frame,class_list[cld],(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
        #cv2.putText(frame,class_list[cld]+'-'+str(dist)+"meters"+str(speed)+"mps",(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,255,255),1)
        cv2.putText(frame,class_list[cld],(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.45,(255,255,255),1)
        cv2.putText(frame,str(dist)+"meters",(cx,cy+20),cv2.FONT_HERSHEY_COMPLEX,0.45,(255,255,255),1)
        cv2.putText(frame,str(max1)+"mps",(cx,cy+40),cv2.FONT_HERSHEY_COMPLEX,0.45,(255,255,255),1)
        cv2.line(frame, centroid1,(centroid1[0],308), (0, 0, 255), 1)




        #-------------in-------------------
        y = 290
        offset = 7
        if y < (cy + offset) and y > (cy - offset):
          dir[id]=0 
        #________________out______________
        y = 330
        offset = 7
        if y < (cy + offset) and y > (cy - offset):
          dir[id]=1 

        
        y = 308
        offset = 7
    
        ''' condition for red line '''
        if y < (cy + offset) and y > (cy - offset):
          ''' this if condition is putting the id and the circle on the object when the center of the object touched the red line.'''
          
          down[id]=cy   #cy is current position. saving the ids of the cars which are touching the red line first. 
          #This will tell us the travelling direction of the car.
          if id in down:         
           #cv2.circle(frame,(cx,cy),4,(0,0,255),-1)
           #cv2.putText(frame,class_list[cld],(cx,cy),cv2.FONT_HERSHEY_COMPLEX,0.8,(0,255,255),2)
           if id in dir and dir[id]==0:#-----------in----------
               cu.add(id)
               ac_in[class_list[cld]].add(id)
           elif id in dir and dir[id]==1:#________out_____________
               cd.add(id)
               ac_out[class_list[cld]].add(id)
            #counter_down.add(id)

    # #red line
    
    text_color = (255,255,255)  # white color for text
    red_color = (0, 0, 255)  # (B, G, R)   
    cv2.line(frame,(0,308),(1600,308),red_color,1)  #  starting cordinates and end of line cordinates
    cv2.putText(frame,(''),(280,308),cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA) 
    
    #_________________yellow lines___________________________
    
    """yellow_color = (0, 255, 255)  # (B, G, R)
    cv2.line(frame, (0, 290), (frame.shape[1], 290), yellow_color, 1)
    cv2.line(frame, (0, 330), (frame.shape[1], 330), yellow_color, 1)"""


    """downwards = (len(counter_down))
    cv2.putText(frame,('going down - ')+ str(downwards),(60,40),cv2.FONT_HERSHEY_SIMPLEX, 0.5, red_color, 1, cv2.LINE_AA) 
    #-----------------------in-count---------------------------
    u = (len(cu))
    cv2.putText(frame,('IN - ')+ str(u),(60,100),cv2.FONT_HERSHEY_SIMPLEX, 0.5, yellow_color, 1, cv2.LINE_AA)
    x=len(ac_in)
    cv2.putText(frame,('IN - ')+ str(ac_in),(60,140),cv2.FONT_HERSHEY_SIMPLEX, 0.5, yellow_color, 1, cv2.LINE_AA)
    #__________________out_count______________________________
    d = (len(cd))
    cv2.putText(frame,('OUT - ')+ str(d),(60,180),cv2.FONT_HERSHEY_SIMPLEX, 0.5, yellow_color, 1, cv2.LINE_AA)
    y=len(ac_out)
    cv2.putText(frame,('OUT - ')+ str(ac_out),(60,200),cv2.FONT_HERSHEY_SIMPLEX, 0.5, yellow_color, 1, cv2.LINE_AA)"""
    for idx, cls in enumerate(dd):
        cv2.putText(frame, f'{cls.upper()} IN: {len(ac_in[cls])}', (20, 20 + idx * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f'{cls.upper()} OUT: {len(ac_out[cls])}', (200, 20 + idx * 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f'OVERSPEED OUT: {len(ov_out)}', (1100, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(frame, f'OVERSPEED in: {len(ov_in)}', (1100, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)



    out.write(frame)

    # cv2.line(frame,(282,308),(1004,308),red_color,3)  #  starting cordinates and end of line cordinates
    # cv2.putText(frame,('red line'),(280,308),cv2.FONT_HERSHEY_SIMPLEX, 0.5, text_color, 1, cv2.LINE_AA)    
    
    cv2.imshow("frames", frame)
    if cv2.waitKey(1)&0xFF==27:
        break
cap.release()
out.release()
cv2.destroyAllWindows()