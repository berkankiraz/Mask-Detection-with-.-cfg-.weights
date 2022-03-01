# Mask-Detection with .cfg ,.weights file
![image](https://user-images.githubusercontent.com/63494351/156134001-b66f064b-55fd-4929-933f-2db7abfbf4cb.png)

img = cv2.imread("write the path of photo")

#In above example,there are two rectangles.If you want to keep just max value,you can use Non-maximum Suppression (NMS).

#.cfg and .weights file's path is written like that:
model = cv2.dnn.readNetFromDarknet("absolute path of .cfg","absolute path of .weights")

.cfg and .weights:
https://drive.google.com/drive/folders/16MsdDvPuF6CxFd0vW2VYya6e5cqZJjZI
