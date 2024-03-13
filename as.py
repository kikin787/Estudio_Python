import cv2
import matplotlib.pyplot as plt

img = cv2.imread('GOAT.jpg')
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
print(img_rgb)