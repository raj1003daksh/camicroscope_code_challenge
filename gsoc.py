import numpy as np 
import cv2
from collections import deque
def solve(A):
    q = deque()
    r = len(A)
    c = len(A[0])
    map = {}
    res = []
        
    for i in range(r):
        t = [-1]*c
        res.append(t)
        
    for i in range(r):
        for j in range(c):
            if A[i][j]==255:
                res[i][j]=0
                q.append((i,j))
                          
    while q:
        src = q.popleft()
        i , j = src[0],src[1]
        if not (src in map):
            map[src]=1
            d = res[i][j]
                
        if i>0 and res[i-1][j]==-1 :
            res[i-1][j] = d+1
            q.append((i-1,j))
                
        if i<r-1 and res[i+1][j]==-1 :
            res[i+1][j] = d+1
            q.append((i+1,j))
                    
        if j>0 and res[i][j-1]==-1 :
            res[i][j-1] = d+1
            q.append((i,j-1))
                    
        if j<c-1 and res[i][j+1]==-1 :
            res[i][j+1] = d+1
            q.append((i,j+1))
       
    return res
      
img=cv2.imread('Building.jpg',0)
print(img.shape)
canny=cv2.Canny(img,100,200)
cv2.imshow("cannyedge",canny)
cv2.waitKey(0)
cv2.destroyAllWindows()
print(canny)
res=solve(canny)
val1 = input("Enter your value of x coordinate: ")
val2=input("Enter the value of y coordinate: ")

print(res[int(val1)][int(val2)])