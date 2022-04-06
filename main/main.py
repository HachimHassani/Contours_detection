
import cv2
import os
import numpy as np
from math import sqrt,floor,pi,cos,sin
import matplotlib.pyplot as plt

moyenneur = np.array([[1/9]*3]*3)

sobel = [np.array([[-1,0,1],[-1,0,1],[-1,0,1]]),np.array([[-1,-1,-1],[0,0,0],[-1,1,1]])]
prewitt = [np.array([[-1,0,1],[-2,0,2],[-1,0,1]]),np.array([[-1,-2,-1],[0,0,0],[1,2,1]])]
laplacien = np.array([[0,1,0],[1,-4,1],[0,1,0]])

here = os.chdir("C:\hm\cpge\MPSI\TIPE\code\imgs")

#filename = os.path.join(here, "Image 2021-06-08 at 12.45.59.jpeg")


def histogram(ima,f):
    ima = gray(ima)
    x,y=ima.shape
    hist = list(range(256))
    for i in range(x):
        for u in range(y):  
            hist[ima[i,u]]+=1

    plt.plot(range(256),hist)
    plt.savefig("histogramme "+ f)
    plt.show()


    pass
def gray(ima):
    return cv2.cvtColor(ima, cv2.COLOR_BGR2GRAY)




def convolution(u,v):
    res = np.zeros(u.shape)
    dim = v.shape[0]//2
    for i in range(v.shape[0]):
        for j in range(v.shape[0]):
            if i - dim == 0 and j - dim == 0:
                res+=u*v[dim,dim]
            elif i - dim == 0 and j - dim != 0:
                if j-dim >0:
                    res[:,:dim-j]+=u[:,j-dim:]*v[i,j]
                else:
                    res[:,dim-j:]+=u[:,:j-dim]*v[i,j]
            elif i - dim != 0 and j - dim == 0:
                if i-dim >0:
                    res[:dim-i,:]+=u[i-dim:,:]*v[i,j]
                else:
                    res[dim-i:,:]+=u[:i-dim,:]*v[i,j]
            else:
                if i-dim >0 and j-dim > 0:
                    res[:dim-i,:dim-j]+=u[i-dim:,j-dim:]*v[i,j]
                elif i-dim >0 and j-dim < 0:
                    res[:dim-i,dim-j:]+=u[i-dim:,:j-dim]*v[i,j]
                elif i-dim <0 and j-dim > 0:
                    res[dim-i:,:dim-j]+=u[:i-dim,j-dim:]*v[i,j]
                elif i-dim <0 and j-dim < 0:
                    res[dim-i:,dim-j:]+=u[:i-dim,:j-dim]*v[i,j]

            
    return res

def seuillage(ima,s):
    for i in range(x):
        for j in range(y):
            if ima[i,j]>=s:
                ima[i,j]=255
            else:
                ima[i,j]=0
    return ima

"""
La transformee de hough:
rho et theta sont les resolutions du rayon et de l angle theta 
"""


#lap = convolution(im,np.array([[0,1,0],[1,-4,1],[0,1,0]]))


def hough_transform(ima):
    x,y = ima.shape
    rho = 1
    theta = 1
    nt=int(180/rho)
    nr=int(floor(sqrt(x**2+y**2))/rho)
    dt = pi/nt
    dr = sqrt(x**2+y**2)/nr
    hou=np.zeros((nt,nr))
    for i in range(x):
        for j in range(y):
            if ima[i,j]!=0:
                for k in range(nt):
                    t = k*dt
                    r = i*cos(t)+(x-i)*sin(t)
                    ir = int(r/dr)
                    if ir>0 and ir < nr:
                        hou[k,ir]+=1
    hou = seuillage(hou,130)
    lis = []
    for i in range(x):
        for j in range(y):
            if hou[i,j]!=0:
                lis.append((i,j))
    plt.axis([0,x,0,y])
    for rho,theta in lis:
        a = cos(theta)
        b = sin(theta)
        x0 = a*rho
        y0 = b*rho
        x1 = int(x0 + 1000*(-b))
        y1 = int(y0 + 1000*(a))
        x2 = int(x0 - 1000*(-b))
        y2 = int(y0 - 1000*(a))
        plt.plot([x1,x2],[y1,y2],color="b")
    return lis
print(os.listdir())

ima = gray(cv2.imread("plaques.jpeg"))
lis = cv2.HoughLines(seuillage(convolution(ima,laplacien),130),rho=1,theta=1)

# for i in os.listdir():
#     try:
#         im = cv2.imread(i)
#         ima = gray(im)
#         cv2.imwrite("gray_"+i,ima)
#         cv2.imwrite("moyenneur_"+i,convolution(ima,moyenneur))
#         cv2.imwrite("sobelx_"+i,convolution(ima,sobel[0]))
#         cv2.imwrite("sobely_"+i,convolution(ima,sobel[1]))
#         cv2.imwrite("prewittx_"+i,convolution(ima,sobel[0]))
#         cv2.imwrite("prewitty_"+i,convolution(ima,sobel[1]))
#         cv2.imwrite("laplacienwm_"+i,convolution(convolution(ima,moyenneur),laplacien))
#         cv2.imwrite("laplaciensm_"+i,convolution(ima,laplacien))
#         histogram(im,i)
#     except Exception as e:


cv2.waitKey(0)
cv2.destroyAllWindows()

        
print('hello')
