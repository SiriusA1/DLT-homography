import numpy as np
import imageio.v2 as imageio
from PIL import Image, ImageDraw

img = imageio.imread("basketball-court.ppm")

# points (corners): bottom left, top left, top right, bottom right
# basketball courts are almost twice as long as they are wide, so I'll choose points keeping that in mind
# p1 = original points
# p2 = final points
p1 = [(277, 279), (192, 24), (52, 247), (75, 401)]
p2 = [(310, 122), (170, 122), (170, 402), (310, 402)]

# line features. used the corners above and the equations ux1+vy1+1=0 and ux2+vy2+1=0 to find the parameters
# l1 = original lines
# l2 = final lines
l1 = [(0.00181, -0.00543), (-0.00303, -0.00483), (0.009884, -0.06618), (-0.002241, -0.001353)]
l2 = [(-0.0082, 0), (0, -0.00588), (-0.00249, 0), (0, -0.00323)]

# setting colors of original corners and intended corners
temp_img = imageio.imread("basketball-court.ppm")
for i in range(4):
        temp_img[p1[i][0]-2:p1[i][0]+3, p1[i][1]-2:p1[i][1]+3, :] = [255, 0, 0]
        temp_img[p2[i][0]-2:p2[i][0]+3, p2[i][1]-2:p2[i][1]+3, :] = [0, 255, 0]

imageio.imsave("corners.ppm", temp_img)

# setting colors of original lines and intended lines
# need to flip the coordinates of the points and repeat the first point to easily draw rectangle
p1_i = [(279, 277), (24, 192), (247, 52), (401, 75), (279, 277)]
p2_i = [(122, 310), (122, 170), (402, 170), (402, 310), (122, 310)]
temp_img = Image.open("basketball-court.ppm")
lines = ImageDraw.Draw(temp_img)  
lines.line(p1_i, fill ="red", width = 2)
lines.line(p2_i, fill ="green", width = 2)

temp_img.save("lines.ppm")

# matrix for points
A = np.zeros((8,9))
# Creating Homography matrix by stacking the 2x9 matrix 4 times into an 8x9
for i in range(4):
        A[i*2,:] = [p1[i][1], p1[i][0], 1, 0, 0, 0, -p2[i][1]*p1[i][1], -p2[i][1]*p1[i][0], -p2[i][1]]
        A[i*2+1,:] = [0, 0, 0, p1[i][1], p1[i][0], 1, -p2[i][0]*p1[i][1], -p2[i][0]*p1[i][0], -p2[i][0]]

# matrix for lines
A_l = np.zeros((8,9))
# Creating Homography matrix by stacking the 2x9 matrix 4 times into an 8x9
for i in range(4):
        A_l[i*2,:] = [0, 0, 0, -l1[i][0], -l1[i][1], -1, l1[i][0]*l2[i][1], l1[i][1]*l2[i][1], l2[i][1]]
        A_l[i*2+1,:] = [l1[i][0], l1[i][1], 1, 0, 0, 0, -l1[i][0]*l2[i][0], -l1[i][1]*l2[i][0], -l2[i][0]]

# A*H = 0

[U,S,V] = np.linalg.svd(A)
[U_l,S_l,V_l] = np.linalg.svd(A_l)
# The last singular vector of V is the solution of H
ans = V[-1,:]
ans_l = V_l[-1,:]
H = np.reshape(ans, (3,3))
H_l = np.reshape(ans_l, (3,3))
# need inverse transpose of H for the line feature calculations
# more info here https://www.cs.ubc.ca/labs/lci/thesis/ankgupta/gupta11crv.pdf
H_l_it = np.linalg.inv(H_l.transpose())

# A*H = 0
print("A*H: "+str(np.sum(np.dot(A,ans))))
print("A_l*H_l: "+str(np.sum(np.dot(A_l,ans_l))))

final = np.zeros((500,940,3))
final_l = np.zeros((500,940,3))

# Iterate over the image to transform every pixel
for i in range(img.shape[0]):
        for j in range(img.shape[1]):
                p = np.array([[j], [i], [1]])
                t = np.dot(H, p)
                x = int(t[0] / t[2])
                y = int(t[1] / t[2])

                p_l = np.array([[j], [i], [1]])
                t_l = np.dot(H_l_it, p_l)
                x_l = int(t_l[0] / t_l[2])
                y_l = int(t_l[1] / t_l[2])

                if x > 0 and y > 0 and y < 500 and x < 940:
                        final[y, x, :] = img[i, j, :]
                
                if x_l > 0 and y_l > 0 and y_l < 500 and x_l < 940:
                        final_l[y_l, x_l, :] = img[i, j, :]

imageio.imsave("no_interp.ppm", final)
imageio.imsave("no_interp_lines.ppm", final_l)

# interpolation for point image
Hi = np.linalg.inv(H)
for i in range(final.shape[0]):
        for j in range(final.shape[1]):
                if sum(final[i, j, :]) == 0:
                        p = np.array([[j], [i],[1]])
                        t = np.dot(Hi, p)
                        x = int(t[0] / t[2])
                        y = int(t[1] / t[2])

                        if x > 0 and y > 0 and x < img.shape[1] and y < img.shape[0]:
                                final[i, j, :] = img[y, x, :]

imageio.imsave("final.ppm", final)

# interpolation for line image
Hi_l = np.linalg.inv(H_l_it)
for i in range(final_l.shape[0]):
        for j in range(final_l.shape[1]):
                if sum(final_l[i, j, :]) == 0:
                        p = np.array([[j], [i],[1]])
                        t = np.dot(Hi_l, p)
                        x = int(t[0] / t[2])
                        y = int(t[1] / t[2])

                        if x > 0 and y > 0 and x < img.shape[1] and y < img.shape[0]:
                                final_l[i, j, :] = img[y, x, :]

imageio.imsave("final_lines.ppm", final_l)