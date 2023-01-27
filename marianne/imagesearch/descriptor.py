"""The image descriptor for marianne"""
# marianne/imagesearch/descriptor.py

import math

import cv2
import imutils
import numpy as np
from skimage.feature import hog

gray_level = 16


class Descriptor:
    def __init__(self, bins):
        # store the number of bins for the 3D histogram
        self.bins = bins

    def hsv_describe(self, image):
        # convert the image to the HSV color space and initialize
        # the features used to quantify the image
        image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        features = []
        # grab the dimensions and compute the center of the image
        (h, w) = image.shape[:2]
        (cX, cY) = (int(w * 0.5), int(h * 0.5))
        # divide the image into four rectangles/segments (top-left,
        # top-right, bottom-right, bottom-left)
        segments = [(0, cX, 0, cY), (cX, w, 0, cY), (cX, w, cY, h), (0, cX, cY, h)]
        # construct an elliptical mask representing the center of the
        # image
        (axesX, axesY) = (int(w * 0.75) // 2, int(h * 0.75) // 2)
        ellipMask = np.zeros(image.shape[:2], dtype="uint8")
        cv2.ellipse(ellipMask, (cX, cY), (axesX, axesY), 0, 0, 360, 255, -1)
        # loop over the segments
        for (startX, endX, startY, endY) in segments:
            # construct a mask for each corner of the image, subtracting
            # the elliptical center from it
            cornerMask = np.zeros(image.shape[:2], dtype="uint8")
            cv2.rectangle(cornerMask, (startX, startY), (endX, endY), 255, -1)
            cornerMask = cv2.subtract(cornerMask, ellipMask)
            # extract a color histogram from the image, then update the
            # feature vector
            hist = self.histogram(image, cornerMask)
            features.extend(hist)
        # extract a color histogram from the elliptical region and
        # update the feature vector
        hist = self.histogram(image, ellipMask)
        features.extend(hist)
        # return the feature vector
        return features

    def color_moments(self, image):
        # Convert BGR to HSV colorspace
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        # Split the channels - h,s,v
        h, s, v = cv2.split(hsv)
        # Initialize the color feature
        features = []
        # N = h.shape[0] * h.shape[1]
        # The first central moment - average
        h_mean = np.mean(h)  # np.sum(h)/float(N)
        s_mean = np.mean(s)  # np.sum(s)/float(N)
        v_mean = np.mean(v)  # np.sum(v)/float(N)
        features.extend([h_mean, s_mean, v_mean])
        # The second central moment - standard deviation
        h_std = np.std(h)  # np.sqrt(np.mean(abs(h - h.mean())**2))
        s_std = np.std(s)  # np.sqrt(np.mean(abs(s - s.mean())**2))
        v_std = np.std(v)  # np.sqrt(np.mean(abs(v - v.mean())**2))
        features.extend([h_std, s_std, v_std])
        # The third central moment - the third root of the skewness
        h_skewness = np.mean(abs(h - h.mean()) ** 3)
        s_skewness = np.mean(abs(s - s.mean()) ** 3)
        v_skewness = np.mean(abs(v - v.mean()) ** 3)
        h_thirdMoment = h_skewness ** (1.0 / 3)
        s_thirdMoment = s_skewness ** (1.0 / 3)
        v_thirdMoment = v_skewness ** (1.0 / 3)
        features.extend([h_thirdMoment, s_thirdMoment, v_thirdMoment])

        return features

    def glcm(self, image, d_x, d_y):
        arr = image.copy()
        ret = [[0.0 for i in range(gray_level)] for j in range(gray_level)]
        (height, width) = image.shape

        max_gray_level = image.max() + 1

        if max_gray_level > gray_level:
            for j in range(height):
                for i in range(width):
                    arr[j][i] = arr[j][i] * gray_level / max_gray_level

        for j in range(height - d_y):
            for i in range(width - d_x):
                if i < width & j < height:
                    rows = arr[j][i]
                    cols = arr[j + d_y][i + d_x]
                    ret[rows][cols] += 1.0

        for i in range(gray_level):
            for j in range(gray_level):
                ret[i][j] /= float(height * width)
        con = 0.0
        eng = 0.0
        asm = 0.0
        idm = 0.0
        for i in range(gray_level):
            for j in range(gray_level):
                con += (i - j) * (i - j) * ret[i][j]
                asm += ret[i][j] * ret[i][j]
                idm += ret[i][j] / (1 + (i - j) * (i - j))
                if ret[i][j] > 0.0:
                    eng += ret[i][j] * math.log(ret[i][j])
        features = [con, eng, asm, idm]
        epsilon = 1e-5
        features = np.abs(np.log(np.abs(features) + epsilon))
        return features

    def gray_matrix(self, image):
        try:
            img_shape = image.shape
        except BaseException:
            print("imread error")
            return -1
        img = image
        img = cv2.resize(
            img,
            (int(img_shape[1] / 2), int(img_shape[0] / 2)),
            interpolation=cv2.INTER_CUBIC,
        )
        img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        glcm_0 = self.glcm(img_gray, 1, 0)
        glcm_1 = self.glcm(img_gray, 0, 1)
        glcm_2 = self.glcm(img_gray, 1, 1)
        glcm_3 = self.glcm(img_gray, -1, 1)
        glcm = [glcm_0, glcm_1, glcm_2, glcm_3]
        features = [x for j in glcm for x in j]
        return features

    def humoments(self, image):
        img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        moments = cv2.moments(img_gray)
        humoments = cv2.HuMoments(moments)
        features = [x for j in humoments for x in j]
        features = np.abs(np.log(np.abs(features)))
        return features
        # img_gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # (row, col) = img_gray.shape
        # m00 = img_gray.sum()
        # m10 = m01 = 0
        # m11 = m20 = m02 = m12 = m21 = m30 = m03 = 0
        # for i in range(row):
        #     m10 += (i * img_gray[i]).sum()
        #     m20 += (i ** 2 * img_gray[i]).sum()
        #     m30 += (i ** 3 * img_gray[i]).sum()
        #     for j in range(col):
        #         m11 += i * j * img_gray[i][j]
        #         m12 += i * j ** 2 * img_gray[i][j]
        #         m21 += i ** 2 * j * img_gray[i][j]
        # for j in range(col):
        #     m01 += (j * img_gray[:, j]).sum()
        #     m02 += (j ** 2 * img_gray[:, j]).sum()
        #     m30 += (j ** 3 * img_gray[:, j]).sum()
        # u10 = m10 / m00
        # u01 = m01 / m00
        # y00 = m00
        # y10 = y01 = 0
        # y11 = m11 - u01 * m10
        # y20 = m20 - u10 * m10
        # y02 = m02 - u01 * m01
        # y30 = m30 - 3 * u10 * m20 + 2 * u10 ** 2 * m10
        # y12 = m12 - 2 * u01 * m11 - u10 * m02 + 2 * u01 ** 2 * m10
        # y21 = m21 - 2 * u10 * m11 - u01 * m20 + 2 * u10 ** 2 * m01
        # y03 = m03 - 3 * u01 * m02 + 2 * u01 ** 2 * m01
        # n20 = y20 / m00 ** 2
        # n02 = y02 / m00 ** 2
        # n11 = y11 / m00 ** 2
        # n30 = y30 / m00 ** 2.5
        # n03 = y03 / m00 ** 2.5
        # n12 = y12 / m00 ** 2.5
        # n21 = y21 / m00 ** 2.5
        # h1 = n20 + n02
        # h2 = (n20 - n02) ** 2 + 4 * n11 ** 2
        # h3 = (n30 - 3 * n12) ** 2 + (3 * n21 - n03) ** 2
        # h4 = (n30 + n12) ** 2 + (n21 + n03) ** 2
        # h5 = (n30 - 3 * n12) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2) + (3 * n21 - n03) * (n21 + n03) \
        #     * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2)
        # h6 = (n20 - n02) * ((n30 + n12) ** 2 - (n21 + n03) ** 2) + 4 * n11 * (n30 + n12) * (n21 + n03)
        # h7 = (3 * n21 - n03) * (n30 + n12) * ((n30 + n12) ** 2 - 3 * (n21 + n03) ** 2) + (3 * n12 - n30) * (n21 + n03) \
        #     * (3 * (n30 + n12) ** 2 - (n21 + n03) ** 2)
        # inv_m7 = [h1, h2, h3, h4, h5, h6, h7]
        # features = np.abs(np.log(np.abs(inv_m7)))

    def ahash(self, image):
        img = cv2.resize(image, (8, 8))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        np_mean = np.mean(gray)
        ahash_01 = (gray > np_mean) + 0
        features = ahash_01.reshape(1, -1)[0].tolist()
        return features

    def phash(self, image):
        img = cv2.resize(image, (32, 32))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        dct = cv2.dct(np.float32(gray))
        dct_roi = dct[0:8, 0:8]

        avreage = np.mean(dct_roi)
        phash_01 = (dct_roi > avreage) + 0
        features = phash_01.reshape(1, -1)[0].tolist()
        return features

    def dhash(self, image):
        res_image = cv2.resize(image, (9, 8))
        gray = cv2.cvtColor(res_image, cv2.COLOR_RGB2GRAY)

        features = [0 for i in range(64)]
        for row in range(8):
            for col in range(8):
                if gray[row][col] > gray[row][col + 1]:
                    a = 1
                else:
                    a = 0
                features[row * 8 + col] = a

        return features

    def mse(self, image):
        img = cv2.resize(image, (32, 32))
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        line_MSEs = []
        for i in range(32):
            avg = np.mean(gray[i, :])
            line_MSE = np.square(gray[i, :] - avg)
            line_MSEs.append(line_MSE)
        features = features = [x for j in line_MSEs for x in j]
        return features

    def hog(self, image):
        res_image = cv2.resize(image, (256, 256))
        gray = res_image / 255
        features = hog(
            gray,
            orientations=20,
            pixels_per_cell=[20, 20],
            cells_per_block=[10, 10],
            visualize=False,
            transform_sqrt=True,
            block_norm="L2-Hys",
        )
        return features
        # img = image/255
        # g_img = np.zeros((img.shape[0],img.shape[1],3,2))
        # for i in range(1,img.shape[0]-1):
        #     for j in range(1,img.shape[1]-1):
        #         gx_b = img[i+1,j,0] - img[i-1,j,0]
        #         gy_b = img[i,j+1,0] - img[i,j-1,0]
        #         gx_g = img[i+1,j,1] - img[i-1,j,1]
        #         gy_g = img[i,j+1,1] - img[i,j-1,1]
        #         gx_r = img[i+1,j,2] - img[i-1,j,2]
        #         gy_r = img[i,j+1,2] - img[i,j-1,2]
        #         gb = (gx_b**2 + gy_b**2)**0.5
        #         gg = (gx_g**2 + gy_g**2)**0.5
        #         gr = (gx_r**2 + gy_r**2)**0.5
        #         if gx_b == 0 and gy_b == 0:
        #             dgb = 0
        #         elif gx_b == 0 and gy_b != 0:
        #             dgb = math.pi/2
        #         else:
        #             dgb = math.atan(gy_b/gx_b)
        #             if dgb < 0:
        #                 dgb = dgb + math.pi
        #         if gx_g == 0 and gy_g == 0:
        #             dgg = 0
        #         elif gx_g == 0 and gy_g != 0:
        #             dgg = math.pi/2
        #         else:
        #             dgg = math.atan(gy_g/gx_g)
        #             if dgg < 0:
        #                 dgg = dgg + math.pi
        #         if gx_r == 0 and gy_r == 0:
        #             dgr = 0
        #         elif gx_r == 0 and gy_r != 0:
        #             dgr = math.pi/2
        #         else:
        #             dgr = math.atan(gy_r/gx_r)
        #             if dgr < 0:
        #                 dgr = dgr + math.pi
        #         g_img[i,j,0,0] = gb
        #         g_img[i,j,1,0] = gg
        #         g_img[i,j,2,0] = gr
        #         g_img[i,j,0,1] = dgb
        #         g_img[i,j,1,1] = dgg
        #         g_img[i,j,2,1] = dgr
        # cell_n = np.zeros((3,9))
        # h = 24
        # w = 24
        # h_size = img.shape[0]//24
        # w_size = img.shape[1]//24

        # cell = np.zeros((h,w,27))
        # for m in range(h):
        #     for n in range(w):
        #         for i in range(h_size*m,h_size*(m+1)):
        #             for j in range(w_size*n,w_size*(n+1)):
        #                 for k in range(3):
        #                     cell_n[k,int(g_img[i,j,k,1]//(math.pi/9))] += g_img[i,j,k,0]
        #         cell[m,n] = cell_n.reshape(27)
        # block = np.zeros((h//2,w//2,27))
        # for p in range(h//2):
        #     for q in range(w//2):
        #         for i in range(2*p,2*p+2):
        #             for j in range(2*q,2*q+2):
        #                 block[p,q] += cell[i,j]
        # block_norm = np.zeros((h//2,w//2,27))
        # for i in range(h//2):
        #     for j in range(w//2):
        #         length = (np.linalg.norm(block[i,j])**2 + 0.000001)**0.5
        #         block_norm[i,j] = block[i,j]/length
        # block_norm = block_norm.reshape(block_norm.shape[0]*block_norm.shape[1],27)
        # features = [x for j in block_norm for x in j]

    def histogram(self, image, mask):
        # extract a 3D color histogram from the masked region of the
        # image, using the supplied number of bins per channel
        hist = cv2.calcHist(
            [image], [0, 1, 2], mask, self.bins, [0, 180, 0, 256, 0, 256]
        )
        # normalize the histogram if we are using OpenCV 2.4
        if imutils.is_cv2():
            hist = cv2.normalize(hist).flatten()
        # otherwise handle for OpenCV 3+
        else:
            hist = cv2.normalize(hist, hist).flatten()
        # return the histogram
        return hist
