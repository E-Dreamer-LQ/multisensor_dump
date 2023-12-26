import cv2
import numpy as np

def calculate_slope(line):
    """
    计算线段line的斜率
    :param line: np.array([[x_1, y_1, x_2, y_2]])
    :return:
    """
    x_1, y_1, x_2, y_2 = line[0]
    return (y_2 - y_1) / (x_2 - x_1)

edge_img = cv2.imread('road_lai_origin.jpg', cv2.IMREAD_GRAYSCALE)
edges = cv2.Canny(edge_img, 40, 68)

mask = np.zeros_like(edge_img)
mask = cv2.fillPoly(mask,
                    np.array([[[211,331], [238,221], [256,221], [280,331]]]),
                    color=255)

masked_edge_img = cv2.bitwise_and(edges, mask)
#cv2.imshow('canny_detection', edges)

# 获取所有线段
#lines = cv2.HoughLinesP(masked_edge_img, 1, np.pi / 291, 13, minLineLength=570, maxLineGap=4)
lines = cv2.HoughLinesP(masked_edge_img, 1, np.pi / 180, 15, minLineLength=40,
                        maxLineGap=20)
# 按照斜率分成车道线
left_lines = [line for line in lines if calculate_slope(line) > 0]
right_lines = [line for line in lines if calculate_slope(line) < 0]

def reject_abnormal_lines(lines, threshold):
    """
    剔除斜率不一致的线段
    :param lines: 线段集合, [np.array([[x_1, y_1, x_2, y_2]]),np.array([[x_1, y_1, x_2, y_2]]),...,np.array([[x_1, y_1, x_2, y_2]])]
    """
    slopes = [calculate_slope(line) for line in lines]
    while len(lines) > 0:
        mean = np.mean(slopes)
        diff = [abs(s - mean) for s in slopes]
        idx = np.argmax(diff)
        if diff[idx] > threshold:
            slopes.pop(idx)
            lines.pop(idx)
        else:
            break
    return lines


print('before filter:')
print('left lines number=')
print(len(left_lines))
print('right lines number=')
print(len(right_lines))

reject_abnormal_lines(left_lines, threshold=0.2)
reject_abnormal_lines(right_lines, threshold=0.01)


print('after filter:')
print('left lines number=')
print(len(left_lines))
print('right lines number=')
print(len(right_lines))

def least_squares_fit(lines):
    """
    将lines中的线段拟合成一条线段
    :param lines: 线段集合, [np.array([[x_1, y_1, x_2, y_2]]),np.array([[x_1, y_1, x_2, y_2]]),...,np.array([[x_1, y_1, x_2, y_2]])]
    :return: 线段上的两点,np.array([[xmin, ymin], [xmax, ymax]])
    """
    # 1. 取出所有坐标点
    x_coords = np.ravel([[line[0][0], line[0][2]] for line in lines])
    y_coords = np.ravel([[line[0][1], line[0][3]] for line in lines])
    # 2. 进行直线拟合.得到多项式系数
    poly = np.polyfit(x_coords, y_coords, deg=1)
    # 3. 根据多项式系数,计算两个直线上的点,用于唯一确定这条直线
    point_min = (np.min(x_coords), np.polyval(poly, np.min(x_coords)))
    point_max = (np.max(x_coords), np.polyval(poly, np.max(x_coords)))
    return np.array([point_min, point_max], dtype=np.int)

print("left lane")
print(least_squares_fit(left_lines))
print("right lane")
print(least_squares_fit(right_lines))

left_line = least_squares_fit(left_lines)
right_line = least_squares_fit(right_lines)
img = cv2.imread('road_lai_origin.jpg', cv2.IMREAD_COLOR)
cv2.line(img, tuple(left_line[0]), tuple(left_line[1]), color=(0, 255, 255), thickness=5)
cv2.line(img, tuple(right_line[0]), tuple(right_line[1]), color=(0, 255, 255), thickness=5)
cv2.imshow('lane', img)
cv2.imshow('masked', masked_edge_img)
cv2.waitKey(0)

