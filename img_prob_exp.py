import numpy as np
import numba as nb
import glob as glob
import cv2 as cv

def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))
@nb.jit(nopython=True)
def calculate_exponential_distance(image, radius, alpha=1):
    height, width = image.shape
    result = np.zeros_like(image, dtype=np.float32)
    nonzero_indices = np.nonzero(image)
    num_nonzero = nonzero_indices[0].shape[0]

    for i in range(height):
        for j in range(width):
            # if image[i, j] != 0:
            # p = np.array([i, j])
            distance_sum_m = 0
            distance_sum_r = 0

            for k in range(num_nonzero):
                x, y = nonzero_indices[0][k], nonzero_indices[1][k]

                dx = x - i
                dy = y - j
                d_rp = np.sqrt(dx * dx + dy * dy)
                exp_d_rp = np.exp(-alpha * d_rp)

                distance_sum_m += exp_d_rp

                if abs(dx) <= radius and abs(dy) <= radius:
                    distance_sum_r += exp_d_rp

            # result[i, j] = distance_sum_m / distance_sum_r
            result[i, j] = distance_sum_r / distance_sum_m

    return result

if __name__ == '__main__':
    images_path = sorted(glob.glob("/users2/local/h22elhaf/Faster_RCNN_for_DOTA/val/Prob_images/images/*.png"))
    print('all_images:', len(images_path))
    for item in images_path:
        print('processing item:', item)
        img = cv.imread(item,cv.IMREAD_GRAYSCALE)
        img_name = item.split('/')[-1][:5]
        img_suffix = item.split('/')[-1][5:]

        plane = np.where(img != 103, 0, img)
        ship = np.where(img != 7, 0, img)
        storage_tank = np.where(img != 44, 0, img)
        baseball_diamond = np.where(img != 36, 0, img)
        tennis_court = np.where(img != 51, 0, img)
        basketball_court = np.where(img != 58, 0, img)
        ground_track_field = np.where(img != 66, 0, img)
        harbor = np.where(img != 76, 0, img)
        bridge = np.where(img != 81, 0, img)
        large_vehicle = np.where(img != 89, 0, img)
        small_vehicle = np.where(img != 14, 0, img)
        helicopter = np.where(img != 21, 0, img)
        roundabout = np.where(img != 126, 0, img)
        soccer_ball_field = np.where(img != 96, 0, img)
        swimming_pool = np.where(img != 29, 0, img)

        CLASSES = {'plane': plane, 'ship': ship, 'storage_tank': storage_tank, 
               'baseball_diamond':baseball_diamond, 'tennis_court':tennis_court,
               'basketball_court':basketball_court, 'ground_track_field':ground_track_field, 
               'harbor':harbor, 'bridge':bridge, 'large_vehicle':large_vehicle, 
               'small_vehicle':small_vehicle, 'helicopter': helicopter, 'roundabout':roundabout, 
               'soccer_ball_field':soccer_ball_field, 'swimming_pool':swimming_pool}
        
        for label, img in CLASSES.items():
            
            save_link = '/users2/local/h22elhaf/Faster_RCNN_for_DOTA/val/Prob_images_exp/'+ img_name + '_' + label + img_suffix
            img = cv.resize(img, (416,416))
            if img.max() != 0:
                mg = calculate_exponential_distance(img, 5, 1)
                mg = mg * 255  # Divide by 255 afterwards
                # mg = NormalizeData(mg)
                cv.imwrite(save_link, mg)
            else:
                cv.imwrite(save_link, img)









# # This is the code that works !!!
# import numpy as np
# import matplotlib.pyplot as plt

# def calculate_exponential_distance(image, radius, alpha=1):
#     height, width = image.shape
#     result = np.zeros_like(image, dtype=np.float32)

#     for i in range(height):
#         for j in range(width):
#             # if image[i, j] == 0:
#             #     continue

#             distance_sum_m = 0
#             distance_sum_r = 0

#             for x in range(height):
#                 for y in range(width):
#                     if image[x, y] != 0:
#                         d_rp = np.sqrt((i - x) ** 2 + (j - y) ** 2)
#                         distance_sum_m += np.exp(-alpha * d_rp)
#                         if abs(x - i) <= radius and abs(y - j) <= radius:
#                             distance_sum_r += np.exp(-alpha * d_rp)

#             result[i, j] = distance_sum_r / distance_sum_m
            
#             # if distance_sum_r != 0:
#                 # result[i, j] = distance_sum_m / distance_sum_r
#             # else:
#             #     result[i, j] = 0

#     return result


# test = np.array([[255,0,0,0,0,0],[255,0,0,0,0,0], [0,255,0,0,0,0], [0,0,255,255,255,0],[0,0,0,0,0,255],[0,0,0,0,0,255]])
# print(test)
# plt.imshow(calculate_exponential_distance(test, 1,alpha=1))