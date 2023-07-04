import cv2
import openslide
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import pytz
import datetime
import os
import tifffile as tiff

# mask 불러와서 원본이미지에 색깔별로 적용
def mask_apply (image, label1, label2):
    mask1 = label1 > 0
    mask1 = mask1.astype(np.uint8)
    mask2 = label2 > 0
    mask2 = mask2.astype(np.uint8)

    mask1_rest = mask1 - mask2
    mask2_rest = mask2 - mask1
    overlap = cv2.bitwise_and(mask1, mask2)
    
    mask1_rest_rgb = (mask1_rest * [0, 0, 255]).astype(np.uint8) # answer = blue
    mask2_rest_rgb = (mask2_rest * [255,  0,  0]).astype(np.uint8) # predict = red
    overlab_rgb = (overlap * [29,  177,  76]).astype(np.uint8) # overlab = green

    
    # mask가 적용된 부분은 마스크로 계산합니다.
    masked_image1 = cv2.multiply(image, mask1_rest).astype(np.uint8)
    masked_image1 = cv2.addWeighted(masked_image1, 0.2, mask1_rest_rgb, 0.8, 0)
    masked_image2 = cv2.multiply(image, mask2_rest).astype(np.uint8)
    masked_image2 = cv2.addWeighted(masked_image2, 0.2, mask2_rest_rgb, 0.8, 0)
    masked_image3 = cv2.multiply(image, overlap).astype(np.uint8)
    masked_image3 = cv2.addWeighted(masked_image3, 0.2, overlab_rgb, 0.8, 0)

    # 이미지 합성
    result = cv2.add(masked_image1, masked_image2)
    result = cv2.add(result, masked_image3)

    # mask가 적용되지 않은 부분은 원본 이미지에서 가져옵니다.
    mask3 = label1 == 0
    mask4 = label2 == 0
    mask3 = mask3.astype(np.uint8)
    mask4 = mask4.astype(np.uint8)
    masked_image3 = cv2.multiply(image, mask3)
    masked_image4 = cv2.multiply(masked_image3, mask4)
    
    # mask가 적용된 부분과 mask가 적용되지 않은 부분을 합성합니다.
    result_image = cv2.add(result, masked_image4)
    return result_image

# 현재 시각
kst = pytz.timezone('Asia/Seoul')
current_datetime = datetime.datetime.now(kst)
day = current_datetime.strftime("%Y_%m_%d")
hour = current_datetime.strftime("%I:%M_%p")

# 경로 지정
output_path = f"git_ignore/output/{day}"
figure_path = f"{output_path}/figure"
file_name = "overlab_1024(10x).tiff"

# 이미지 불러오기
predict = cv2.imread(r'git_ignore/output/2023_06_21/figure/test_segmentation_1024(10x).tiff')
answer = cv2.imread(r'git_ignore/output/answer_segmentation.tiff')

# 원본 이미지
slide = openslide.OpenSlide("git_ignore/PDA_svs_img/C3L-01637-21.svs")
slide_arr = np.array(slide.read_region((0,0), 0, slide.level_dimensions[0]).convert("RGB")).astype(np.uint8)

# 예측 마스크
pred_label = ((predict - (0.7*slide_arr).astype(np.uint8)[...,::-1])/0.3).astype(np.uint8)
pred_label = ((pred_label!=0)).astype(np.uint8)

# 정답 마스크
answer_label = ((answer - (0.7*slide_arr).astype(np.uint8)[...,::-1])/0.3).astype(np.uint8)
answer_label = ((answer_label!=0)).astype(np.uint8)


# run
result = mask_apply(slide_arr, answer_label, pred_label).astype(np.uint8) # 마스크 적용 및 BGR to RGB

os.makedirs(f"{figure_path}", exist_ok=True)
tiff.imsave(f"{figure_path}/{file_name}",result)

print("done!!")
