import cv2

# 이미지 불러오기
img1 = cv2.imread(r'git_ignore\output\answer_segmentation.tiff')
img2 = cv2.imread(r'git_ignore\output\2023_06_20\figure\test_1024segmentation.tiff')


# 이미지 크기 확인
height1, width1, _ = img1.shape
height2, width2, _ = img2.shape

# 두 이미지 크기가 같아야 합성 가��
if (height1 != height2) or (width1 != width2):
    print('두 이미지 크기가 다릅니다.')
else:
    # 이미지 합성
    blend = cv2.addweighted(img1, img2)

    # 결과 이미지 출력 및 저장
    cv2.imshow('image blend', blend)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    cv2.imwrite('result.tif', blend)