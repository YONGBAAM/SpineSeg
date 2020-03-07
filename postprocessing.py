'''
뭘 할거냐면

가장 큰 커넥팅 오브젝트 파악 ㅇㅋ
(이걸 네트워크에 넣어서)

라인 찾기
비슷한 알고리즘으로 (위 아래 무시 방법? - 걍 일괄처리 할까 논문에서나온대로)
논문에선 걍일괄처리했음

이미지 출력때문 노말라이즈


한숨자고 골뱅이일 후 이거 나온걸로 테스팅툴 만들기








'''


import numpy as np

line_pre = np.load('./train_lines/1.npy')

import matplotlib.pyplot as plt
from label_io import read_labels
from skimage.measure import label
# lines = read_labels('./model/line', title ='labels_pred_rel')
# N = lines.shape[0]
# lines = lines.reshape(N, 512, 256)
#
# ln = lines[0]
# ln_thres = np.zeros_like(ln)
# mask = ln > 0.25
# ln_thres[mask] = 1
# _lb, maxno = label(ln_thres, return_num=True)
#
# labeled_lines = []
# for i in range(1,maxno+1):
#     mask = _lb == i
#     labeled_lines.append(mask*1)
#
# sorted(labeled_lines, key = lambda x:np.sum(x), reverse = True)
# print(len(labeled_lines))

def get_line_coord(line_img):
    row_thres = 100
    H,W = line_img.shape

    ccords = np.zeros(H)

    for i in range(H):
        rowsum = 0
        indsum = 0
        for j,pixel in enumerate(line_img[i]):
            rowsum += pixel
            if pixel >0.9:
                indsum += j

        if rowsum < row_thres:#no detect
            middle_j = 0
        else:
            middle_j = indsum/rowsum

        ccords[i] = middle_j

    start = 0
    end = 0
    started = False
    for i,c in enumerate(ccords):
        if not started:
            if c >0:
                started = True
                start = i
                end = start
        if started:
            if c ==0:
                end = i
    return ccords, (start, end)


sample_seg = np.load('./train_labels/1.npy')
m = sample_seg > 0.25
seg_th = np.zeros_like(sample_seg)
seg_th[m] = 1

lines, (s,e) =get_line_coord(seg_th)
r_axis = np.arange(len(lines))
# xy = np.concatenate((lines, r_axis), axis = 1)




from label_io import hwc

plt.figure()
# plt.subplot(221)
# plt.imshow(hwc(ln))
# plt.subplot(222)
# plt.imshow(hwc(ln_thres))
plt.imshow(hwc(sample_seg))
plt.scatter(lines, r_axis)
# plt.subplot(224)
# plt.imshow(hwc(labeled_lines[1]))
plt.show()



