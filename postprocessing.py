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
lines = read_labels('./model/line', title ='labels_pred_rel')
N = lines.shape[0]
lines = lines.reshape(N, 512, 256)

ln = lines[0]
ln_thres = np.zeros_like(ln)
mask = ln > 0.25
ln_thres[mask] = 1
_lb, maxno = label(ln_thres, return_num=True)

labeled_lines = []
for i in range(1,maxno+1):
    mask = _lb == i
    labeled_lines.append(mask*1)

sorted(labeled_lines, key = lambda x:np.sum(x), reverse = True)
print(len(labeled_lines))


from label_io import hwc

plt.figure()
plt.subplot(221)
plt.imshow(hwc(ln))
plt.subplot(222)
plt.imshow(hwc(ln_thres))
# plt.subplot(223)
# plt.imshow(hwc(labeled_lines[0]))
# plt.subplot(224)
# plt.imshow(hwc(labeled_lines[1]))
plt.show()