import numpy as np
from skimage.filters import sobel


def norm(arr):
    return (arr-np.min(arr))/(np.max(arr)-np.min(arr))

def edg(img):
    res = np.zeros_like(img, dtype=np.float)
    for i in range(img.shape[2]):
        res[:,:,i] = sobel(img[:,:,i])
    res = norm(np.linalg.norm(res, axis=2))
    return res

def change_colors(img):
    n = np.max(img)
    res = np.zeros((img.shape[0], img.shape[1], 3))
    cols = np.random.randint(0, 256, size=(n, 3))
    p = np.arange(0, n)
    np.random.shuffle(p)
    for i in range(n):
        res[img == i] = cols[i]
    return res