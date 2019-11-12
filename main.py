import numpy as np
from skimage.filters import median
from skimage.color import rgb2hsv, rgb2gray
from skimage import io

from dice_detector import DiceDetector
from node import img_to_nodes
from comparator import Comparator
from utils import change_colors

import matplotlib.pyplot as plt

def analyze():
    k = 8
    for i in range(k, k+1):
        img = io.imread('small/' + str(i) + '.jpg')
        img = median(img)
        # res = img_to_nodes(img)
        # res = change_colors(res)
        # # res = sg.mark_boundaries(img, res)
        # res = np.array(res, dtype=np.float)

        dd = DiceDetector()
        dd.detect(img)
        img = dd.mark_sides(img)
        io.imsave('m/' + str(i) + '.png', img)

        # labels, nodes = img_to_nodes(img, rgb2gray(img) < 0.5)
        # for n in nodes.values():
        #     print(len(n.neighbours))
        # c = Comparator(nodes, labels)
        # c.compare([], 3)
        # c.compare([], 6)
        # c.compare([], 6)
        # plt.imshow(mask)
        # plt.show()
        # io.imsave('n/' + str(i) + '.png', change_colors(labels))

analyze()