import numpy as np
from skimage.color import rgb2hsv, rgb2gray
from skimage import io

from dice_detector import DiceDetector
from node import img_to_nodes
from comparator import Comparator
from utils import change_colors
from side_detector import SideDetector

def analyze():
    k = 5
    for i in range(24, 28):
        img = io.imread('small/' + str(i) + '.jpg')
        

        dd = DiceDetector()
        dd.detect(img)
        img = dd.mark_sides(img)
        io.imsave('m/' + str(i) + '.png', img)

        # gray = rgb2gray(img)
        # minv = np.percentile(gray, 3)
        # maxv = np.percentile(gray, 97)
        # gray = (gray-minv)/(maxv-minv)
        # labels, nodes = img_to_nodes(img, gray > 0.4)
        # # print(labels[297, 465])
        # sd = SideDetector(img)
        # sd.detect(nodes[2])#2, 28, 5
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