from skimage import io
import skimage.transform as tr
import skimage.segmentation as sg
import numpy as np

import matplotlib.pyplot as plt

from node import Node

class Comparator:
    def __init__(self, all_nodes, labels):
        self.patterns = {}
        self.vertices = {}
        for i in range(1, 7):
            self.init_pattern(i)
        self.all_nodes = all_nodes
        self.labels = labels

    def init_pattern(self, pattern_id):
        max_coordinates = [1, 2, 2, 3, 3, 3]
        img = io.imread('p/' + str(pattern_id) + '.png')
        pattern = np.array(img[:,:,0] < 127, dtype=np.int)
        num = 2
        while True:
            ids = np.argwhere(pattern == 1)
            if len(ids) == 0:
                break
            pattern = sg.flood_fill(pattern, (ids[0][0], ids[0][1]), np.int(num))
            num+=1
        
        vertices = []
        for i in range(2, max_coordinates[pattern_id-1]+2):
            ids = np.argwhere(pattern == i)
            vertices.append((np.mean(ids[:,0]), np.mean(ids[:,1])))
        
        self.patterns[pattern_id] = pattern
        self.vertices[pattern_id] = vertices

    def prepare_pattern(self, nodes, pattern_id, reduce=True):
        src = np.array(self.vertices[pattern_id])
        src[:,[0,1]] = src[:,[1,0]]
        dst = np.array([n.center for n in nodes])
        if reduce:
            d = max(np.max(dst[:,0])-np.min(dst[:,0]), np.max(dst[:,1])-np.min(dst[:,1]))
            self.shift_x = int(max(0, np.min(dst[:,0])-2*d))
            self.shift_y = int(max(0, np.min(dst[:,1])-2*d))
            dst[:,0]-=self.shift_x
            dst[:,1]-=self.shift_y
            output_shape = (int(5*d), int(5*d))
        else:
            self.shift_x = 0
            self.shift_y = 0
            output_shape = self.labels.shape
        # dst = np.array([[540, 152], [558, 158], [550, 145]])
        # dst+=-100
        dst[:,[0,1]] = dst[:,[1,0]]
        assert len(src) == len(dst)
        at = tr.AffineTransform()
        at.estimate(dst, src)
        pattern = tr.warp(np.array(self.patterns[pattern_id], dtype=np.float), at, output_shape=output_shape, order=0, cval=-1)
        pattern = np.array(pattern, dtype=np.int)
        return pattern
    
    def match_pattern(self, pattern, num):
        penalty = 0
        for color in range(2, num+2):
            ids = np.argwhere(pattern == color)
            seq = (ids[:,0]+self.shift_x, ids[:,1]+self.shift_y)
            regions = np.unique(self.labels[seq])
            # print(regions)
            region_penalty = 0
            minv = np.inf
            region_inside = False
            for r in regions:
                count = np.count_nonzero(self.labels[seq] == r)
                node = self.all_nodes[r]
                # print(count, node.area)
                if count/node.area >  0.5:
                    region_penalty+= node.area-count
                    region_inside = True
                else:
                    region_penalty+= count
                    minv = min(minv, node.area-2*count)

            if region_inside == False:
                region_penalty+= minv
            penalty+= region_penalty/max(ids.shape[0], 1e-6)

        penalty/=num
        return penalty

    def compare(self, nodes, pattern_id):
        pattern = self.prepare_pattern(nodes, pattern_id, reduce=True)
        # plt.imshow(pattern)
        # plt.show()
        return self.match_pattern(pattern, pattern_id)
    
    def get_pattern(self, nodes, pattern_id):
        pattern = self.prepare_pattern(nodes, pattern_id, reduce=False)
        return pattern >= 0