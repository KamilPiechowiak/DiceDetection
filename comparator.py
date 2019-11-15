from skimage import io
import skimage.transform as tr
import skimage.segmentation as sg
import numpy as np

import matplotlib.pyplot as plt

from node import Node

MIN_SIZE = 20
MAX_SIZE = 0.02
INF = 1e18

class Comparator:
    def __init__(self, all_nodes, labels):
        self.patterns = [{}, {}]
        self.vertices = [{}, {}]
        self.all_vertices = [{}, {}]
        self.black_area = [{}, {}]
        for i in range(6, 0, -1):
            self.init_pattern(i, 0)
        for i in range(6, 0, -1):
            self.init_pattern(i, 1)
        self.all_nodes = all_nodes
        self.labels = labels

    def init_pattern(self, pattern_id, pattern_size):
        coordinates = [[2], [2,3], [2,3], [2,3,4], [2,3,4], [2,3,6]]
        dir_path = 'rs/'
        if pattern_size == 1:
            dir_path = 'ps/'
        img = io.imread(dir_path + str(pattern_id) + '.png')
        pattern = np.array(img < 127, dtype=np.int)
        self.black_area[pattern_size][pattern_id] = np.count_nonzero(pattern)
        num = 2
        while True:
            ids = np.argwhere(pattern == 1)
            if len(ids) == 0:
                break
            pattern = sg.flood_fill(pattern, (ids[0][0], ids[0][1]), np.int(num))
            num+=1
        
        vertices = []
        for i in coordinates[pattern_id-1]:
            ids = np.argwhere(pattern == i)
            vertices.append((np.mean(ids[:,0]), np.mean(ids[:,1])))
        
        if pattern_id == 2 or pattern_id == 3:
            vertices.append(self.vertices[pattern_size][4][2])
        if pattern_id == 1:
            vertices.append(self.all_vertices[pattern_size][4][2])
            vertices.append(self.all_vertices[pattern_size][4][3])
        
        self.patterns[pattern_size][pattern_id] = pattern
        self.vertices[pattern_size][pattern_id] = vertices

        all_vertices = []
        for i in range(2, pattern_id+2):
            ids = np.argwhere(pattern == i)
            all_vertices.append((np.mean(ids[:,0]), np.mean(ids[:,1])))
        self.all_vertices[pattern_size][pattern_id] = np.array(all_vertices)
    

    def prepare_pattern(self, nodes, pattern_id, pattern_size, reduce=True):
        src = np.array(self.vertices[pattern_size][pattern_id])
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
        dst[:,[0,1]] = dst[:,[1,0]]
        assert len(src) == len(dst)
        at = tr.AffineTransform()
        at.estimate(dst, src)
        pattern = tr.warp(np.array(self.patterns[pattern_size][pattern_id], dtype=np.float), at, output_shape=output_shape, order=0, cval=-1)
        pattern = np.array(pattern, dtype=np.int)
        
        at2 = tr.AffineTransform()
        at2.estimate(src[:,[1,0]], dst[:,[1,0]])
        self.current_coord = at2(self.all_vertices[pattern_size][pattern_id])
        self.current_coord[:,0]+=self.shift_x
        self.current_coord[:,1]+=self.shift_y
        return pattern

    def color2seq(self, pattern, color):
        if type(color) is tuple:
            ids = np.argwhere((color[0] <= pattern) & (pattern <= color[1]))
        else:
            ids = np.argwhere(pattern == color)

        return ( np.clip(ids[:,0]+self.shift_x, 0, self.labels.shape[0]-1),
                 np.clip(ids[:,1]+self.shift_y, 0, self.labels.shape[1]-1))

    def match_pattern(self, pattern, num):
        area = np.count_nonzero(pattern >= 0)
        all_seq = self.color2seq(pattern, (0, num+1))
        if all_seq[0].shape[0] == 0 or np.max(all_seq[0])-np.min(all_seq[0]) < MIN_SIZE or np.max(all_seq[1])-np.min(all_seq[1]) < MIN_SIZE:
            return INF, []
        if area > self.labels.size*MAX_SIZE or area < MIN_SIZE**2:
            return INF, []
        all_regions = np.unique(self.labels[all_seq])
        region_area = {r: np.count_nonzero(self.labels[all_seq] == r) for r in all_regions}
        region_black_area = {r: 0 for r in all_regions}

        for color in range(2, num+2):
            seq = self.color2seq(pattern, color)
            regions = np.unique(self.labels[seq])
            for r in regions:
                count = np.count_nonzero(self.labels[seq] == r)
                region_black_area[r]+=count
        
        maxv, white_region = 0, 0
        for r in all_regions:
            a = region_area[r]-region_black_area[r]
            if a > maxv:
                maxv, white_region = a, r
        
        for c in self.current_coord:
            x, y, = int(c[0]), int(c[1])
            if x < 0 or x >= self.labels.shape[0] or y < 0 or y >= self.labels.shape[1]:
                return INF, []
            if self.labels[x, y] == white_region:
                return INF, []
        
        penalty = 0
        seq = self.color2seq(pattern, 0)
        penalty_black=np.count_nonzero(self.labels[seq] != white_region)
        seq = self.color2seq(pattern, (2, num+1))
        penalty_white=np.count_nonzero(self.labels[seq] == white_region)
        penalty=(penalty_white+penalty_black)/max(seq[0].shape[0], 1e-6)

        if all_regions.size < 3 and num >= 2:
            return INF, []        

        return penalty, [self.all_nodes[r] for r in all_regions]

    def compare(self, nodes, pattern_id):
        pattern = self.prepare_pattern(nodes, pattern_id, 0, reduce=True)
        return self.match_pattern(pattern, pattern_id)
    
    def get_pattern(self, nodes, pattern_id):
        pattern = self.prepare_pattern(nodes, pattern_id, 1, reduce=False)
        return pattern >= 0