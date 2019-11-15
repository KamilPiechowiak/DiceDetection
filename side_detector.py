import numpy as np
from operator import itemgetter
import matplotlib.pyplot as plt
from skimage.draw import line
import skimage.filters as fl

from node import Node
from utils import edg, norm_with_percentiles

EPS = 1e-6
ANGLES = 12
MIN_RADIUS = 2
MIN_SIN = 0.8

class SideDetector:

    def __init__(self, img):
        self.edg = norm_with_percentiles(edg(img), 3, 98)
        self.edg = fl.median(self.edg)
        self.img = np.stack([self.edg]*3, axis=-1)
    
    def get_line_point_angle(self, a, theta, d_min, d_max):
        v = np.array([np.cos(theta), np.sin(theta)])
        v/= np.linalg.norm(v)
        b = a+v*d_min
        c = a+v*d_max
        return self.get_line_point_point(b, c)

    def get_line_point_point(self, a, b): #a, b - points
        a, b = np.round(a).astype(int), np.round(b).astype(int)
        return line(a[0], a[1], b[0], b[1])

    def get_angle_sin(self, a, b): #a, b - segments
        return (a[0]*b[1]-a[1]*b[0])/np.linalg.norm(a)/np.linalg.norm(b)

    def segment_equation(self, a):
        return [a[0][1]-a[1][1], a[1][0]-a[0][0], a[0][1]*a[1][0]-a[1][1]*a[0][0]]

    def det(self, a):
        return a[0][0]*a[1][1]-a[0][1]*a[1][0]

    def intersection(self, a, b):
        c = self.segment_equation(a)
        d = self.segment_equation(b)
        w = self.det([[c[0], c[1]], [d[0], d[1]]])
        w_x = self.det([[c[2], c[1]], [d[2], d[1]]])
        w_y = self.det([[c[0], c[2]], [d[0], d[2]]])
        return w_x/w, w_y/w

    def get_line_value(self, a, b):
        rr, cc = self.get_line_point_point(a, b)
        return np.mean(self.edg[rr, cc])
    
    def propose_nodes(self, node, a, b): #a, b - segments
        center = np.array(node.center)
        p = self.intersection(a, b)
        q = p+2*(center-p)
        u = b[1]-b[0]
        r = self.intersection(a, np.array([q, q+u]))
        s = r+2*(center-r)
        n1 = Node(-1, None, None, True)
        n1.center = (center+p)/2
        n2 = Node(-1, None, None, True)
        n2.center = (center+r)/2
        # self.draw_line(p, r)
        # self.draw_line(q, r, [0, 1, 0])
        self.img[int(n1.center[0]), int(n1.center[1])] = [0, 1, 1]
        self.img[int(n2.center[0]), int(n2.center[1])] = [0, 1, 1]
        # plt.imshow(self.img)
        # plt.show()
        try:
            val = self.get_line_value(p, r)+self.get_line_value(p, s)
        except:
            val = np.inf
        # val+= self.get_line_value(q, r)+self.get_line_value(q, s)
        val/=2
        return val, [node, n1, n2]
    
    def draw_line(self, a, b, col=[1, 0, 0]):
        rr, cc = line(int(a[0]), int(a[1]), int(b[0]), int(b[1]))
        self.img[rr, cc] = col

    def detect(self, node):
        self.img[int(node.center[0]), int(node.center[1])] = [0, 1, 0]
        threshold = 0.8
        points = []
        n = ANGLES
        r = (node.area/np.pi)**(0.5)
        if r < MIN_RADIUS:
            return -1, []
        for theta in range(0, 360, 360//n):
            l = self.get_line_point_angle(np.array(node.center), np.deg2rad(theta), 2*r, 8*r)
            if np.min(l[0]) < 0 or np.max(l[0]) >= self.edg.shape[0] or np.min(l[1]) < 0 or np.max(l[1]) > self.edg.shape[1]:
                return -1, []
            p = self.edg[l[0], l[1]]
            maxv = np.max(p[1:-1])
            idx = (p[1:-1] >= p[:-2]) & (p[1:-1] >= p[2:]) & (p[1:-1] >= min(threshold, maxv))
            id = np.argmax(idx)+1
            points.append((l[0][id], l[1][id]))
            self.img[int(l[0][id]), int(l[1][id])] = [1, 1, 0]
        
        proposals = []
        # cols = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        for i in range(n):
            a, b = points[i], points[-n+1+i]
            # rr, cc = line(a[0], a[1], b[0], b[1])
            # img[rr, cc] = cols[i%3]
            # print(a, b)
            idx = self.get_line_point_point(np.array(a, dtype=np.float), np.array(b, dtype=np.float))
            val = np.mean(self.edg[idx[0], idx[1]])
            proposals.append((np.array(a), np.array(b), val))

        proposals.sort(key=itemgetter(2), reverse=True)


        pairs = [(0, 1), (0, 2), (1, 2), (0, 3), (0, 4), (0, 5)]
        found = False
        for a, b in pairs:
            val = self.get_angle_sin(proposals[a][1]-proposals[a][0], proposals[b][1]-proposals[b][0])
            if abs(val) > MIN_SIN:
                found = True
                self.draw_line(proposals[a][0], proposals[a][1], [1,0,1])
                self.draw_line(proposals[b][0], proposals[b][1], [1,0,1])
                break
        if found == False:
            return -1, []

        return self.propose_nodes(node, np.array(proposals[a][0:2]), np.array(proposals[b][0:2]))
        