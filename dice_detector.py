from node import img_to_nodes, Node
from comparator import Comparator
from skimage.measure import find_contours
from skimage.draw import polygon_perimeter
import skimage.morphology as mp
from tqdm import tqdm
import numpy as np
from operator import itemgetter
from skimage.color import rgb2gray

class DiceDetector():

    def generate_fake_node(self, a, b, ratio):
        a = np.array(a.center)
        b = np.array(b.center)
        v = b-a
        u = np.array([v[1], -v[0]])
        u*=ratio/2
        mid = (a+b)/2
        n = Node(-1, None, None, True)
        n.center = mid+u
        return n
        
    def detect_with_node(self, node, dots):
        best = 1e9
        best_nodes = []
        best_regions = []
        if dots > 3:
            for i in node.neighbours:
                for j in node.neighbours:
                    if i == j:
                        continue
                    nodes = [node, self.nodes[i], self.nodes[j]]
                    penalty, regions = self.comparator.compare(nodes, dots)
                    if penalty < best:
                        best = penalty
                        best_nodes = nodes
                        best_regions = regions
        elif dots in {2, 3}:
            for i in node.neighbours:
                for ratio in [0.5, 0.75, 1, 1.33, 2]:
                    nodes = [node, self.nodes[i], self.generate_fake_node(node, self.nodes[i], ratio)]
                    # print([n.center for n in nodes])
                    penalty, regions = self.comparator.compare(nodes, dots)
                    if penalty < best:
                        best = penalty
                        best_nodes = nodes
                        best_regions = regions
        else:
            pass #TODO
        return best, best_nodes, best_regions


    def detect_single(self, img, mask):
        thresholds = [0.4]*7
        thresholds[2] = 0.2
        labels, self.nodes = img_to_nodes(img, mask)
        self.comparator = Comparator(self.nodes, labels)

        res = []

        for dots in range(6, 1, -1):
            matches = []
            for node in tqdm(self.nodes.values()):
                matches.append(self.detect_with_node(node, dots))

            matches.sort(key=itemgetter(0))
            for m in matches:
                if m[0] > thresholds[dots]:
                    break
                vis = False
                for n in m[1]:
                    if n.visited:
                        vis = True
                if vis:
                    continue
                pattern = self.comparator.get_pattern(m[1], dots)
                res.append((find_contours(pattern, 0.5)[0], dots))
                print(m[0])
                for n in m[2]:
                    print(n.a, end=' ')
                    n.visited = True
                print(' ')

        return res

    def detect(self, img):
        gray = rgb2gray(img)
        minv = np.percentile(gray, 3)
        maxv = np.percentile(gray, 97)
        gray = (gray-minv)/(maxv-minv)
        self.res = self.detect_single(img, gray < 0.3)
        self.res+= self.detect_single(img, gray > 0.4)
        return self.res

    def mark_sides(self, img):
        colors = np.array([
            (0, 0, 0),
            (1, 0, 0),
            (1, 1, 0),
            (0, 1, 0),
            (0, 1, 1),
            (0, 0, 1),
            (1, 0, 1)
        ])
        for c in self.res:
            border = polygon_perimeter(c[0][:,0], c[0][:,1])
            border_mask = np.zeros((img.shape[0], img.shape[1]))
            border_mask[border] = 1
            border_mask = mp.dilation(border_mask)
            img[border_mask==1] = colors[c[1]]*255
        return img
