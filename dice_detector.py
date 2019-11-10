from node import img_to_nodes, Node
from comparator import Comparator
from skimage.measure import find_contours
from skimage.draw import polygon_perimeter
import skimage.morphology as mp
from tqdm import tqdm
import numpy as np
from operator import itemgetter

class DiceDetector():
        
    def detect_with_node(self, node, dots):
        best = 1e9
        best_nodes = []
        if dots > 3:
            for i in node.neighbours:
                for j in node.neighbours:
                    nodes = [node, self.nodes[i], self.nodes[j]]
                    penalty = self.comparator.compare(nodes, dots)
                    if penalty < best:
                        best = penalty
                        best_nodes = nodes
        elif dots in {2, 3}:
            for i in node.neighbours:
                nodes = [node, self.nodes[i]]
                penalty = self.comparator.compare(nodes, dots)
                if penalty < best:
                    best = penalty
                    best_nodes = nodes
        else:
            pass #TODO
        return best, best_nodes


    def detect(self, img):
        threshold = 0.6
        labels, self.nodes = img_to_nodes(img)
        self.comparator = Comparator(self.nodes, labels)

        self.res = []

        for dots in range(6, 3, -1):
            matches = []
            for node in tqdm(self.nodes.values()):
                matches.append(self.detect_with_node(node, dots))

            matches.sort(key=itemgetter(0))
            for m in matches:
                if m[0] > threshold:
                    break
                vis = False
                for n in m[1]:
                    if n.visited:
                        vis = True
                if vis:
                    continue
                pattern = self.comparator.get_pattern(m[1], dots)
                self.res.append((find_contours(pattern, 0.5)[0], dots))
                for n in m[1]:
                    print(n.a, end=' ')
                    n.visited = True
                print(' ')

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
