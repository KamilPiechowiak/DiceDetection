import numpy as np
import skimage.segmentation as sg
from skimage.future import graph
import skimage.measure as me

from utils import edg

def get_neighbours(a, graph, d=2):
        if d == 1:
            return set(graph[a])
        res = set()
        for b in graph[a]:
            res|= get_neighbours(b, graph, d-1)
        res-=set(graph[a])
        res-=set([a])
        return res

class Node:
    def __init__(self, a, graph, indices, fake=False):
        self.visited = False
        self.a = a
        if fake:
            return
        self.graph = graph
        self.indices = indices
        self.neighbours = get_neighbours(self.a, self.graph, 2)
        self.center = (np.mean(indices[:,0]), np.mean(indices[:,1]))
        self.area = indices.shape[0]
    
    def update_neighbours(self, nodes, img_size):
        r = np.sqrt(self.area/np.pi)
        to_remove = []
        for i in self.neighbours:
            n = nodes[i]
            dist = np.linalg.norm(np.array(self.center)-np.array(n.center))
            area_ratio = self.area/n.area
            if dist > 8*r:
                to_remove.append(i)
            elif n.area > img_size/40:
                to_remove.append(i)
            elif area_ratio > 2 or area_ratio < 1/2:
                to_remove.append(i)

        self.neighbours-=set(to_remove)
        if self.area > img_size/40:
            self.neighbours = set()

def img_to_nodes(img, mask):

    def weight_boundary(graph, src, dst, n):
        """
        Handle merging of nodes of a region boundary region adjacency graph.

        This function computes the `"weight"` and the count `"count"`
        attributes of the edge between `n` and the node formed after
        merging `src` and `dst`.


        Parameters
        ----------
        graph : RAG
            The graph under consideration.
        src, dst : int
            The vertices in `graph` to be merged.
        n : int
            A neighbor of `src` or `dst` or both.

        Returns
        -------
        data : dict
            A dictionary with the "weight" and "count" attributes to be
            assigned for the merged node.

        """
        default = {'weight': 0.0, 'count': 0}

        count_src = graph[src].get(n, default)['count']
        count_dst = graph[dst].get(n, default)['count']

        weight_src = graph[src].get(n, default)['weight']
        weight_dst = graph[dst].get(n, default)['weight']

        count = count_src + count_dst
        return {
            'count': count,
            'weight': (count_src * weight_src + count_dst * weight_dst)/count
        }

    def merge_boundary(graph, src, dst):
        """Call back called before merging 2 nodes.

        In this case we don't need to do any computation here.
        """
        pass

    def separate_regions(labels, graph):
        indices = {}
        for x in range(labels.shape[0]):
            for y in range(labels.shape[1]):
                id = labels[x][y]
                if id not in indices:
                    indices[id] = []
                indices[id].append((x, y))
        nodes = {}
        for key, value in indices.items():
            nodes[key] = Node(key, graph, np.array(value))
        for n in nodes.values():
            n.update_neighbours(nodes, labels.size)
        return nodes

    edges = edg(img)
    # labels = sg.slic(img, compactness=10, min_size_factor=0.001)
    labels = me.label(mask)
    g = graph.rag_boundary(labels, edges)
    labels = graph.merge_hierarchical(labels, g, thresh=0.2, rag_copy=False,
                                   in_place_merge=True,
                                   merge_func=merge_boundary,
                                   weight_func=weight_boundary)
    g = graph.rag_boundary(labels, edges)
    nodes = separate_regions(labels, g)

    return labels, nodes