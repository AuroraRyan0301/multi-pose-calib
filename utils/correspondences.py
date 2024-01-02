import numpy as np 
from sklearn.neighbors import NearestNeighbors

from .mesh import TriMesh 
from .intersect import Intersect
from .octree import OCTree

class Correspondences:
	def __init__(self):
		pass

	def nearest_tri_normal(self, src, tgt, octree = None, con_ind = None, dist_threshold = 0.03, normal_threshold = 0.9):
		if not hasattr(self, 'tester'):
			self.tester = Intersect()
		if src.vert_normal is None:
			src.cal_vert_normal()
		tgt.cal_face_normal()

		if octree is None:
			octree = OCTree()
			octree.from_triangles(tgt.vertices, tgt.faces, np.arange(tgt.face_num()))

		ray_ind, face_ind, weights, dist = self.tester.rays_octree_intersect(octree, \
			tgt.vertices, tgt.faces, tgt.face_normal, \
			src.vertices, src.vert_normal, np.arange(src.vert_num()), dist_threshold, normal_threshold)
		weights = np.reshape(weights, (-1, 3))

		tgt_face_ind = np.full(src.vert_num(), -1, dtype = np.int32)
		tgt_weights = np.full((src.vert_num(), 3), 0, dtype = np.float32)
		tgt_dist = np.full(src.vert_num(), np.inf, dtype = np.float32)

		for i in range(len(ray_ind)):
			if not con_ind is None: 
				if ray_ind[i] in con_ind and face_ind[i] in con_ind[ray_ind[i]]:
					continue
					
			if tgt_dist[ray_ind[i]] > dist[i]:
				tgt_dist[ray_ind[i]] = dist[i]
				tgt_face_ind[ray_ind[i]] = face_ind[i]
				tgt_weights[ray_ind[i]] = weights[i]

		src_ind = np.where(tgt_dist < dist_threshold)[0]

		return src_ind, tgt_face_ind[src_ind], tgt_weights[src_ind]