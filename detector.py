
from utils.octree import OCTree
from utils.mesh import TriMesh
from utils.correspondences import Correspondences
from sklearn.mixture import GaussianMixture

import numpy as np
import os
from PIL import Image
from tqdm import tqdm

import xml.dom.minidom


class LdmkProjector:
	def __init__(self):
		self.cor_builder = Correspondences()
		
	def solve_eq_size2(self, coef, dest):
		d = coef[0, 0] * coef[1, 1] - coef[1, 0] * coef[0, 1]
		dx = dest[:, 0] * coef[1, 1] - dest[:, 1] * coef[0, 1]
		dy = dest[:, 1] * coef[0, 0] - dest[:, 0] * coef[1, 0]
		return dx / d, dy / d 

	def inverse_project(self, scan, scan_octree, lmk, cam_mat, trans_mat):
		# import ipdb; ipdb.set_trace()
		# inv = np.linalg.inv(cam_mat)
		# lmk_mine = np.array([lmk[0, 0],lmk[0, 1], 1])
		# result = inv.dot(lmk_mine)
		# # result 添加一个0
		# result = np.insert(result,3,0)

		lmk_3d_x, lmk_3d_y = self.solve_eq_size2(cam_mat[:2, :2], np.column_stack((lmk[:, 0] - cam_mat[0, 2], lmk[:, 1] - cam_mat[1, 2])))
		lmk_3d_norm =  np.column_stack((lmk_3d_x, lmk_3d_y, np.ones(len(lmk))))

		tm_inv = np.linalg.inv(trans_mat)

		# rotate_result = tm_inv.dot(result)

		lmk_3d_norm = np.hstack((lmk_3d_norm, np.ones((len(lmk_3d_norm), 1)))).dot(tm_inv.T)[:, :3]
		ray_start = np.tile(tm_inv[:3, 3], len(lmk_3d_x)).reshape((-1, 3))
		ray_dir = lmk_3d_norm - ray_start
		ray_dir = ray_dir / (np.sqrt(np.sum(ray_dir ** 2, axis = 1))[:, np.newaxis] + 1e-12)

		ray_mesh = TriMesh()
		ray_mesh.vertices = ray_start
		ray_mesh.vert_normal = ray_dir

		ray_ind, tgt_face_ind, weights = self.cor_builder.nearest_tri_normal(ray_mesh, scan, scan_octree, dist_threshold = np.inf - 10, normal_threshold = -1)
		
		return ray_ind, tgt_face_ind, weights

	def gmm_fit(self, verts, num = 1):
		if len(verts.flatten()) <= 3:
			return verts
		gmm = GaussianMixture(n_components = num, random_state = 0).fit(verts)
		ind = np.argmax(gmm.weights_)
		return gmm.means_[ind]

	def ldmk_3d_detect(self, scan, lmk2d_lst, cam_mat_lst, trans_mat_lst):
		octree = OCTree()
		octree.from_triangles(scan.vertices, scan.faces, np.arange(scan.face_num()))
		scan.cal_face_normal()

		view_num = len(lmk2d_lst)
		ldmk_num = lmk2d_lst[0].shape[0]

		lmk3d_lst = np.zeros((view_num, ldmk_num, 3), dtype = np.float32)
		normal_lst = np.zeros((view_num, ldmk_num, 3), dtype = np.float32)
		lmk3d_mask = np.full((view_num, ldmk_num), False)

		for i in range(len(lmk2d_lst)):
			src_ind, tgt_face_ind, weights = self.inverse_project(scan, octree, lmk2d_lst[i], cam_mat_lst[i], trans_mat_lst[i])
			lmk3d_lst[i, src_ind, :] = np.sum(scan.vertices[scan.faces[tgt_face_ind]] * weights[:, :, np.newaxis], axis = 1)
			normal_lst[i, src_ind, :] = scan.face_normal[tgt_face_ind]
			lmk3d_mask[i, src_ind] = True 

		lmk3d = np.zeros((ldmk_num, 3), dtype = np.float32)

		for i in range(ldmk_num):
			lmk3d_i = lmk3d_lst[:, i, :]
			mask_i = lmk3d_mask[:, i]
			lmk3d_i = lmk3d_i[mask_i]

			ray_mesh = TriMesh()
			ray_mesh.vertices = self.gmm_fit(lmk3d_i).reshape((-1, 3))
			ray_mesh.vert_normal = np.mean(normal_lst[:, i, :], axis = 0).reshape((-1, 3))
			src_ind, tgt_face_ind, weights = self.cor_builder.nearest_tri_normal(ray_mesh, scan, octree, dist_threshold = np.inf - 10, normal_threshold = -1)

			if len(src_ind) == 0:
				print(f'3d landmark err!: {i}')
			if len(src_ind) != 0:
				lmk3d[i] = np.sum(scan.vertices[scan.faces[tgt_face_ind]] * weights[:, :, np.newaxis], axis = 1).flatten()
		
		return lmk3d

def parse_cam_param(xml_path, cam_idx_lst):
	root = xml.dom.minidom.parse(xml_path)
	cam_params = root.getElementsByTagName('camera')
	sensor_params = root.getElementsByTagName('sensor')

	cam_mat_lst = []
	trans_mat_lst = []
	valid_lst = []

	for cam_idx in cam_idx_lst:
		if '.jpg' in cam_idx or '.JPG' in cam_idx or '.png' in cam_idx or '.PNG' in cam_idx:
			cam_idx = cam_idx[:-4]
		cam_param = None
		sensor_param = None

		for p in cam_params:
			if p.getAttribute('label') == cam_idx:
				cam_param = p
				break

		sensor_id = cam_param.getAttribute('sensor_id')
		for p in sensor_params:
			if p.getAttribute('id') == sensor_id:
				sensor_param = p 
				break

		width = int(sensor_param.getElementsByTagName('resolution')[0].getAttribute('width'))
		height = int(sensor_param.getElementsByTagName('resolution')[0].getAttribute('height'))
	
		f = float(sensor_param.getElementsByTagName('f')[0].firstChild.data)
		cx = float(sensor_param.getElementsByTagName('cx')[0].firstChild.data)
		cy = float(sensor_param.getElementsByTagName('cy')[0].firstChild.data)

		cam_mat = np.zeros((3, 3))
		cam_mat[2, 2] = 1.0

		# if width > height:
		# 	cam_mat[0, 1] = -f 
		# 	cam_mat[1, 0] = f 
		# 	cam_mat[0, 2] = height / 2 - cy
		# 	cam_mat[1, 2] = width / 2 + cx
		# else:
		cam_mat[0, 0] = f 
		cam_mat[1, 1] = f 
		cam_mat[0, 2] = width / 2 + cx 
		cam_mat[1, 2] = height / 2 + cy

		if len(cam_param.getElementsByTagName('transform')) == 0:
			valid_lst.append(False)
			continue

		transform = cam_param.getElementsByTagName('transform')[0].firstChild.data
		ss = transform.split(' ')
		trans_mat = np.zeros((4, 4))
		for i in range(4):
			for j in range(4):
				trans_mat[i, j] = float(ss[i * 4 + j])

		trans_mat = np.linalg.inv(trans_mat)

		cam_mat_lst.append(cam_mat) 
		trans_mat_lst.append(trans_mat)
		valid_lst.append(True)
	
	return cam_mat_lst, trans_mat_lst, valid_lst

def gmm_fit(verts, num = 1):
    if len(verts.flatten()) <= 3:
        return verts[0]
    gmm = GaussianMixture(n_components = num, random_state = 0).fit(verts)
    ind = np.argmax(gmm.weights_)
    return gmm.means_[ind]