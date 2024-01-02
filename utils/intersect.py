import numpy as np 
from .mesh import TriMesh

class Intersect:
	def __init__(self):
		pass

	def rays_tris_intersect(self, rays_start, rays_dir, tris):
		v0v1 = tris[:, 1, :] - tris[:, 0, :] 
		v0v2 = tris[:, 2, :] - tris[:, 0, :]

		pvec = np.cross(rays_dir, v0v2, axis = 1) 
		det = np.sum(v0v1 * pvec, axis = 1)
		check_ind = np.where(np.abs(det) > 1e-12)[0]
		inv_det = 1 / det[check_ind]

		u = np.full(len(rays_start), -1).astype(np.float32)
		v = np.full(len(rays_start), -1).astype(np.float32)
		t = np.full(len(rays_start), np.inf).astype(np.float32)

		tvec = rays_start[check_ind] - tris[check_ind, 0, :]
		u[check_ind] = np.sum(tvec * pvec[check_ind], axis = 1) * inv_det 

		qvec = np.cross(tvec, v0v1[check_ind])
		v[check_ind] = np.sum(qvec * rays_dir[check_ind], axis = 1) * inv_det 

		t[check_ind] = np.sum(v0v2[check_ind] * qvec, axis = 1) * inv_det

		return u, v, t

	def rays_aabb_intersect(self, rays_start, rays_dir, aabb_start, aabb_end, check_dir = False):
		dir_inv = np.zeros(rays_dir.shape, dtype = np.float32)
		dir_inv[np.abs(rays_dir) < 1e-10] = np.inf 
		dir_inv[np.abs(rays_dir) >= 1e-10] = np.divide(1.0, rays_dir[np.abs(rays_dir) >= 1e-10])

		t1 = (aabb_start - rays_start) * dir_inv
		t2 = (aabb_end - rays_start) * dir_inv 

		t = np.array([t1, t2])
		tmin = np.min(t, axis = 0)
		tmax = np.max(t, axis = 0)

		t_enter = np.max(tmin, axis = 1)
		t_exit = np.min(tmax, axis = 1)

		if check_dir:
			return np.logical_and(t_enter < t_exit, t_exit >= 0)
		return t_enter < t_exit

	def rays_octree_intersect(self, octree, vertices, faces, face_normals, rays_start, rays_dir, ray_ind, dist_threshold, normal_threshold):
		if not octree.item_ind is None:
			rays_num = len(ray_ind)
			tris_num = len(octree.item_ind)

			ray_ind_repeat = ray_ind.repeat(tris_num)
			u, v, t = self.rays_tris_intersect(rays_start[ray_ind_repeat],\
				rays_dir[ray_ind_repeat], \
				vertices[faces[np.tile(octree.item_ind, rays_num)]])

			u = u.reshape((rays_num, tris_num))
			v = v.reshape((rays_num, tris_num))
			t = t.reshape((rays_num, tris_num))

			intersect_mask = np.logical_and(np.logical_and(np.logical_and(u <= 1, u >= 0), np.logical_and(v <= 1, v >= 0)), u + v <= 1)
			t[np.where(~intersect_mask)] = np.inf
			t[np.where(t > dist_threshold)] = np.inf

			int_ray_ind, int_face_ind = np.where(t < np.inf - 1)
			dist = np.abs(t[int_ray_ind, int_face_ind])
			weights = np.column_stack(((1 - u - v)[int_ray_ind, int_face_ind], u[int_ray_ind, int_face_ind], v[int_ray_ind, int_face_ind]))
			int_ray_ind = ray_ind[int_ray_ind]
			int_face_ind = octree.item_ind[int_face_ind]

			cos = np.sum(rays_dir[int_ray_ind] * face_normals[int_face_ind], axis = 1)
			n_valid_mask = cos > normal_threshold

			return int_ray_ind[n_valid_mask], int_face_ind[n_valid_mask], weights[n_valid_mask].flatten(), dist[n_valid_mask].flatten() 
		else:
			int_ray_ind = np.array([], dtype = np.int32)
			int_face_ind = np.array([], dtype = np.int32)
			weights = np.array([], dtype = np.float32)
			dist = np.array([], dtype = np.float32)

			for i in range(len(octree.childs)):
				ray_check_mask = self.rays_aabb_intersect(rays_start[ray_ind], rays_dir[ray_ind], octree.childs[i].start, octree.childs[i].end)
				ray_check_ind = ray_ind[ray_check_mask]
				if len(ray_check_ind) == 0:
					continue 

				child_int_ray_ind, child_int_face_ind, child_weights, childs_dist = self.rays_octree_intersect(\
					octree.childs[i], \
					vertices, faces, face_normals, \
					rays_start, rays_dir, ray_check_ind, dist_threshold, normal_threshold)

				int_ray_ind = np.append(int_ray_ind, child_int_ray_ind)
				int_face_ind = np.append(int_face_ind, child_int_face_ind)
				weights = np.append(weights, child_weights)
				dist = np.append(dist, childs_dist)

			return int_ray_ind, int_face_ind, weights, dist