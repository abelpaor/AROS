import numpy as np
import sys

import trimesh


class Trainer:
    normal_env = np.asarray([])

    pv_points = np.asarray([])
    pv_vectors = np.asarray([])
    pv_norms = np.asarray([])

    pv_max_norm = sys.float_info.min
    pv_min_norm = sys.float_info.max
    pv_mapped_norms = np.asarray([])

    sampler = None

    def __init__(self, tri_mesh_ibs, tri_mesh_env, sampler):
        self.sampler = sampler
        self._get_env_normal(tri_mesh_env)
        self._get_provenance_vectors(tri_mesh_ibs, tri_mesh_env)
        self._set_pv_min_max_mapped_norms()
        self._get_mapped_norms()
        self._order_by_mapped_norms()

    def _order_by_mapped_norms(self):
        idx_order = np.argsort(self.pv_mapped_norms)[::-1]
        self.pv_points = self.pv_points[idx_order]
        self.pv_vectors = self.pv_vectors[idx_order]
        self.pv_norms = self.pv_norms[idx_order]
        self.pv_mapped_norms = self.pv_mapped_norms[idx_order]

    def _get_provenance_vectors(self, tri_mesh_ibs, tri_mesh_env):
        self.sampler.execute(tri_mesh_ibs, tri_mesh_env)
        self.pv_points = self.sampler.pv_points
        self.pv_vectors = self.sampler.pv_vectors
        self.pv_norms = self.sampler.pv_norms

    def _get_env_normal(self, tri_mesh_env):
        (_, __, triangle_id) = tri_mesh_env.nearest.on_surface(np.array([0, 0, 0]).reshape(-1, 3))
        self.normal_env = tri_mesh_env.face_normals[triangle_id]

    def _map_norm(self, norm, max, min):
        return (norm - min) * (0 - 1) / (max - min) + 1
        # return(value_in-min_original_range)*(max_mapped-min_mapped)/(max_original_range-min_original_range)+min_mapped

    def _set_pv_min_max_mapped_norms(self):
        self.pv_max_norm = self.pv_norms.max()
        self.pv_min_norm = self.pv_norms.min()

    def _get_mapped_norms(self):
        self.pv_mapped_norms = np.asarray(
            [self._map_norm(norm, self.pv_max_norm, self.pv_min_norm) for norm in self.pv_norms])

    def get_info(self):
        info = {}
        info['normal_env'] = str(self.normal_env[0][0]) + ',' + str(self.normal_env[0][1]) + ',' + str(self.normal_env[0][2])
        info['pv_max_norm'] = self.pv_max_norm
        info['pv_min_norm'] = self.pv_min_norm
        info['sampler'] = self.sampler.get_info()
        return info




class TrainerClearance(Trainer):
    cv_sampler = None
    cv_points = np.asarray([])
    cv_vectors = np.asarray([])
    cv_norms = np.asarray([])

    def __init__(self, tri_mesh_ibs, tri_mesh_env, tri_mesh_obj, pv_sampler, cv_sampler):
        """
        It tries to generates the clearance vectors using only the IBS surface and points related to
        Parameters
        ----------
        tri_mesh_ibs: IBS mesh
        tri_mesh_env: Environment mesh
        tri_mesh_obj: Object mesh
        pv_sampler: sample used to select provenance vectors
        cv_sampler: sampler used to select clearance vector
        """
        super().__init__(tri_mesh_ibs, tri_mesh_env, pv_sampler)
        self.cv_sampler = cv_sampler
        self._get_clearance_vectors(tri_mesh_ibs, tri_mesh_obj)

    def _get_clearance_vectors(self, tri_mesh_ibs, tri_mesh_obj):
        self.cv_sampler.execute(tri_mesh_ibs, tri_mesh_obj)
        self.cv_points = self.cv_sampler.cv_points
        self.cv_vectors = self.cv_sampler.cv_vectors
        self.cv_norms = self.cv_sampler.cv_norms

    def get_info(self):
        info = super().get_info()
        info['cv_sampler'] = self.cv_sampler.get_info()
        return info
