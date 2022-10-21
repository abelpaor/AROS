import numpy as np
from scipy.spatial import Voronoi

import trimesh

import aros.util as util


class IBS:

    def __init__(self, np_cloud_env, np_cloud_obj):

        self.size_cloud_env = np_cloud_env.shape[0]
        size_cloud_obj = np_cloud_obj.shape[0]

        self.points = np.empty((self.size_cloud_env + size_cloud_obj, 3), np.float64)
        self.points[:self.size_cloud_env] = np_cloud_env
        self.points[self.size_cloud_env:] = np_cloud_obj

        self.voro = Voronoi(self.points)

        self.generate_ibs_structure()

    def generate_ibs_structure(self):
        env_idx_vertices = []
        obj_idx_vertices = []

        # check the region formed around point in the environment
        # point_region : Index of the Voronoi region for each input point
        for env_idx_region in self.voro.point_region[:self.size_cloud_env]:
            # voronoi region of environment point
            # regions: Indices of the Voronoi vertices forming each Voronoi region
            env_idx_vertices.extend(self.voro.regions[env_idx_region])

        for obj_idx_region in self.voro.point_region[self.size_cloud_env:]:
            # voronoi region of object point
            obj_idx_vertices.extend(self.voro.regions[obj_idx_region])

        env_idx_vertices = set(env_idx_vertices)
        obj_idx_vertices = set(obj_idx_vertices)

        idx_ibs_vertices = list(env_idx_vertices.intersection(obj_idx_vertices))
        # idx_ibs_vertices.sort(reverse=True)

        # avoid index "-1" for vertices extraction
        # valid_index = list(filter(lambda idx: idx != -1, idx_ibs_vertices))
        # valid_index = [idx for idx in idx_ibs_vertices if idx != -1]
        try:
            idx_ibs_vertices.pop(idx_ibs_vertices.index(-1))
            self.vertices = self.voro.vertices[idx_ibs_vertices]
        except ValueError as e:
            self.vertices = self.voro.vertices[idx_ibs_vertices]

        points_in_voronoi = np.full(len(self.voro.vertices)+1, False, dtype=bool)
        points_in_voronoi[idx_ibs_vertices] = True
        points_in_voronoi[-1]=True # this is for the -1 value (vertex on the infinite)
        points_in_voronoi_mapping_to = np.empty(len(self.voro.vertices)+1, dtype=int)
        for idx in range(len(idx_ibs_vertices)):
            points_in_voronoi_mapping_to[ idx_ibs_vertices[idx] ] = idx
        points_in_voronoi_mapping_to[-1] = -1

        # generate ridge vertices lists
        self.ridge_vertices = []  # Indices of the Voronoi vertices forming each Voronoi ridge
        self.ridge_points = []  # Indices of the points between which each Voronoi ridge lie
        for i in range(len(self.voro.ridge_vertices)):

            ridge = self.voro.ridge_vertices[i]
            #print("ridge "+len(self.voro.ridge_vertices[i])+", ridge_points" + len(self.voro.ridge_points) )
            # only process ridges in which all vertices are defined in ridges defined by Voronoi
            if all(points_in_voronoi[ridge]):
                mapped_idx_ridge = points_in_voronoi_mapping_to[ridge]
                self.ridge_vertices.append(tuple(mapped_idx_ridge))

                ridge_points = self.voro.ridge_points[i]
                self.ridge_points.append(ridge_points)

    def get_trimesh(self):
        tri_faces = []
        for ridge in self.ridge_vertices:
            if -1 in ridge:
                continue
            l_ridge = len(ridge)
            if l_ridge ==3:
                tri_faces.append(ridge)
            else:
                for pos in range(l_ridge - 2):
                    tri_faces.append((ridge[-1], ridge[pos], ridge[pos + 1]))

        mesh = trimesh.Trimesh(vertices=self.vertices, faces=tri_faces)
        mesh.fix_normals()

        return mesh


class IBSMesh(IBS):

    init_size_sampling = -1
    resamplings = -1
    improve_by_collision = -1

    def __init__(self, init_size_sampling=400, resamplings=4, improve_by_collision=True):
        self.init_size_sampling = init_size_sampling
        self.resamplings = resamplings
        self.improve_by_collision = improve_by_collision

    def execute(self, tri_mesh_env, tri_mesh_obj ):
        np_cloud_env_poisson = util.sample_points_poisson_disk(tri_mesh_env, self.init_size_sampling)
        np_cloud_obj_poisson = util.sample_points_poisson_disk(tri_mesh_obj, self.init_size_sampling)

        np_cloud_obj = self.__project_points_in_sampled_mesh(tri_mesh_obj, np_cloud_obj_poisson, np_cloud_env_poisson)
        np_cloud_env = self.__project_points_in_sampled_mesh(tri_mesh_env, np_cloud_env_poisson, np_cloud_obj)

        for i in range(1, self.resamplings):
            np_cloud_obj = self.__project_points_in_sampled_mesh(tri_mesh_obj, np_cloud_obj, np_cloud_env)
            np_cloud_env = self.__project_points_in_sampled_mesh(tri_mesh_env, np_cloud_env, np_cloud_obj)

        if self.improve_by_collision:

            self.__improve_sampling_by_collision_test(tri_mesh_env, tri_mesh_obj, np_cloud_env, np_cloud_obj)

        else:

            super(IBSMesh, self).__init__(np_cloud_env, np_cloud_obj)

    def __improve_sampling_by_collision_test(self, tri_mesh_env, tri_mesh_obj, np_cloud_env, np_cloud_obj):

        collision_tester = trimesh.collision.CollisionManager()
        collision_tester.add_object('env', tri_mesh_env)
        collision_tester.add_object('obj', tri_mesh_obj)
        in_collision = True

        while in_collision:

            super(IBSMesh, self).__init__(np_cloud_env, np_cloud_obj)

            tri_mesh_ibs = self.get_trimesh()

            in_collision, data = collision_tester.in_collision_single(tri_mesh_ibs, return_data=True)

            if not in_collision:
                break

            print("------------------ ")
            print("contact points: ", len(data))

            contact_points_obj = []
            contact_points_env = []

            for i in range(len(data)):
                if "env" in data[i].names:
                    contact_points_env.append(data[i].point)
                if "obj" in data[i].names:
                    contact_points_obj.append(data[i].point)

            if len(contact_points_env) > 0:
                np_contact_points_env = np.unique(np.asarray(contact_points_env), axis=0)
                np_cloud_env = np.concatenate((np_cloud_env, np_contact_points_env))

            if len(contact_points_obj) > 0:
                np_contact_points_obj = np.unique(np.asarray(contact_points_obj), axis=0)
                np_cloud_obj = np.concatenate((np_cloud_obj, np_contact_points_obj))

            if len(contact_points_env) > 0:
                np_cloud_obj = self.__project_points_in_sampled_mesh(tri_mesh_obj, np_cloud_obj, np_contact_points_env)

            if len(contact_points_obj) > 0:
                np_cloud_env = self.__project_points_in_sampled_mesh(tri_mesh_env, np_cloud_env, np_contact_points_obj)

    def __project_points_in_sampled_mesh(self, tri_mesh_sampled, np_sample, np_to_project):
        if np_to_project.shape[0] == 0:
            return np_sample

        (nearest_points, __, __) = tri_mesh_sampled.nearest.on_surface(np_to_project)

        np_new_sample = np.empty((len(np_sample) + len(nearest_points), 3))

        np_new_sample[:len(np_sample)] = np_sample
        np_new_sample[len(np_sample):] = nearest_points
        np_new_sample = np.unique(np_new_sample, axis=0)

        return np_new_sample

    def get_info(self):
        info = {}
        info['init_size_sampling'] = self.init_size_sampling
        info['resamplings'] = self.resamplings
        info['improve_by_collision'] = self.improve_by_collision
        return info