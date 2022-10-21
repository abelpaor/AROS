import numpy as np

import trimesh

from aros import util


class MaxDistancesCalculator:
    # influence_radio
    # sum_max_distances

    def __init__(self, pv_points, pv_vectors, tri_mesh_obj, consider_collision_with_object, radio_ratio = 2):
        '''
        Measures the max reachable distance of provenances vectors in a interaction
        :param pv_points: IBS key points, origin of provenance vectors
        :param pv_vectors: Provenance vectors
        :param tri_mesh_obj: objects triangular mesh
        :param consider_collision_with_object: consider collision with object to determine ray intersections
        '''
        self.consider_collision_with_object = consider_collision_with_object
        self.radio_ratio = radio_ratio
        self.influence_radio, self.influence_center = util.influence_sphere(tri_mesh_obj, radio_ratio)

        self.sphere_of_influence = trimesh.primitives.Sphere(radius=self.influence_radio,
                                                             center=self.influence_center, subdivisions=5)

        expected_intersections = pv_points + pv_vectors

        scene_to_collide_rays = self.sphere_of_influence

        # measure collision with object itself
        if self.consider_collision_with_object:
            scene_to_collide_rays += tri_mesh_obj

        # looking for the nearest ray intersections
        (__,
         idx_ray,
         calculated_intersections) = scene_to_collide_rays.ray.intersects_id(
            ray_origins=pv_points,
            ray_directions=pv_vectors,
            return_locations=True,
            multiple_hits=False)

        self.max_distances = np.linalg.norm(calculated_intersections - expected_intersections, axis=1)
        self.sum_max_distances = np.sum(self.max_distances)

    def get_info(self):
        info = {}
        info['radio_ratio'] = self.radio_ratio
        info['obj_influence_radio'] = self.influence_radio
        info['sum_max_distances'] = self.sum_max_distances
        info['consider_collision_with_object'] = self.consider_collision_with_object
        return info
