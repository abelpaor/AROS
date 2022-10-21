import logging

import numpy as np
import trimesh
from sklearn.preprocessing import normalize
from vedo import load, Spheres, vtk2trimesh, merge, trimesh2vtk

from it.util import sample_points_poisson_disk_radius


class BubbleFiller():

    def __init__(self, env_file):
        self.vedo_env = load(env_file).bc("t")
        self.tri_mesh_env = vtk2trimesh(self.vedo_env)

    def calculate_fine_bubble_filler(self, fine_spheres_radio):
        """
        Calculate the bubble of small regions, where "small" is defined using both fine and gross sphere radios
        """
        fine_sphere_diameter = fine_spheres_radio * 2

        logging.info("Sampling for FINE spheres")

        # n_seed_points = int(self.tri_mesh_env.area / pow(7 * fine_spheres_radio / 9, 2))
        # seed_points = sample_points_poisson_disk(self.tri_mesh_env, n_seed_points)
        seed_points, seed_normals = sample_points_poisson_disk_radius(self.tri_mesh_env,
                                                                      radius=6*fine_spheres_radio/9,
                                                                      best_choice_sampling=False)
        # avg_distance = calculate_average_distance_nearest_neighbour(seed_points)
        # logging.info(  "average distance nearest neighbour : " + str(avg_distance))
        # seed_normals = get_normal_nearest_point_in_mesh(self.tri_mesh_env, seed_points)
        seed_inverted_normals = -seed_normals


        logging.info("Calculation ray intersections for FINE spheres")
        (__, collided_index_ray, collided_collision_point) = self.tri_mesh_env.ray.intersects_id(
            ray_origins=seed_points, ray_directions=seed_inverted_normals,
            return_locations=True, multiple_hits=False)

        logging.info("Generating FINE spheres")

        # #### collided rays
        collided_vects = collided_collision_point - seed_points[collided_index_ray]
        collided_norms = np.linalg.norm(collided_vects, axis=1)

        minimum_norm_index = collided_index_ray[fine_sphere_diameter < collided_norms]
        collided_minimum_norm_dest = collided_collision_point[collided_norms > fine_sphere_diameter]

        collided_minimum_norm_orig = seed_points[minimum_norm_index]
        collided_minimum_norm_vect = collided_minimum_norm_dest - collided_minimum_norm_orig
        collided_minimum_norm_norms = np.linalg.norm(collided_minimum_norm_vect, axis=1)
        collided_minimum_norm_normalized = normalize(collided_minimum_norm_vect)

        sphere_centres = collided_minimum_norm_orig + collided_minimum_norm_normalized * fine_spheres_radio

        fine_bubbles = Spheres(sphere_centres, r=fine_spheres_radio, c="blue", alpha=1, res=4).lighting("plastic")

        logging.info("Polishing FINE spheres")

        fine_bubbles.cutWithMesh(self.vedo_env)

        return fine_bubbles  # , seed_points

    def calculate_gross_bubble_filler(self, gross_spheres_radio):

        gross_sphere_diameter = gross_spheres_radio*2

        logging.info("Sampling for GROSS spheres")

        # n_seed_points = int(self.tri_mesh_env.area/pow(5*gross_spheres_radio/9, 2))
        # seed_points = sample_points_poisson_disk(self.tri_mesh_env, n_seed_points)
        # avg_distance = calculate_average_distance_nearest_neighbour(seed_points)
        # logging.info(  "average distance nearest neighbour : " + str(avg_distance))
        # seed_normals = get_normal_nearest_point_in_mesh(self.tri_mesh_env, seed_points)
        seed_points, seed_normals = sample_points_poisson_disk_radius(self.tri_mesh_env,
                                                                      radius=3 * gross_spheres_radio / 9,
                                                                      best_choice_sampling=False,
                                                                      use_geodesic_distance=False)
        seed_inverted_normals = -seed_normals

        logging.info("Calculation ray intersections for GROSS spheres")

        (__, collided_index_ray, collided_collision_point) = self.tri_mesh_env.ray.intersects_id(
            ray_origins=seed_points, ray_directions=seed_inverted_normals,
            return_locations=True, multiple_hits=False)

        collided_vects = collided_collision_point - seed_points[collided_index_ray]
        collided_norms = np.linalg.norm(collided_vects, axis=1)
        minimum_norm_index = collided_index_ray[gross_sphere_diameter < collided_norms]

        no_collided_index_ray = [i for i in range(seed_points.shape[0]) if i not in collided_index_ray]

        indexes = list(minimum_norm_index) + no_collided_index_ray

        vects_normalized = normalize(seed_inverted_normals[indexes])
        vects_orig = seed_points[indexes]

        sphere_centres = vects_orig + vects_normalized * gross_spheres_radio

        gross_bubbles = Spheres(sphere_centres, r=gross_spheres_radio, c="orange", alpha=1, res=4).lighting("plastic")

        logging.info("Polishing BIG spheres")

        gross_bubbles.cutWithMesh(self.vedo_env)

        return gross_bubbles  # , seed_points, Lines(vects_orig, sphere_centres, c=gross_bubbles.color())


    def calculate_floor_holes_filler(self, hole_filler_sphere_radio, extension=.40):
        """

        :param hole_filler_sphere_radio:
        :param extension: a gap to fill after the bounding box size
        :return:
        """
        logging.info("Calculating bounding box")
        orig_bb = self.tri_mesh_env.bounding_box
        transform = np.eye(4)
        transform[:3, 3] = orig_bb.centroid
        extents = orig_bb.extents + [extension*2, extension*2,0]

        bb = trimesh.primitives.Box(transform=transform,
                              extents=extents,
                              mutable=False)

        old_idx = np.asarray(bb.vertices)[:, 2].argsort()[0:4]
        vertices = np.asarray(bb.vertices)[old_idx]
        old_faces = [face for face in np.array(bb.faces) if
                     face[0] in old_idx and face[1] in old_idx and face[2] in old_idx]
        l_old_faces = [list(face) for face in old_faces]
        l_old_idx = list(old_idx)
        new_faces = [[l_old_idx.index(l_old_faces[0][0]), l_old_idx.index(l_old_faces[0][1]),
                      l_old_idx.index(l_old_faces[0][2])],
                     [l_old_idx.index(l_old_faces[1][0]), l_old_idx.index(l_old_faces[1][1]),
                      l_old_idx.index(l_old_faces[1][2])]]

        tri_mesh_hole_filler = trimesh.Trimesh(vertices=vertices, faces=new_faces)

        logging.info("Sampling on plane hole filler")

        # n_seed_points = int(tri_mesh_hole_filler.area / pow(hole_filler_sphere_radio, 2))
        # seed_points = sample_points_poisson_disk(tri_mesh_hole_filler, n_seed_points)
        # avg_distance = calculate_average_distance_nearest_neighbour(seed_points)
        # logging.info(  "average distance nearest neighbour : " + str(avg_distance))
        # seed_normals = get_normal_nearest_point_in_mesh(tri_mesh_hole_filler, seed_points)
        seed_points, seed_normals = sample_points_poisson_disk_radius(tri_mesh_hole_filler,
                                                                      radius=2*hole_filler_sphere_radio/3)

        logging.info("Calculation floor hole spheres")

        vects_normalized = normalize(seed_normals)
        vects_orig = seed_points

        sphere_centres = vects_orig + vects_normalized * hole_filler_sphere_radio

        bubbles = Spheres(sphere_centres, r=hole_filler_sphere_radio, alpha=1, res=4).lighting("plastic")

        logging.info("Polishing floor hole spheres")

        hole_filler = merge(trimesh2vtk(tri_mesh_hole_filler), bubbles).color("green")

        return hole_filler
