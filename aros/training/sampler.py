import math

from collections import Counter
from abc import ABC, abstractmethod
import numpy as np

import trimesh

import aros.util as util


# TODO establish default rates

class Sampler(ABC):
    SAMPLE_SIZE = 512

    # input for sampling execution
    tri_mesh_ibs = None
    tri_mesh_env = None

    # variables for execution
    np_cloud_ibs = np.array([])
    idx_ibs_cloud_sample = []

    # outputs
    pv_points = np.array([])
    pv_vectors = np.array([])
    pv_norms = np.array([])

    def __init__(self):
        super().__init__()

    def execute(self, tri_mesh_ibs, tri_mesh_env):
        self.tri_mesh_ibs = tri_mesh_ibs
        self.tri_mesh_env = tri_mesh_env

        self.get_clouds_to_sample()
        self.get_sample()


    def map_norm(self, norm, max, min):
        return (norm - min) * (0 - 1) / (max - min) + 1
        # return(value_in-min_original_range)*(max_mapped-min_mapped)/(max_original_range-min_original_range)+min_mapped

    def get_sample(self):
        self.idx_ibs_cloud_sample = np.random.randint(0, self.np_cloud_ibs.shape[0], self.SAMPLE_SIZE)
        self.pv_points = self.np_cloud_ibs[self.idx_ibs_cloud_sample]
        (closest_points_in_env, norms, __) = self.tri_mesh_env.nearest.on_surface(self.pv_points)
        self.pv_vectors = closest_points_in_env - self.pv_points
        self.pv_norms = norms

    @abstractmethod
    def get_clouds_to_sample(self):
        pass

    def get_info(self):
        info = {}
        info['sampler_name'] = self.__class__.__name__
        info['sample_size'] = self.SAMPLE_SIZE
        info['ibs_points'] = self.np_cloud_ibs.shape[0]
        return info


class WeightedSampler(Sampler, ABC):
    BATCH_SIZE_FOR_CLSST_POINT = 1000

    rolls = np.array([])

    np_cloud_env = np.array([])
    norms = np.array([])

    def __init__(self, rate_generated_random_numbers=500):
        self.rate_generated_random_numbers = rate_generated_random_numbers
        super().__init__()

    def get_sample(self):
        max_norm = self.norms.max()
        min_norm = self.norms.min()
        mapped_norms = [self.map_norm(norm, max_norm, min_norm) for norm in self.norms]
        sum_mapped_norms = sum(mapped_norms)
        probabilities = [float(mapped) / sum_mapped_norms for mapped in mapped_norms]

        n_rolls = self.np_cloud_ibs.shape[0] * self.rate_generated_random_numbers
        self.rolls = np.random.choice(self.np_cloud_ibs.shape[0], n_rolls, p=probabilities)
        votes = Counter(self.rolls).most_common(self.SAMPLE_SIZE)

        self.idx_ibs_cloud_sample = [tuple[0] for tuple in votes]
        self.pv_points = self.np_cloud_ibs[self.idx_ibs_cloud_sample]
        self.pv_vectors = self.np_cloud_env[self.idx_ibs_cloud_sample] - self.pv_points
        self.pv_norms = self.norms[self.idx_ibs_cloud_sample]

    def choosing_with_other_rate(self, rate_generated_random_numbers):
        self.rate_generated_random_numbers = rate_generated_random_numbers
        self.get_sample()

    def get_info(self):
        info = super().get_info()
        info['generated_random_numbers']=self.rolls.size
        info['rate_generated_random_numbers'] = self.rate_generated_random_numbers
        return info

class PoissonDiscRandomSampler(Sampler):
    rate_ibs_samples = 25

    def __init__(self, rate_ibs_samples=25):
        self.rate_ibs_samples = rate_ibs_samples
        super().__init__()

    def get_clouds_to_sample(self):
        self.np_cloud_ibs = util.sample_points_poisson_disk(self.tri_mesh_ibs, self.SAMPLE_SIZE * self.rate_ibs_samples)


    def get_info(self):
        info = super().get_info()
        info['rate_ibs_samples'] = self.rate_ibs_samples
        return info



class PoissonDiscWeightedSampler(WeightedSampler):

    def __init__(self, rate_ibs_samples=25, rate_generated_random_numbers=500):
        self.rate_ibs_samples = rate_ibs_samples
        super().__init__(rate_generated_random_numbers)

    def get_clouds_to_sample(self):
        n_ibs_samples = self.SAMPLE_SIZE * self.rate_ibs_samples
        self.np_cloud_ibs = util.sample_points_poisson_disk(self.tri_mesh_ibs, n_ibs_samples)

        iterations = math.ceil(n_ibs_samples / self.BATCH_SIZE_FOR_CLSST_POINT)

        l_closest = []
        l_norms = []
        for it in range(iterations):
            idx_from = it * self.BATCH_SIZE_FOR_CLSST_POINT
            idx_to = idx_from + self.BATCH_SIZE_FOR_CLSST_POINT
            (closest_points, norms, __) = self.tri_mesh_env.nearest.on_surface(self.np_cloud_ibs[idx_from: idx_to])
            l_closest.extend(closest_points)
            l_norms.extend(norms)

        self.np_cloud_env = np.asarray(l_closest)
        self.norms = np.asarray(l_norms)

        bad_indexes = np.argwhere(np.isnan(l_norms))

        self.np_cloud_env = np.delete(self.np_cloud_env, bad_indexes, 0)
        self.np_cloud_ibs = np.delete(self.np_cloud_ibs, bad_indexes, 0)
        self.norms = np.delete(self.norms, bad_indexes, 0)

    def get_info(self):
        info = super().get_info()
        info['rate_ibs_samples'] = self.rate_ibs_samples
        return info


class OnVerticesRandomSampler(Sampler):

    def get_clouds_to_sample(self):
        self.np_cloud_ibs = np.asarray(self.tri_mesh_ibs.vertices)


class OnVerticesWeightedSampler(WeightedSampler):

    def get_clouds_to_sample(self):
        self.np_cloud_ibs = np.asarray(self.tri_mesh_ibs.vertices)
        size_input_cloud = self.np_cloud_ibs.shape[0]
        iterations = math.ceil(size_input_cloud / self.BATCH_SIZE_FOR_CLSST_POINT)

        l_closest = []
        l_norms = []

        for it in range(iterations):
            idx_from = it * self.BATCH_SIZE_FOR_CLSST_POINT
            idx_to = idx_from + self.BATCH_SIZE_FOR_CLSST_POINT
            (closest_points, norms, __) = self.tri_mesh_env.nearest.on_surface(self.np_cloud_ibs[idx_from: idx_to])
            l_closest += list(closest_points)
            l_norms += list(norms)

        self.np_cloud_env = np.asarray(l_closest)
        self.norms = np.asarray(l_norms)

        bad_idxs = np.argwhere(np.isnan(l_norms))

        self.np_cloud_env = np.delete(self.np_cloud_env, bad_idxs, 0)
        self.np_cloud_ibs = np.delete(self.np_cloud_ibs, bad_idxs, 0)
        self.norms = np.delete(self.norms, bad_idxs, 0)


class OnGivenPointCloudRandomSampler(Sampler):
    np_input_cloud = np.array([])

    BATCH_SIZE_FOR_CLSST_POINT = 1000

    def __init__(self, np_input_cloud):
        self.np_input_cloud = np_input_cloud
        super().__init__()

    def get_clouds_to_sample(self):

        # With environment points, find the nearest point in the IBS surfaces
        size_input_cloud = self.np_input_cloud.shape[0]
        iterations = math.ceil(size_input_cloud / self.BATCH_SIZE_FOR_CLSST_POINT)
        l_closest = []
        l_norms = []
        for it in range(iterations):
            idx_from = it * self.BATCH_SIZE_FOR_CLSST_POINT
            idx_to = idx_from + self.BATCH_SIZE_FOR_CLSST_POINT
            (closest_points, norms, __) = self.tri_mesh_ibs.nearest.on_surface(self.np_input_cloud[idx_from: idx_to])
            l_closest += list(closest_points)
            l_norms += list(norms)

        self.np_cloud_ibs = np.asarray(l_closest)
        bad_indexes = np.argwhere(np.isnan(l_norms))
        self.np_cloud_ibs = np.delete(self.np_cloud_ibs, bad_indexes, 0)

        # Calculate all PROVENANCE VECTOR RELATED TO IBS SURFACE SAMPLES
        size_cloud_ibs = self.np_cloud_ibs.shape[0]
        iterations = math.ceil(size_cloud_ibs / self.BATCH_SIZE_FOR_CLSST_POINT)
        l_closest = []
        l_norms = []
        for it in range(iterations):
            idx_from = it * self.BATCH_SIZE_FOR_CLSST_POINT
            idx_to = idx_from + self.BATCH_SIZE_FOR_CLSST_POINT
            (closest_points, norms, __) = self.tri_mesh_env.nearest.on_surface(self.np_cloud_ibs[idx_from: idx_to])
            l_closest += list(closest_points)
            l_norms += list(norms)

        self.np_cloud_env = np.asarray(l_closest)
        self.norms = np.asarray(l_norms)
        bad_indexes = np.argwhere(np.isnan(l_norms))
        self.np_cloud_ibs = np.delete(self.np_cloud_ibs, bad_indexes, 0)
        self.np_cloud_env = np.delete(self.np_cloud_env, bad_indexes, 0)
        self.norms = np.delete(self.norms, bad_indexes, 0)



class OnGivenPointCloudWeightedSampler(WeightedSampler):

    def __init__(self, np_input_cloud, rate_generated_random_numbers=500):
        self.np_input_cloud = np_input_cloud
        super().__init__(rate_generated_random_numbers)

    def get_clouds_to_sample(self):
        size_input_cloud = self.np_input_cloud.shape[0]
        iterations = math.ceil(size_input_cloud / self.BATCH_SIZE_FOR_CLSST_POINT)

        l_closest = []
        l_norms = []

        for it in range(iterations):
            idx_from = it * self.BATCH_SIZE_FOR_CLSST_POINT
            idx_to = idx_from + self.BATCH_SIZE_FOR_CLSST_POINT
            (closest_points, norms, __) = self.tri_mesh_ibs.nearest.on_surface(self.np_input_cloud[idx_from: idx_to])
            l_closest += list(closest_points)
            l_norms += list(norms)

        self.np_cloud_ibs = np.asarray(l_closest)
        bad_indexes = np.argwhere(np.isnan(l_norms))
        self.np_cloud_ibs = np.delete(self.np_cloud_ibs, bad_indexes, 0)

        # Calculate all PROVENANCE VECTOR RELATED TO IBS SURFACE SAMPLES
        size_cloud_ibs = self.np_cloud_ibs.shape[0]
        iterations = math.ceil(size_cloud_ibs / self.BATCH_SIZE_FOR_CLSST_POINT)
        l_closest = []
        l_norms = []
        for it in range(iterations):
            idx_from = it * self.BATCH_SIZE_FOR_CLSST_POINT
            idx_to = idx_from + self.BATCH_SIZE_FOR_CLSST_POINT
            (closest_points, norms, __) = self.tri_mesh_env.nearest.on_surface(self.np_cloud_ibs[idx_from: idx_to])
            l_closest += list(closest_points)
            l_norms += list(norms)

        self.np_cloud_env = np.asarray(l_closest)
        self.norms = np.asarray(l_norms)
        bad_indexes = np.argwhere(np.isnan(l_norms))
        self.np_cloud_ibs = np.delete(self.np_cloud_ibs, bad_indexes, 0)
        self.np_cloud_env = np.delete(self.np_cloud_env, bad_indexes, 0)
        self.norms = np.delete(self.norms, bad_indexes, 0)



class SamplerClearance(ABC):
    sample_size = 256

    # input for sampling execution
    tri_mesh_ibs = None
    tri_mesh_obj = None
    # work variables
    np_src_cloud = np.array([])
    # outputs
    cv_points = np.array([])
    cv_vectors = np.array([])
    cv_norms = np.array([])

    def __init__(self, sample_size=None):
        super().__init__()
        if sample_size is not None:
            self.sample_size = sample_size

    def execute(self, tri_mesh_ibs, tri_mesh_obj):
        """
        Generate the clearance vectors
        Parameters
        ----------
        tri_mesh_ibs: mesh structure of Interaction Bisector Surface (IBS)
        tri_mesh_obj: mesh structure of object

        Returns
        -------
        generate instance variables cv_points, cv_vectors, cv_norms associated with the clearance vectors
        """
        self.tri_mesh_ibs = tri_mesh_ibs
        self.tri_mesh_obj = tri_mesh_obj
        self.get_source_cloud()
        self.calculate_clearance_vectors()

    @abstractmethod
    def calculate_clearance_vectors(self):
        pass

    @abstractmethod
    def get_source_cloud(self):
        pass

    def get_info(self):
        info = {}
        info['sampler_clearance_name'] = self.__class__.__name__
        info['sample_clearance_size'] = self.sample_size
        return info


class OnIBSPoissonDiscSamplerClearance(SamplerClearance):
    """
    Generates provenance vectors by sampling on the IBS and extend them to the nearest point in the object
    """

    def __init__(self):
        super().__init__()

    def get_source_cloud(self):
        self.np_src_cloud = util.sample_points_poisson_disk(self.tri_mesh_ibs, self.sample_size)

    def calculate_clearance_vectors(self):
        self.cv_points = self.np_src_cloud
        (closest_points_in_obj, norms, __) = self.tri_mesh_obj.nearest.on_surface(self.cv_points)
        self.cv_vectors = closest_points_in_obj - self.cv_points
        self.cv_norms = norms


class OnObjectPoissonDiscSamplerClearance(SamplerClearance):
    """
    Generates clearance vectors by
    1) poisson disc sampling on object
    2) finding the nearest point from OBJECT SAMPLES to ( IBS U Sphere_of_influence)
    """
    influence_radio_ratio = 1.2

    def get_source_cloud(self):
        self.np_src_cloud = util.sample_points_poisson_disk(self.tri_mesh_obj, self.sample_size)

    def calculate_clearance_vectors(self):
        sphere_ro, sphere_center = util.influence_sphere(self.tri_mesh_obj, self.influence_radio_ratio)
        sphere = trimesh.primitives.Sphere(radius=sphere_ro, center=sphere_center)

        wrapper = self.tri_mesh_ibs + sphere

        (self.cv_points, self.cv_norms, __) = wrapper.nearest.on_surface(self.np_src_cloud)

        self.cv_vectors = self.np_src_cloud - self.cv_points

    def get_info(self):
        info = super().get_info()
        info['influence_radio_ratio'] = self.influence_radio_ratio
        return info

class PropagateFromSpherePoissonDiscSamplerClearance(OnObjectPoissonDiscSamplerClearance):
    """
    Generates clearance vectors by
    1) sampling on a sphere of influence,
    2) generate rays from samples to the sphere centre
    3) find intersection of rays in object obtaining OBJECT SAMPLES
        IF NO INTERSECTION: find the nearest point from "circle sample" to object
    4) finding nearest point from OBJECT SAMPLES to IBS
    """


    def get_source_cloud(self):
        sphere_ro, sphere_center = util.influence_sphere(self.tri_mesh_obj, self.influence_radio_ratio)
        sphere = trimesh.primitives.Sphere(radius=sphere_ro, center=sphere_center)
        sphere_samples = util.sample_points_poisson_disk(sphere, self.sample_size)

        # find intersection with rays from the sphere of influence
        rays_to_center = sphere_center - sphere_samples
        (__, index_ray, ray_collission_on_object) = self.tri_mesh_obj.ray.intersects_id(
            ray_origins=sphere_samples, ray_directions=rays_to_center,
            return_locations=True, multiple_hits=False)
        self.np_src_cloud = np.empty((self.sample_size, 3))
        self.np_src_cloud[index_ray] = ray_collission_on_object

        # no intersection, then uses the nearest point in object
        no_index_ray = [i for i in range(self.sample_size) if i not in index_ray]
        (closest_in_obj, __, __) = self.tri_mesh_obj.nearest.on_surface(sphere_samples[no_index_ray])
        self.np_src_cloud[no_index_ray] = closest_in_obj

        return


class PropagateObjectNormalFromSpherePoissonDiscSamplerClearance(SamplerClearance):
    """
    Generates clearance vectors by
    1) sampling on a sphere of influence,
    2) generate rays from samples to the sphere centre
    3) find intersection of rays in object obtaining OBJECT SAMPLES
        IF NO INTERSECTION: find the nearest point from every "circle sample" in object
    4) calculate normal for every sample in object
    5) follow direction of normal until reaching IBS or the sphere of influence
    6) starting point of clearance vector in the sampling normal direction no further than threshold or IBS
    """
    influence_radio_ratio = 1.2
    distance_threshold = 0.05
    # work variables
    np_src_cloud_normal_vector = np.array([])

    def __init__(self, sample_size=None, distance_threshold=None):
        super().__init__()
        if sample_size is not None:
            self.sample_size = sample_size
        if distance_threshold is not None:
            self.distance_threshold = distance_threshold

    def get_source_cloud(self):
        sphere_ro, sphere_center = util.influence_sphere(self.tri_mesh_obj, self.influence_radio_ratio)
        sphere = trimesh.primitives.Sphere(radius=sphere_ro, center=sphere_center)
        sphere_samples = util.sample_points_poisson_disk(sphere, self.sample_size)

        # find intersection with rays from the sphere of influence
        rays_to_center = sphere_center - sphere_samples
        (index_triangle, index_ray, collision_point_on_object) = self.tri_mesh_obj.ray.intersects_id(
            ray_origins=sphere_samples, ray_directions=rays_to_center,
            return_locations=True, multiple_hits=False)
        # initialize work variables
        self.np_src_cloud = np.empty((self.sample_size, 3))
        self.np_src_cloud_normal_vector = np.empty((self.sample_size, 3))

        self.np_src_cloud[index_ray] = collision_point_on_object
        self.np_src_cloud_normal_vector[index_ray] = self.tri_mesh_obj.face_normals[index_triangle]

        # no intersection, then uses the nearest point in object
        no_index_ray = [i for i in range(self.sample_size) if i not in index_ray]
        (closest_in_obj, __, index_triangle) = self.tri_mesh_obj.nearest.on_surface(sphere_samples[no_index_ray])
        self.np_src_cloud[no_index_ray] = closest_in_obj
        self.np_src_cloud_normal_vector[no_index_ray] = self.tri_mesh_obj.face_normals[index_triangle]

        return

    def calculate_clearance_vectors(self):
        sphere_ro, sphere_center = util.influence_sphere(self.tri_mesh_obj, self.influence_radio_ratio)
        sphere = trimesh.primitives.Sphere(radius=sphere_ro, center=sphere_center)

        wrapper = self.tri_mesh_ibs + sphere

        (index_triangle, index_ray, locations) = wrapper.ray.intersects_id(
            ray_origins=self.np_src_cloud, ray_directions=self.np_src_cloud_normal_vector,
            return_locations=True, multiple_hits=False)

        raw_cv = self.np_src_cloud - locations
        norm_raw_cv = np.linalg.norm(raw_cv, axis=1)

        self.cv_points = np.zeros(locations.shape)
        self.cv_norms = np.zeros(norm_raw_cv.shape)

        idx_less_equal_than = norm_raw_cv <= self.distance_threshold
        idx_more_than = ~idx_less_equal_than

        # first assign all points in a SMALLER distance than the threshold
        self.cv_points[idx_less_equal_than] = locations[idx_less_equal_than]
        self.cv_norms[idx_less_equal_than] = norm_raw_cv[idx_less_equal_than]
        # adjust points to BIGGER distance than threshold
        self.cv_points[idx_more_than] = self.np_src_cloud[idx_more_than] + (
                    self.np_src_cloud_normal_vector[idx_more_than] * self.distance_threshold)
        self.cv_norms[idx_more_than] = self.distance_threshold

        self.cv_vectors = self.np_src_cloud - self.cv_points

    def get_info(self):
        info = super().get_info()
        info['influence_radio_ratio'] = self.influence_radio_ratio
        info['distance_threshold'] = self.distance_threshold
        return info

class PropagateNormalObjectPoissonDiscSamplerClearance(PropagateObjectNormalFromSpherePoissonDiscSamplerClearance):
    """
    Generates clearance vectors by
    1) poisson sampling on object,
    2) calculate normal for every sample in object
    3) follow direction of normal until reaching IBS or the sphere of influence
    4) starting point of clearance vector in the sampling normal direction no further than threshold or IBS
    5) vector goes from IBS or a point farther than distance_threshold to the sampling point in object
    """


    def get_source_cloud(self):
        self.np_src_cloud = util.sample_points_poisson_disk(self.tri_mesh_obj, self.sample_size)

        (closest_in_obj, __, index_triangle) = self.tri_mesh_obj.nearest.on_surface(self.np_src_cloud)

        self.np_src_cloud_normal_vector = self.tri_mesh_obj.face_normals[index_triangle]

        return
