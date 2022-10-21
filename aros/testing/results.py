import math
import json
import copy
import numpy as np

from transforms3d.affines import compose

from aros import util
from aros.testing.deglomerator import Deglomerator


class Results:
    distances = None
    distances_summary = None
    missed = None
    raw_distances = None

    def __init__(self, distances, resumed_distances, missed, raw_distances):
        self.distances = distances
        self.distances_summary = resumed_distances
        self.missed = missed
        self.raw_distances = raw_distances



class Analyzer:
    results = None

    def __init__(self, idx_ray, calculated_intersections, num_it_to_test, influence_radius, num_orientations,
                 expected_intersections):
        self.idx_ray = idx_ray
        self.calculated_intersections = calculated_intersections
        self.num_it_to_test = num_it_to_test
        self.influence_radius = influence_radius
        self.num_orientations = num_orientations
        self.expected_intersections = expected_intersections

    '''def raw_measured_scores(self):

        distances = self._distances_between_calculated_and_expected_intersections()

        distances_summary, missed = self._resume( distances)

        return distances, distances_summary, missed '''

    def measure_scores(self):
        if self.results is None:
            raw_distances = self._distances_between_calculated_and_expected_intersections()
            # avoid consider distances farther than the influence radius
            distances = self._avoid_distances_farther_influence_radius(raw_distances)
            # summarizing distances and count pv missed by distances or because of lack of intersection
            distances_summary, missed = self._resume( distances)
            self.results = Results(distances, distances_summary, missed, raw_distances)

        return self.results.distances, self.results.distances_summary, self.results.missed

    def best_angle_by_distance_by_affordance(self):
        if self.results is None:
            self.measure_scores()

        best_scores = np.empty((self.num_it_to_test, 4), np.float64)

        index = 0
        for by_affordance_dist in self.results.distances_summary:
            (score, orientation) = min((sco, ori) for ori, sco in enumerate(by_affordance_dist))
            angle = (2 * math.pi / self.num_orientations) * orientation
            missing =  self.results.missed[index][orientation]
            best_scores[index] = [orientation, angle, score, missing]
            index += 1

        return best_scores

    def calculated_pvs_intersection(self, num_interaction, orientation):
        num_pv_by_interaction = (self.expected_intersections.shape[0] / self.num_it_to_test)
        num_pv_by_orientation = num_pv_by_interaction / self.num_orientations

        idx_from = num_interaction * num_pv_by_interaction + orientation * num_pv_by_orientation
        idx_to = idx_from + num_pv_by_orientation

        temp_pv_intersections = np.empty(self.expected_intersections.shape)
        temp_pv_intersections[:] = math.nan
        temp_pv_intersections[self.idx_ray] = self.calculated_intersections
        idx_intersected = [ray for ray in self.idx_ray if ray >= idx_from and ray < idx_to]
        pv_intersections = temp_pv_intersections[idx_intersected]
        return pv_intersections

    def _count_nan(self, a):
        return len(a[np.isnan(a)])

    def _resume(self, all_distances):
        resumed_distances = np.array(list(map(np.nansum,
                                              np.split(
                                                  all_distances,
                                                  self.num_it_to_test * self.num_orientations
                                              )))).reshape(self.num_it_to_test, self.num_orientations)

        resumed_missed = np.array(list(map(self._count_nan,
                                   np.split(
                                       all_distances,
                                       self.num_it_to_test * self.num_orientations
                                   )))).reshape(self.num_it_to_test, self.num_orientations)

        return resumed_distances, resumed_missed

    def _distances_between_calculated_and_expected_intersections(self):
        # calculated offline during training iT
        trained_intersections = self.expected_intersections[self.idx_ray]

        intersections_distances = np.linalg.norm(self.calculated_intersections - trained_intersections, axis=1)

        all_distances = np.empty(len(self.expected_intersections))
        all_distances[:] = math.nan

        all_distances[self.idx_ray] = intersections_distances

        return all_distances

    def _avoid_distances_farther_influence_radius(self, all_distances):
        '''
        Filters distances bigger than the maximum distance turning them in missing intersection (mat.nan)
        :param all_distances: All measured distance of intersections in the direction of each provenance vector
        :return: distances no bigger than the maximum distance measured with the influence sphere
        '''
        filter_distances = np.copy(all_distances)
        # avoid consider distances farther than the influence radius
        pv_by_interaction = int(self.expected_intersections.shape[0] / self.num_it_to_test)
        for interaction in range(self.num_it_to_test):
            idx_from = pv_by_interaction * interaction
            idx_to = idx_from + pv_by_interaction
            to_check = filter_distances[idx_from:idx_to]
            greater_than = util.compare_nan_array(np.greater, to_check, self.influence_radius[interaction])
            #to_check[to_check > self.influence_radius[interaction]] = math.nan
            to_check[greater_than] = math.nan
            filter_distances[idx_from:idx_to] = to_check

        return filter_distances


class ResultsClearance:
    collision_vectors_norms = None
    is_smaller_norm_by_inter_and_ori = None
    resumed_smaller_norm_by_inter_and_ori = None
    percentage_smaller_norm_by_inter_ori = None

    def __init__(self, collision_vectors_norms, is_smaller_norm_by_inter_and_ori,
                 resumed_smaller_norm_by_inter_and_ori, percentage_smaller_norm_by_inter_ori):

        self.collision_vectors_norms = collision_vectors_norms
        self.is_smaller_norm_by_inter_and_ori = is_smaller_norm_by_inter_and_ori
        self.resumed_smaller_norm_by_inter_and_ori = resumed_smaller_norm_by_inter_and_ori
        self.percentage_smaller_norm_by_inter_ori = percentage_smaller_norm_by_inter_ori


class AnalyzerClearance:
    results = None

    def __init__(self, idx_ray, idx_ray_intersections, num_it_to_test, num_cv, num_orientations,
                 compiled_cv_begin, compiled_cv_norms):
        self.idx_ray = idx_ray
        self.idx_ray_intersections = idx_ray_intersections
        self.num_it_to_test = num_it_to_test
        self.num_cv_per_it_and_ori = num_cv
        self.num_orientations = num_orientations
        self.compiled_cv_begin = compiled_cv_begin
        self.compiled_cv_norms = compiled_cv_norms
        self.compare_norms_collided_and_clearance_vectors()

    def compare_norms_collided_and_clearance_vectors(self):
        if self.results is None:
            collision_vectors_norms = self._calculate_collision_vectors_norms()
            is_s_n_int_ori = self._exist_smaller_norms_compared_with_trained_by_iter_and_ori(collision_vectors_norms)
            res_sm_n_int_ori = self._count_smaller_norms_compared_with_trained(is_s_n_int_ori)
            per_sm_n_int_ori = self._calculate_percentage_smaller_norms_by_iter_and_ori(res_sm_n_int_ori)
            self.results = ResultsClearance(collision_vectors_norms, is_s_n_int_ori, res_sm_n_int_ori, per_sm_n_int_ori)
        return self.results

    def _calculate_percentage_smaller_norms_by_iter_and_ori(self, resumed_by_inter_and_ori):
        percentage_by_inter_ori = np.zeros(resumed_by_inter_and_ori.shape)
        for num_it in range(self.num_it_to_test):
            percentage_by_inter_ori[num_it, :] = resumed_by_inter_and_ori[num_it,:] / self.num_cv_per_it_and_ori
        return percentage_by_inter_ori

    def _exist_smaller_norms_compared_with_trained_by_iter_and_ori(self, collision_vectors_norms):
        comparison = collision_vectors_norms < self.compiled_cv_norms
        results_by_iter_and_ori = comparison.reshape(self.num_it_to_test,
                                                      self.num_orientations,
                                                      self.num_cv_per_it_and_ori)
        return results_by_iter_and_ori

    def _count_smaller_norms_compared_with_trained(self, results_by_iter_and_ori):
        by_cv_set = results_by_iter_and_ori.reshape(self.num_it_to_test*self.num_orientations, self.num_cv_per_it_and_ori)
        resumed_by_inter_and_ori = np.array(list(map(np.sum, by_cv_set)))
        resumed_by_inter_and_ori = resumed_by_inter_and_ori.reshape(self.num_it_to_test, self.num_orientations)

        return resumed_by_inter_and_ori

    def _calculate_collision_vectors_norms(self):
        collision_vecs_norms = np.linalg.norm(self.idx_ray_intersections - self.compiled_cv_begin[self.idx_ray], axis=1)
        all_norms = np.empty(self.compiled_cv_begin.shape[0])
        all_norms[:] = math.inf
        all_norms[self.idx_ray] = collision_vecs_norms

        return all_norms
