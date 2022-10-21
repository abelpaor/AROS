import math
import json
import os
import numpy as np
import open3d as o3d
from transforms3d.derivations.eulerangles import z_rotation

from aros.training.trainer import TrainerClearance


class Agglomerator:
    ORIENTATIONS = None
    it_trainer = None
    def __init__(self, it_trainer, num_orientations=8):
        self.ORIENTATIONS = num_orientations
        self.it_trainer = it_trainer

        orientations = [x * (2 * math.pi / self.ORIENTATIONS) for x in range(0, self.ORIENTATIONS)]

        agglomerated_pv_points = []
        agglomerated_pv_vectors = []
        agglomerated_pv_vdata = []
        agglomerated_normals = []

        self.trainer = it_trainer
        self.sample_size = it_trainer.pv_points.shape[0]

        pv_vdata = np.zeros((self.sample_size, 3), np.float64)
        pv_vdata[:, 0:2] = np.hstack((it_trainer.pv_norms.reshape(-1, 1), it_trainer.pv_mapped_norms.reshape(-1, 1)))

        for angle in orientations:
            rotation = z_rotation(angle)
            agglomerated_pv_points.append(np.dot(it_trainer.pv_points, rotation.T))
            agglomerated_pv_vectors.append(np.dot(it_trainer.pv_vectors, rotation.T))
            agglomerated_pv_vdata.append(pv_vdata)
            agglomerated_normals.append(np.dot(it_trainer.normal_env, rotation.T))

        self.agglomerated_pv_points = np.asarray(agglomerated_pv_points).reshape(-1, 3)
        self.agglomerated_pv_vectors = np.asarray(agglomerated_pv_vectors).reshape(-1, 3)
        self.agglomerated_pv_vdata = np.asarray(agglomerated_pv_vdata).reshape(-1, 3)
        self.agglomerated_normals = np.asarray(agglomerated_normals).reshape(-1, 3)


class AgglomeratorClearance(Agglomerator):

    def __init__(self, it_trainer, num_orientations=8):
        assert isinstance(it_trainer, TrainerClearance)

        super().__init__(it_trainer, num_orientations)

        orientations = [x * (2 * math.pi / self.ORIENTATIONS) for x in range(0, self.ORIENTATIONS)]

        agglomerated_cv_points = []
        agglomerated_cv_vectors = []
        agglomerated_cv_vdata = []

        self.sample_clearance_size = it_trainer.cv_points.shape[0]
        cv_vdata = np.zeros((self.sample_clearance_size, 3), np.float64)
        cv_vdata[:, 0:1] = it_trainer.cv_norms.reshape(-1, 1)

        for angle in orientations:
            rotation = z_rotation(angle)
            agglomerated_cv_points.append(np.dot(it_trainer.cv_points, rotation.T))
            agglomerated_cv_vectors.append(np.dot(it_trainer.cv_vectors, rotation.T))
            agglomerated_cv_vdata.append(cv_vdata)

        self.agglomerated_cv_points = np.asarray(agglomerated_cv_points).reshape(-1, 3)
        self.agglomerated_cv_vectors = np.asarray(agglomerated_cv_vectors).reshape(-1, 3)
        self.agglomerated_cv_vdata = np.asarray(agglomerated_cv_vdata).reshape(-1, 3)
