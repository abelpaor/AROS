import json
from os import path
import numpy as np
import open3d as o3d


class Deglomerator:

    definition = None
    num_orientations = None
    sample_size = None
    influence_radio = None
    normal_env = None
    pv_points = None
    pv_vectors = None
    pv_data = None
    affordance_name = None
    object_name = None
    working_path = None

    def __init__(self, working_path, affordance_name, object_name):
        #print(affordance_name + ' ' + object_name)
        self.affordance_name = affordance_name
        self.object_name = object_name
        self.working_path = working_path
        self.read_definition()
        self.readAgglomeratedDescriptor()

    def read_definition(self):
        definition_file = self.working_path + '/' + self.affordance_name + "_" + self.object_name + ".json"
        with open(definition_file) as jsonfile:
            self.definition = json.load(jsonfile)
        self.num_orientations = int(self.definition['orientations'])
        self.sample_size = int(self.definition["trainer"]["sampler"]['sample_size'])
        self.influence_radio = self.definition['max_distances']['obj_influence_radio']
        self.normal_env = np.fromstring(self.definition['trainer']['normal_env'], sep=',')

    def readAgglomeratedDescriptor(self):
        base_nameU = self.get_agglomerated_files_name_pattern()

        self.pv_points = np.asarray(o3d.io.read_point_cloud(base_nameU + "_points.pcd").points)
        self.pv_vectors = np.asarray(o3d.io.read_point_cloud(base_nameU + "_vectors.pcd").points)
        self.pv_data = np.asarray(o3d.io.read_point_cloud(base_nameU + "_vdata.pcd").points)

    def object_filename(self):
        obj_filename = path.join(self.working_path, self.affordance_name + "_" + self.object_name + "_object.ply")
        return obj_filename

    def get_agglomerated_files_name_pattern(self):
        base_nameU = self.working_path + "/UNew_" + self.affordance_name + "_" + self.object_name + "_descriptor_" + str(
            self.definition['orientations'])
        return base_nameU



class DeglomeratorClearance(Deglomerator):
    cv_points = None
    cv_vectors = None
    cv_vectors_norms = None
    sample_clearance_size = None

    def __init__(self, working_path, affordance_name, object_name):
        super().__init__(working_path, affordance_name, object_name)

    def read_definition(self):
        super().read_definition()
        self.sample_clearance_size = self.definition["trainer"]["cv_sampler"]["sample_clearance_size"]

    def readAgglomeratedDescriptor(self):
        super().readAgglomeratedDescriptor()
        base_nameU = self.get_agglomerated_files_name_pattern()
        self.cv_points = np.asarray(o3d.io.read_point_cloud(base_nameU + "_clearance_points.pcd").points)
        self.cv_vectors = np.asarray(o3d.io.read_point_cloud(base_nameU + "_clearance_vectors.pcd").points)
        self.cv_vectors_norms = np.asarray(o3d.io.read_point_cloud(base_nameU + "_clearance_vdata.pcd").points)[:, 0]
