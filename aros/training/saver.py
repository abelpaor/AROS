import math
import json
import os
import numpy as np
import open3d as o3d
from transforms3d.derivations.eulerangles import z_rotation

from aros import util
from aros.training.agglomerator import AgglomeratorClearance


class Saver:
    output_dir = os.path.join('.', 'output', 'descriptors_repository')
    directory = None

    def __init__(self, affordance_name, env_name, obj_name, agglomerator, max_distances,  ibs_calculator, tri_mesh_obj,
                 output_subdir=None):
        if output_subdir is not None:
            self.output_dir = os.path.join(self.output_dir, output_subdir)

        self.directory = os.path.join(self.output_dir, affordance_name)

        if not os.path.exists(self.directory):
            os.makedirs(self.directory)

        self._save_info(affordance_name, env_name, obj_name, agglomerator, max_distances, ibs_calculator, tri_mesh_obj)
        self._save_agglomerated_it_descriptor(affordance_name, obj_name, agglomerator)
        self._save_meshes(affordance_name, obj_name, agglomerator, ibs_calculator, tri_mesh_obj)
        self._save_maxdistances(affordance_name, obj_name, max_distances);

    def _save_meshes(self, affordance_name, obj_name, agglomerator, ibs_calculator, tri_mesh_obj):
        file_name_pattern = os.path.join(self.directory, affordance_name + "_" + obj_name)

        tri_mesh_env = agglomerator.it_trainer.sampler.tri_mesh_env
        tri_mesh_ibs_segmented = agglomerator.it_trainer.sampler.tri_mesh_ibs
        tri_mesh_ibs = ibs_calculator.get_trimesh()

        tri_mesh_ibs_segmented.export(file_name_pattern + "_ibs_mesh_segmented.ply", "ply")
        tri_mesh_ibs.export(file_name_pattern + "_ibs_mesh.ply", "ply")
        tri_mesh_env.export(file_name_pattern + "_environment.ply", "ply")
        tri_mesh_obj.export(file_name_pattern + "_object.ply", "ply")

    def _save_agglomerated_it_descriptor(self, affordance_name, obj_name, agglomerator):
        file_name_pattern = os.path.join(self.directory, "UNew_" + affordance_name + "_" +
                                         obj_name + "_descriptor_" + str(agglomerator.ORIENTATIONS))

        pcd = o3d.geometry.PointCloud()

        pcd.points = o3d.utility.Vector3dVector(agglomerator.agglomerated_pv_points)
        o3d.io.write_point_cloud(file_name_pattern + "_points.pcd", pcd, write_ascii=True)

        pcd.points = o3d.utility.Vector3dVector(agglomerator.agglomerated_pv_vectors)
        o3d.io.write_point_cloud(file_name_pattern + "_vectors.pcd", pcd, write_ascii=True)

        pcd.points = o3d.utility.Vector3dVector(agglomerator.agglomerated_pv_vdata)
        o3d.io.write_point_cloud(file_name_pattern + "_vdata.pcd", pcd, write_ascii=True)

        pcd.points = o3d.utility.Vector3dVector(agglomerator.agglomerated_normals)
        o3d.io.write_point_cloud(file_name_pattern + "_normals_env.pcd", pcd, write_ascii=True)


    def _save_info(self, affordance_name, env_name, obj_name, agglomerator,max_distances, ibs_calculator, tri_mesh_obj):
        data = {}
        data['it_descriptor_version'] = 2.0
        data['affordance_name'] = affordance_name
        data['env_name'] = env_name
        data['obj_name'] = obj_name
        #data['obj_influence_radio'],__ = util.influence_sphere(tri_mesh_obj)
        #data['sample_size'] = agglomerator.it_trainer.sampler.SAMPLE_SIZE
        data['orientations'] = agglomerator.ORIENTATIONS
        data['trainer'] = agglomerator.it_trainer.get_info()
        data['ibs_calculator'] = ibs_calculator.get_info()
        data['max_distances'] = max_distances.get_info()
        # data['reference'] = {}
        # data['reference']['idxRefIBS'] = 8
        # data['reference']['refPointIBS'] = '8,8,8'
        # data['scene_point'] = {}
        # data['scene_point']['idxScenePoint'] = 9
        # data['scene_point']['refPointScene'] = '9,9,9'
        # data['ibs_point_vector'] = {}
        # data['ibs_point_vector']['idx_ref_ibs'] = 10
        # data['ibs_point_vector']['vect_scene_to_ibs'] = '10,10,10'
        # data['obj_point_vector'] = {}
        # data['obj_point_vector']['idx_ref_object'] = 11
        # data['obj_point_vector']['vect_scene_to_object'] = '11,11,11'

        with open(os.path.join(self.directory, affordance_name + '_' + obj_name + '.json'), 'w') as outfile:
            json.dump(data, outfile, indent=4)

    def _save_maxdistances(self,affordance_name, obj_name, max_distances):
       output_file = os.path.join(self.directory, affordance_name + '_' + obj_name + '_maxdistances.txt')
       np.savetxt( output_file,  max_distances.max_distances)






class SaverClearance(Saver):

    def __init__(self, affordance_name, env_name, obj_name, agglomerator, max_distances, ibs_calculator, tri_mesh_obj,
                 output_subdir=None):

        assert isinstance(agglomerator, AgglomeratorClearance)

        super().__init__(affordance_name, env_name, obj_name, agglomerator, max_distances, ibs_calculator, tri_mesh_obj,
                 output_subdir)

    def _save_agglomerated_it_descriptor(self, affordance_name, obj_name, agglomerator):
        file_name_pattern = os.path.join(self.directory, "UNew_" + affordance_name + "_" +
                                         obj_name + "_descriptor_" + str(agglomerator.ORIENTATIONS))
        pcd = o3d.geometry.PointCloud()

        pcd.points = o3d.utility.Vector3dVector(agglomerator.agglomerated_pv_points)
        o3d.io.write_point_cloud(file_name_pattern + "_points.pcd", pcd, write_ascii=True)

        pcd.points = o3d.utility.Vector3dVector(agglomerator.agglomerated_pv_vectors)
        o3d.io.write_point_cloud(file_name_pattern + "_vectors.pcd", pcd, write_ascii=True)

        pcd.points = o3d.utility.Vector3dVector(agglomerator.agglomerated_pv_vdata)
        o3d.io.write_point_cloud(file_name_pattern + "_vdata.pcd", pcd, write_ascii=True)

        pcd.points = o3d.utility.Vector3dVector(agglomerator.agglomerated_normals)
        o3d.io.write_point_cloud(file_name_pattern + "_normals_env.pcd", pcd, write_ascii=True)

        pcd.points = o3d.utility.Vector3dVector(agglomerator.agglomerated_cv_points)
        o3d.io.write_point_cloud(file_name_pattern + "_clearance_points.pcd", pcd, write_ascii=True)

        pcd.points = o3d.utility.Vector3dVector(agglomerator.agglomerated_cv_vectors)
        o3d.io.write_point_cloud(file_name_pattern + "_clearance_vectors.pcd", pcd, write_ascii=True)

        pcd.points = o3d.utility.Vector3dVector(agglomerator.agglomerated_cv_vdata)
        o3d.io.write_point_cloud(file_name_pattern + "_clearance_vdata.pcd", pcd, write_ascii=True)


    def _save_info(self, affordance_name, env_name, obj_name, agglomerator, max_distances, ibs_calculator,
                   tri_mesh_obj):
        data = {'it_descriptor_version': 2.1,
                'affordance_name': affordance_name,
                'env_name': env_name,
                'obj_name': obj_name,
                # 'sample_size': agglomerator.it_trainer.sampler.SAMPLE_SIZE,
                'orientations': agglomerator.ORIENTATIONS,
                'trainer': agglomerator.it_trainer.get_info(),
                'ibs_calculator': ibs_calculator.get_info(),
                'max_distances': max_distances.get_info()}

        with open(os.path.join(self.directory, affordance_name + '_' + obj_name + '.json'), 'w') as outfile:
            json.dump(data, outfile, indent=4)

