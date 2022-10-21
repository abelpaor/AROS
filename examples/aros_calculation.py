from os.path import join as opj

import trimesh
from vedo import Plotter, trimesh2vtk, Lines, Spheres, Arrows

from aros import util
from aros.training.agglomerator import AgglomeratorClearance
from aros.training.ibs import IBSMesh
from aros.training.maxdistancescalculator import MaxDistancesCalculator
from aros.training.sampler import OnGivenPointCloudWeightedSampler, PropagateNormalObjectPoissonDiscSamplerClearance
from aros.training.saver import SaverClearance
from aros.training.trainer import TrainerClearance

if __name__ == '__main__':

    base_dir = "../data/sitting_sofa_no_optimization"
    human_sitting = opj(base_dir, "sitting_human.ply")
    scene = opj(base_dir,"sofa.ply")


    ibs_init_size_sampling = 400
    ibs_resamplings = 4
    sampler_rate_ibs_samples = 5
    sampler_rate_generated_random_numbers = 500


    tri_mesh_env = trimesh.load_mesh(scene)
    tri_mesh_obj = trimesh.load_mesh(human_sitting)

    obj_name = "human_sitting"
    env_name = "sofa"
    affordance_name = "sit-able"

    influence_radio_bb = 2
    extension, middle_point = util.influence_sphere(tri_mesh_obj, influence_radio_bb)
    tri_mesh_env_segmented = util.slide_mesh_by_bounding_box(tri_mesh_env, middle_point, extension)

    ################################
    # GENERATING AND SEGMENTING IBS MESH
    ################################
    influence_radio_ratio = 1.2
    ibs_calculator = IBSMesh(ibs_init_size_sampling, ibs_resamplings)
    ibs_calculator.execute(tri_mesh_env_segmented, tri_mesh_obj)

    tri_mesh_ibs = ibs_calculator.get_trimesh()

    sphere_ro, sphere_center = util.influence_sphere(tri_mesh_obj, influence_radio_ratio)
    tri_mesh_ibs_segmented = util.slide_mesh_by_sphere(tri_mesh_ibs, sphere_center, sphere_ro)

    np_cloud_env = ibs_calculator.points[: ibs_calculator.size_cloud_env]
    np_cloud_obj = ibs_calculator.points[ibs_calculator.size_cloud_env:]

    print("hello")

    # vp.camera = old_camera
    l_to_plot = []

    vedo_env = trimesh2vtk(tri_mesh_env).c((.3, .3, .3)).alpha(1)
    vedo_obj = trimesh2vtk(tri_mesh_obj).c((0, 1, 0)).alpha(1)
    vedo_ibs = trimesh2vtk(tri_mesh_ibs_segmented).c((0, 0, 1)).alpha(.39)

    vedo_env.lighting(ambient=0.5, diffuse=0.4, specular=0.2, specularPower=1, specularColor=(1, 1, 1))
    vedo_obj.lighting(ambient=0.5, diffuse=0.2, specular=0.1, specularPower=1, specularColor=(1, 1, 1))

    l_to_plot.append(vedo_env)
    l_to_plot.append(vedo_obj)

    vp = Plotter(bg="white", size=(1600, 1200))
    vp.show(l_to_plot)
    camera = vp.camera

    # voronoi so difficult to observe
    # for ridge in ibs_calculator.voro.ridge_vertices:
    #     if -1 in ridge:
    #         continue
    #     to_far=False
    #     points_3d =[]
    #     for pos_point in ridge:
    #         if np.linalg.norm(ibs_calculator.voro.vertices[pos_point]) > influence_radio_bb:
    #             to_far=True
    #             break
    #         else:
    #             points_3d.append(ibs_calculator.voro.vertices[pos_point])
    #     if not to_far:
    #         l_to_plot.append(vedo.Line(points_3d, closed=False, c="black", alpha=1, lw=1))

    vp = Plotter(bg="white", size=(1600, 1200))
    l_to_plot.append(vedo_ibs)
    vp.camera=camera
    vp.show(l_to_plot)






    ################################
    # SAMPLING IBS MESH
    ################################

    pv_sampler = OnGivenPointCloudWeightedSampler(np_input_cloud=np_cloud_env,
                                                  rate_generated_random_numbers=sampler_rate_generated_random_numbers)


    cv_sampler = PropagateNormalObjectPoissonDiscSamplerClearance()
    trainer = TrainerClearance(tri_mesh_ibs=tri_mesh_ibs_segmented, tri_mesh_env=tri_mesh_env,
                               tri_mesh_obj=tri_mesh_obj, pv_sampler=pv_sampler, cv_sampler=cv_sampler)


    clearance_vectors = Lines(trainer.cv_points, trainer.cv_points + trainer.cv_vectors, c=(252,147,0), lw=3, alpha=1).lighting("plastic")
    provenance_vectors = Arrows(trainer.pv_points, trainer.pv_points + trainer.pv_vectors, c='red', alpha=1).lighting("plastic")
    cv_from = Spheres(trainer.cv_points, r=.002, c=[(252,147,0)]*len(trainer.cv_points), alpha=1).lighting("plastic")

    l_to_plot.append(clearance_vectors)
    l_to_plot.append(provenance_vectors)
    l_to_plot.append(cv_from)

    vp = Plotter(bg="white", size=(1600, 1200))
    vp.camera=camera
    vp.show(l_to_plot)



    l_to_plot = []
    l_to_plot.append(vedo_env)
    l_to_plot.append(vedo_obj)
    l_to_plot.append(vedo_ibs)
    l_to_plot.append(clearance_vectors)
    l_to_plot.append(provenance_vectors)
    l_to_plot.append(cv_from)
    vp = Plotter(bg="white", size=(1600, 1200))
    vp.show(l_to_plot)
    camera = vp.camera


    output_subdir = "IBSMesh_" + str(ibs_init_size_sampling) + "_" + str(ibs_resamplings) + "_"
    output_subdir += pv_sampler.__class__.__name__ + "_" + str(sampler_rate_ibs_samples) + "_"
    output_subdir += str(sampler_rate_generated_random_numbers) + "_"
    output_subdir += cv_sampler.__class__.__name__ + "_" + str(cv_sampler.sample_size)

    agglomerator = AgglomeratorClearance(trainer, num_orientations=8)
    max_distances = MaxDistancesCalculator(pv_points=trainer.pv_points, pv_vectors=trainer.pv_vectors,
                                           tri_mesh_obj=tri_mesh_obj, consider_collision_with_object=True,
                                           radio_ratio=influence_radio_ratio)
    SaverClearance(affordance_name, env_name, obj_name, agglomerator,
                   max_distances, ibs_calculator, tri_mesh_obj, output_subdir)