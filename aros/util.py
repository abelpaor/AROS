import trimesh
import numpy as np
import math
import open3d as o3d
import point_cloud_utils as pcu

def compare_nan_array(func, a, thresh):
    """
    Compare element of an array a using the function passed as argument (np.greater, np.greater_equal, np.less,
    np.less_equal, np.equal, np.not_equal). Helpful to avoid RuntimeWarning provoked by comparisson
    with math.nan values
    Parameters
    ----------
    func - Function used to compare
    a - array to compare
    thresh - value used as threshold of the comparisson

    Returns
    -------
    Array of boolean values with the result after comparisson with a thresh
    """
    out = ~np.isnan(a)
    out[out] = func(a[out] , thresh)
    return out


def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::
    >>> angle_between((1, 0, 0), (0, 1, 0))
    1.5707963267948966
    >>> angle_between((1, 0, 0), (1, 0, 0))
    0.0
    >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def sample_points_poisson_disk(tri_mesh, number_of_points, init_factor=5):
    o3d_mesh = o3d.geometry.TriangleMesh()
    o3d_mesh.vertices = o3d.utility.Vector3dVector(tri_mesh.vertices)
    o3d_mesh.triangles = o3d.utility.Vector3iVector(tri_mesh.faces)

    od3_cloud_poisson = o3d.geometry.TriangleMesh.sample_points_poisson_disk(o3d_mesh, number_of_points, init_factor)

    return np.asarray(od3_cloud_poisson.points)

def sample_points_poisson_disk_radius(tri_mesh, radius=0.01, use_geodesic_distance= True, best_choice_sampling = True):
    """
    Generates samples so that them are approximately evenly separated.
    :param tri_mesh: mesh to sample
    :param radius: the desired separation
    :param use_geodesic_distance:
    :param best_choice_sampling:
    :return:
    """
    vertices = np.asarray(tri_mesh.vertices)
    triangles = np.asarray(tri_mesh.faces)
    normals = np.asarray(tri_mesh.vertex_normals, order='C')

    v_poisson, n_poisson = pcu.sample_mesh_poisson_disk(vertices, triangles, normals, num_samples=-1, radius=radius,
                                                        use_geodesic_distance=use_geodesic_distance,
                                                        best_choice_sampling=best_choice_sampling, random_seed=0)

    return v_poisson, n_poisson


def get_normal_nearest_point_in_mesh(tri_mesh, sampled_points):
    """
    Get the normal of the nearest point in mesh
    CAUTION: It generates files while calculating, it seems that using in a parallel way could generate some crash
    in execution
    :param tri_mesh:
    :param sampled_points:
    :return:
    """
    normals = []
    (closest_points, distances, triangle_id) = tri_mesh.nearest.on_surface(sampled_points)
    normals.append( tri_mesh.face_normals[triangle_id] )
    return  np.asarray(normals).reshape(-1,3)



def slide_mesh_by_bounding_box(tri_mesh, box_center, box_extension):
    max_x_plane = box_center + np.array([box_extension, 0, 0])
    min_x_plane = box_center - np.array([box_extension, 0, 0])
    max_y_plane = box_center + np.array([0, box_extension, 0])
    min_y_plane = box_center - np.array([0, box_extension, 0])
    max_z_plane = box_center + np.array([0, 0, box_extension])
    min_z_plane = box_center - np.array([0, 0, box_extension])

    extracted = tri_mesh.slice_plane(plane_normal=np.array([-1, 0, 0]), plane_origin=max_x_plane)
    extracted = extracted.slice_plane(plane_normal=np.array([1, 0, 0]), plane_origin=min_x_plane)

    extracted = extracted.slice_plane(plane_normal=np.array([0, -1, 0]), plane_origin=max_y_plane)
    extracted = extracted.slice_plane(plane_normal=np.array([0, 1, 0]), plane_origin=min_y_plane)

    extracted = extracted.slice_plane(plane_normal=np.array([0, 0, -1]), plane_origin=max_z_plane)
    extracted = extracted.slice_plane(plane_normal=np.array([0, 0, 1]), plane_origin=min_z_plane)

    return extracted


def slide_mesh_by_sphere(tri_mesh, sphere_center, sphere_ro, level=16):
    # angle with respect X axis and Z
    angle = [x * 2 * math.pi / level for x in range(level)]
    output_mesh = trimesh.Trimesh(vertices=tri_mesh.vertices,
                                  faces=tri_mesh.faces,
                                  process=False)
    points_on_sphere = []
    normals = []
    for theta in angle:
        for phi in angle:
            point = np.array([sphere_ro * math.cos(theta) * math.sin(phi),
                            sphere_ro * math.sin(theta) * math.sin(phi),
                            sphere_ro * math.cos(phi)])

            sphere_point = point + sphere_center
            points_on_sphere.append(sphere_point)
            normals.append( sphere_center - sphere_point )

    points_on_sphere,idxs = np.unique( np.asarray(points_on_sphere).reshape(-1, 3), return_index=True, axis=0)
    normals = (np.asarray(normals).reshape(-1, 3))[idxs]
    output_mesh = output_mesh.slice_plane(plane_normal=normals, plane_origin=points_on_sphere)

    return output_mesh


def extract_cloud_by_bounding_box(np_cloud, box_center, box_extension):
    max_x_plane = box_center[0] + box_extension
    min_x_plane = box_center[0] - box_extension
    max_y_plane = box_center[1] + box_extension
    min_y_plane = box_center[1] - box_extension
    max_z_plane = box_center[2] + box_extension
    min_z_plane = box_center[2] - box_extension

    extracted = [point for point in np_cloud if point[0] > min_x_plane]
    extracted = [point for point in extracted if point[0] < max_x_plane]
    extracted = [point for point in extracted if point[1] > min_y_plane]
    extracted = [point for point in extracted if point[1] < max_y_plane]
    extracted = [point for point in extracted if point[2] > min_z_plane]
    extracted = [point for point in extracted if point[2] < max_z_plane]

    return np.asarray(extracted)


def extract_cloud_by_sphere(np_cloud, np_sphere_centre, sphere_ro):
    o3d_cloud = o3d.geometry.PointCloud()
    o3d_cloud.points = o3d.utility.Vector3dVector(np_cloud)

    pcd_tree = o3d.geometry.KDTreeFlann(o3d_cloud)

    # it returns 1: number of point returned, 2: idx of point returned, 3: distances
    [__, idx, __] = pcd_tree.search_radius_vector_3d(np_sphere_centre, sphere_ro)

    return idx, np_cloud[idx]


def get_edges(vertices, ridge_vertices, idx_extracted=None):
    # cutting faces in the polygon mesh
    faces_idx_ridges_from = []
    faces_idx_ridges_to = []
    for ridge in ridge_vertices:
        if -1 in ridge:
            continue
        if idx_extracted is not None:
            face_in_boundary = True
            for idx_vertex in ridge:
                if idx_vertex not in idx_extracted:
                    face_in_boundary = False
                    break
            if not face_in_boundary:
                continue

        for i in range(-1, len(ridge) - 1):
            faces_idx_ridges_from.append(ridge[i])
            faces_idx_ridges_to.append(ridge[i + 1])

    edges_from = vertices[faces_idx_ridges_from]
    edges_to = vertices[faces_idx_ridges_to]

    return edges_from, edges_to


def influence_sphere(tri_mesh_obj, radio_ratio=1.5):
    '''
    Defines radio of influence of a given object
    :param tri_mesh_obj: Object to calculate sphere of influence
    :param radio_ratio: expansion of sphere of influence 1.1 means 110%, 2 mean 200% (ro is as long as the
    diagonal of the bounding box of the object)
    :return: center and ro parameter of the sphere of influence
    '''
    obj_min_bound = np.asarray(tri_mesh_obj.vertices).min(axis=0)
    obj_max_bound = np.asarray(tri_mesh_obj.vertices).max(axis=0)
    sphere_center = np.asarray(obj_max_bound + obj_min_bound) / 2
    sphere_ro = np.linalg.norm(obj_max_bound - sphere_center) * radio_ratio

    return sphere_ro, sphere_center
