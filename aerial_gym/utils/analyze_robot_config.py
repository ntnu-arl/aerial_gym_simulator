import numpy as np
import itertools
from scipy.spatial import ConvexHull
import itertools
import matplotlib.pyplot as plt
from cvxopt import matrix, solvers
import torch

def check_flyability(robot_model):
    
    #max_force = calc_maximum_admissible_force(robot_model.convex_hull_admissible_set_vertices)
    #check_hover = (max_force >= robot_model.total_mass*9.81)
    
    wrench_hover_F = np.array([0,0,robot_model.total_mass*9.81])
    check_hover,_ = check_wrench_in_admissible_set(robot_model.convex_hull_admissible_set_vertices[:,:3],wrench_hover_F)
    
    min_authority = 1e-2
    wrench_Mx_pl = np.array([min_authority,0,0])
    wrench_Mx_mi = np.array([-min_authority,0,0])
    wrench_My_pl = np.array([0,min_authority,0])
    wrench_My_mi = np.array([0,-min_authority,0])
    wrench_Mz_pl = np.array([0,0,min_authority])
    wrench_Mz_mi = np.array([0,0,-min_authority])
    check_Mx_pl,_ = check_wrench_in_admissible_set(robot_model.convex_hull_admissible_set_vertices[:,3:],wrench_Mx_pl)
    check_Mx_mi,_ = check_wrench_in_admissible_set(robot_model.convex_hull_admissible_set_vertices[:,3:],wrench_Mx_mi)
    check_My_pl,_ = check_wrench_in_admissible_set(robot_model.convex_hull_admissible_set_vertices[:,3:],wrench_My_pl)
    check_My_mi,_ = check_wrench_in_admissible_set(robot_model.convex_hull_admissible_set_vertices[:,3:],wrench_My_mi)
    check_Mz_pl,_ = check_wrench_in_admissible_set(robot_model.convex_hull_admissible_set_vertices[:,3:],wrench_Mz_pl)
    check_Mz_mi,_ = check_wrench_in_admissible_set(robot_model.convex_hull_admissible_set_vertices[:,3:],wrench_Mz_mi)
    check_authority = check_Mx_pl and check_Mx_mi and check_My_pl and check_My_mi and check_Mz_pl and check_Mz_mi
    
    return check_hover and check_authority
    

def analyze_robot_config(robot_model):
    """ Analyze the robot configuration and print the results.
    Args:
        robot_model (RobotModel object): robot model defining the URDF
    """
    
    print("############ CONFIGURATION EVALUATION ############")
    print("\n")
    
    print("Mass and Inertia: \n   - total mass: " + str(robot_model.total_mass) + "\n   - inertia matrix: \n \n " + str(robot_model.inertia_matrix)+ "\n")
    
    print("Actuation:")
    
    # check hover ability
    wrench_hover_F = np.array([0,0,robot_model.total_mass*9.81])
    check,_ = check_wrench_in_admissible_set(robot_model.convex_hull_admissible_set_vertices[:,:3],wrench_hover_F)
    print("   - Hovering upright in \"0\" orientation is in the admissible set: " + str(check))
    
    max_force = calc_maximum_admissible_force(robot_model.convex_hull_admissible_set_vertices)
    check = (max_force >= robot_model.total_mass*9.81)
    print("   - There exist a orientation in which the robot can hover: " + str(check))
    
    # check symmetry
    check = check_symmetry_admissible_subset(calc_convex_hull_admissible_set(robot_model.min_u, 
                                                                             robot_model.max_u,
                                                                             robot_model.actuator_mixing_matrix[:3]),
                                                                             "xy")
    print("   - Adm.force set is symmetric with respect to the xy-plane: " + str(check))
    check = check_symmetry_admissible_subset(calc_convex_hull_admissible_set(robot_model.min_u, 
                                                                             robot_model.max_u,
                                                                             robot_model.actuator_mixing_matrix[:3]),
                                                                             "xz")
    print("   - Adm.force set is symmetric with respect to the xz-plane: " + str(check))
    check = check_symmetry_admissible_subset(calc_convex_hull_admissible_set(robot_model.min_u, 
                                                                             robot_model.max_u,
                                                                             robot_model.actuator_mixing_matrix[:3]),
                                                                             "yz")
    print("   - Adm.force set is symmetric with respect to the yz-plane: " + str(check))
    check = check_symmetry_admissible_subset(calc_convex_hull_admissible_set(robot_model.min_u, 
                                                                             robot_model.max_u,
                                                                             robot_model.actuator_mixing_matrix[3:]),
                                                                             "xy")
    print("   - Adm.torque set is symmetric with respect to the xy-plane: " + str(check))
    check = check_symmetry_admissible_subset(calc_convex_hull_admissible_set(robot_model.min_u, 
                                                                             robot_model.max_u,
                                                                             robot_model.actuator_mixing_matrix[3:]),
                                                                             "xz")
    print("   - Adm.torque set is symmetric with respect to the xz-plane: " + str(check))
    check = check_symmetry_admissible_subset(calc_convex_hull_admissible_set(robot_model.min_u, 
                                                                             robot_model.max_u,
                                                                             robot_model.actuator_mixing_matrix[3:]),
                                                                             "yz")
    print("   - Adm.torque set is symmetric with respect to the yz-plane: " + str(check))
    
    print(np.shape(robot_model.convex_hull_admissible_set_vertices))
    
    # check control authority around each axis
    min_authority = 1e-2
    wrench_Mx_pl = np.array([min_authority,0,0])
    wrench_Mx_mi = np.array([-min_authority,0,0])
    wrench_My_pl = np.array([0,min_authority,0])
    wrench_My_mi = np.array([0,-min_authority,0])
    wrench_Mz_pl = np.array([0,0,min_authority])
    wrench_Mz_mi = np.array([0,0,-min_authority])
    check_Mx_pl,_ = check_wrench_in_admissible_set(robot_model.convex_hull_admissible_set_vertices[:,3:],wrench_Mx_pl)
    check_Mx_mi,_ = check_wrench_in_admissible_set(robot_model.convex_hull_admissible_set_vertices[:,3:],wrench_Mx_mi)
    check_My_pl,_ = check_wrench_in_admissible_set(robot_model.convex_hull_admissible_set_vertices[:,3:],wrench_My_pl)
    check_My_mi,_ = check_wrench_in_admissible_set(robot_model.convex_hull_admissible_set_vertices[:,3:],wrench_My_mi)
    check_Mz_pl,_ = check_wrench_in_admissible_set(robot_model.convex_hull_admissible_set_vertices[:,3:],wrench_Mz_pl)
    check_Mz_mi,_ = check_wrench_in_admissible_set(robot_model.convex_hull_admissible_set_vertices[:,3:],wrench_Mz_mi)
    print("   - attitude control authority around each axis: " + str(check_Mx_pl and check_Mx_mi and 
                                                                     check_My_pl and check_My_mi and 
                                                                     check_Mz_pl and check_Mz_mi))
    print("\n")
            
    print("##################################################")
    
    #visualize_admissible_set_forces(robot_model,wrench=np.array([0,0,0,0,0,0]))
    #visualize_admissible_set_torques(robot_model,wrench=np.array([0,0,0,0,0,0]))

def visualize_admissible_set_forces(robot_model,wrench=None):
    """ Visualize the admissible set of forces.
    Args:
        robot_model (RobotModel object): robot model defining the URDF
        wrench (numpy array (6,)): wrench to be visualized in the admissible set 
    """
    
    num_points = 0
    n = 1
    while num_points < 5000:
        control_set = np.array(list(itertools.product(np.linspace(robot_model.min_u,robot_model.max_u,num=n),repeat = robot_model.n_motors)))
        n += 1
        num_points = control_set.shape[0]
    
    if isinstance(robot_model.actuator_mixing_matrix,np.ndarray):
        force_actuator_mixing_matrix = robot_model.actuator_mixing_matrix[:3]
    elif torch.is_tensor(robot_model.actuator_mixing_matrix):
        force_actuator_mixing_matrix = robot_model.actuator_mixing_matrix[:3].cpu().numpy()
        
    approx_admissible_set_forces = (force_actuator_mixing_matrix @ control_set.T).T
    
    # NOTE the number of points in the convex hull changes when mapping samples from
    # the control set instead of taking the vertices of the control set. This should
    # not be the case and is due to numerical issues when calculating the convex hull. 
    # The additional points found in the convex hull can still be expressed as convex linear 
    # combinations of the vertices found with mapping the vertices of the control set. 
    
    # if vertices collaps to a lower dimensional hyper plane jiggle them a bit
    try:
        convex_hull_admissible_set_forces = ConvexHull(approx_admissible_set_forces )
    except:
        convex_hull_admissible_set_forces = ConvexHull(approx_admissible_set_forces , qhull_options='QJ')
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(approx_admissible_set_forces[:,0], approx_admissible_set_forces[:,1], approx_admissible_set_forces[:,2], color='r', marker='o', s=10)
    ax.scatter(approx_admissible_set_forces[convex_hull_admissible_set_forces.vertices,0],
                approx_admissible_set_forces[convex_hull_admissible_set_forces.vertices,1],
                approx_admissible_set_forces[convex_hull_admissible_set_forces.vertices,2], color='b', marker='o', s=30)
    if wrench is not None:
        ax.scatter(wrench[0], wrench[1], wrench[2], color='g', marker='*', s=50)
        
    #ax.set_aspect('equal', adjustable='box')
    
    ax.set_xlabel("fx [N]")
    ax.set_ylabel("fy [N]")
    ax.set_zlabel("fz [N]")

    plt.show()

def visualize_admissible_set_torques(robot_model,wrench=None):
    """ Visualize the admissible set of torques.
    Args:
        robot_model (RobotModel object): robot model defining the URDF
        wrench (numpy array (6,)): wrench to be visualized in the admissible set
    """

    num_points = 0
    n = 1
    while num_points < 5000:
        control_set = np.array(list(itertools.product(np.linspace(robot_model.min_u,robot_model.max_u,num=n),repeat = robot_model.n_motors)))
        n+=1
        num_points = control_set.shape[0]
    
    if isinstance(robot_model.actuator_mixing_matrix,np.ndarray):
        torque_actuator_mixing_matrix = robot_model.actuator_mixing_matrix[3:]
    elif torch.is_tensor(robot_model.actuator_mixing_matrix):
        torque_actuator_mixing_matrix = robot_model.actuator_mixing_matrix[3:].cpu().numpy()
    
    approx_admissible_set_torques = (torque_actuator_mixing_matrix @ control_set.T).T
    
    # NOTE the number of points in the convex hull changes when mapping samples from
    # the control set instead of taking the vertices of the control set. This should
    # not be the case and is due to numerical issues when calculating the convex hull. 
    # The additional points found in the convex hull can still be expressed as convex linear 
    # combinations of the vertices found with mapping the vertices of the control set. 
    
    # if vertices collaps to a lower dimensional hyper plane jiggle them a bit
    try:
        convex_hull_admissible_set_torques = ConvexHull(approx_admissible_set_torques)
    except:
        convex_hull_admissible_set_torques = ConvexHull(approx_admissible_set_torques , qhull_options='QJ')
    
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d')
    ax.scatter(approx_admissible_set_torques[:,0], approx_admissible_set_torques[:,1], approx_admissible_set_torques[:,2], color='r', marker='o', s=10)
    ax.scatter(approx_admissible_set_torques[convex_hull_admissible_set_torques.vertices,0],
            approx_admissible_set_torques[convex_hull_admissible_set_torques.vertices,1],
            approx_admissible_set_torques[convex_hull_admissible_set_torques.vertices,2], color='b', marker='o', s=30)
    if wrench is not None:
        ax.scatter(wrench[3], wrench[4], wrench[5], color='g', marker='*', s=50)
    
    #ax.set_aspect('equal', adjustable='box')
    
    ax.set_xlabel("Mx [Nm]")
    ax.set_ylabel("My [Nm]")
    ax.set_zlabel("Mz [Nm]")
    
    plt.show()

def calc_convex_hull_admissible_set(u_min, u_max, actuator_mixing_matrix):
    """ Calculates the convex hull of the admissible wrench set.
    Args: 
        u_min (float): minimum control input
        u_max (float): maximum control input
        actuator_mixing_matrix (numpy array (6, #motors)): actuator mixing matrix
    """
    
    vertices_control_set = np.array(list(itertools.product([u_min,u_max], repeat=actuator_mixing_matrix.shape[1])))
    image_vertices_control_set = (actuator_mixing_matrix @ vertices_control_set.T).T
    
    # if vertices collaps to a lower dimensional hyper plane jiggle them a bit
    try:
        convex_hull_admissible_set = ConvexHull(image_vertices_control_set)
    except:
        convex_hull_admissible_set = ConvexHull(image_vertices_control_set, qhull_options='QJ')
    
    convex_hull_admissible_set_vertices = image_vertices_control_set[convex_hull_admissible_set.vertices]
    
    return convex_hull_admissible_set_vertices

def calc_maximum_admissible_force(admissible_set_vertices):
    """ Calculates the maximum admissible force.
    Args: 
        admissible_set_vertices (numpy array (#motors,6)): vertices of the admissible set
    Returns:
        norm_maximum_admissible_force (float): norm of the maximal admissible force
    """
    
    norm_maximum_admissible_force = np.linalg.norm(admissible_set_vertices[:,:3], axis=1).max()
    
    return norm_maximum_admissible_force

def check_symmetry_admissible_subset(admissible_subset_vertices,sym_plane):
    """ Really basic symmetry check. Checks if a point cloud is symmetric with respect to a given coordinate plane.
        Points lying on the plane do not break the symmetry. If two points on the same spot on one side 
        have only one symmetric partner on the other side the symmetry is broken. 
    Args:
        admissible_subset_vertices (numpy array (#motors,3)): admissible subset vertices (force or torque subset)
        sym_plane (string): plane of symmetry ("xy", "xz", "yz")
    Raises:
        ValueError: not recognized symmetry plane
    Returns:
        check (bool): true if the admissible subset is symmetric
    """
    
    v = admissible_subset_vertices
    n_points = v.shape[0]
    sym_crit = np.zeros((n_points,n_points))
    for i in range(n_points):
        for j in range(n_points):
            if sym_plane == "xy":
                sym_crit[i,j] = sum(abs((v[i] + v[j] - [2.0*v[i,0],2.0*v[i,1],0])*v[i,2]))
            elif sym_plane == "xz":
                sym_crit[i,j] = sum(abs((v[i] + v[j] - [2.0*v[i,0],0,2.0*v[i,2]])*v[i,1]))
            elif sym_plane == "yz":
                sym_crit[i,j] = sum(abs((v[i] + v[j] - [0,2.0*v[i,1],2.0*v[i,2]])*v[i,0]))
            else:
                raise ValueError("symmetry plane not recognized")
    
    # check if all points have a symmetric partner not 2 the same
    for i in range(n_points):
        idx_symm_ps = np.where(np.array(sym_crit[i]) == 0)[0]
        if idx_symm_ps.shape[0] !=1 and idx_symm_ps.shape[0] != n_points:
            for idx in idx_symm_ps:
                idx_corr_sym_points = np.where(np.array(sym_crit[idx]) == 0)[0]
                if len(idx_corr_sym_points) != len(idx_symm_ps):
                    return False

    
    check = (np.sum(np.abs(np.min(np.array(sym_crit),axis=1))) <= 1e-5)
    
    return check

def check_wrench_in_admissible_set(admissible_set_vertices,wrench):
    check, alpha = check_point_in_convex_polytope(wrench, admissible_set_vertices)
    return check, alpha
    
def check_point_in_convex_polytope(point, polytope_vertices):
    check, alpha = solve_linear_combination_qp(point, polytope_vertices)
    return check, alpha

def solve_linear_combination_qp(u, C):
    """ Solves qp to determine whether a point can be expressed by a convex linear 
        combination of the vertices of a convex hull and therefor is part of the interior.
        
        The qp has the form:
        min 1/2 a^TC^TWCa + a^TC^TWu
            s.t. a^T1 = 1
                    a >= 0
        where C are the vertices of the convex hull and a are the coefficients of the linear combination.
    Args:
        u (numpy array (d,)): point to be checked
        C (numpy array (d,n)): vertices of the convex hull
    Returns:
        check (bool): True if the point is part of the interior of the convex hull
        alph (numpy array (n,)): coefficients of the linear combination
    """
    
    solvers.options['show_progress'] = False
    X = np.array(C).T
    n = X.shape[1]
    d = X.shape[0]
    W = np.eye(d)
    Q = matrix(X.T@W@X)
    p = matrix(-X.T@W@u)
    G = matrix(-np.eye(n))
    h = matrix(np.zeros((n)))
    A = matrix(np.ones((1,n)))
    b = matrix(np.ones((1)))

    sol = solvers.qp(Q, p, G, h, A, b)
    alph = np.array(sol['x'])
    cost = np.linalg.norm((X@alph).squeeze() - u)
    
    return cost < 1e-2, alph