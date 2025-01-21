import numpy as np
import torch
from pytorch3d.transforms import euler_angles_to_matrix
from scipy.spatial.transform import Rotation as Rot
from aerial_gym.utils.analyze_robot_config import calc_convex_hull_admissible_set, analyze_robot_config, check_flyability



class RobotModel:
    def __init__(self, pars):
        """ Initializes a robot model with the given parameters and gives an analyses 
            of the robot configuration."""
            
        self.cfg_name = pars.config_name
        
        # TODO: replace gravity constant with actual gravity from simulation
        self.g = torch.tensor([0,0,9.81], dtype=torch.float32) #device="cuda"
        self.cq = pars.cq
        
        self.frame_mass = pars.frame_mass
        self.motor_masses = pars.motor_masses
        self.sensor_masses = pars.sensor_masses
        
        self.min_u = pars.min_u 
        self.max_u = pars.max_u 
        
        # translations and orientations in root link frame
        self.motor_orientations = np.array(self.convert_orientations(pars.motor_orientations))
        self.motor_translations = np.array(pars.motor_translations)
        self.n_motors = len(self.motor_orientations)
        self.motor_directions = np.array(pars.motor_directions)
        
        self.sensor_translations = np.array(pars.sensor_translations)
        self.sensor_orientations = np.array(self.convert_orientations(pars.sensor_orientations))
        self.n_sensors = len(self.sensor_masses)
        
        n_arms = self.n_motors + self.n_sensors
        self.motor_mask = [1 + n_arms + i for i in range(0, self.n_motors)]
        
        self.total_mass = None
        self.calc_total_mass()
        self.r_com = None
        self.calc_center_of_mass()
        
        # translations and orientations in center of mass frame
        self.motor_translations_com = self.motor_translations - self.r_com
        self.motor_orientations_com = self.motor_orientations
        if self.n_sensors > 0:
            self.sensor_translations_com = self.sensor_translations - self.r_com
            self.sensor_orientations_com = self.sensor_orientations
        else:
            self.sensor_translations_com = []
            self.sensor_orientations_com = []
        
        self.inertia_matrix = None
        self.calc_inertia_matrix()
        
        self.actuator_mixing_matrix = None
        self.calc_actuator_mixing_matrix()
        
        # calculate the vertices of the admissible wrench set
        self.convex_hull_admissible_set_vertices = calc_convex_hull_admissible_set(self.min_u, 
                                                                                   self.max_u, 
                                                                                   self.actuator_mixing_matrix)
        
        self.rank_act_mix_matrix = np.linalg.matrix_rank(self.actuator_mixing_matrix)
        self.pinv_allocation_matrix = np.linalg.pinv(self.actuator_mixing_matrix)
        
    def check_flyability(self):
        """ Check if the robot is flyable."""
        return check_flyability(self)
        
    def analyze_config(self):
        """ Analyze the robot configuration and print the results."""
        analyze_robot_config(self)
    
    def convert_orientations(self,orientations):
        """ Converts orientation representations to rotation matrices.
        Args:
            orientations (list): orientations represented as rotation matrices, 
                                 quaternions (x,y,z,w) or euler angles in rdeg (x,y,z)
        Raises:
            ValueError: orientation representation is not known
        Returns:
            orientations_rot_mat (list): orientations represented as rotation matrices
        """

        if np.shape(orientations) != (0,):
            if np.shape(orientations[0]) == (3,3):
                orientations_rot_mat = orientations
            elif np.shape(orientations[0]) == (4,):
                orientations_rot_mat = [Rot.from_quat(orient_quat).
                                        as_matrix() for orient_quat in orientations]
            elif np.shape(orientations[0]) == (3,):
                orientations_rot_mat = [Rot.from_euler("xyz",orient_euler, degrees = True).
                                        as_matrix() for orient_euler in orientations]
            else:
                raise ValueError(orientations[0],"orientation representation not known")
            
            return orientations_rot_mat
        
    def calc_total_mass(self):
        self.total_mass = np.sum(self.motor_masses) + np.sum(self.sensor_masses) + self.frame_mass
    
    def calc_center_of_mass(self):
        """ Calculates the center of mass of the robot."""
        self.r_com = np.zeros(3)
        for m,r in zip(self.motor_masses,self.motor_translations):
            self.r_com += (m*np.array(r))/self.total_mass
        if self.n_sensors > 0:
            for s,r in zip(self.sensor_masses,self.sensor_translations):
                self.r_com += (s*np.array(r))/self.total_mass 
    
    def calc_inertia_matrix(self):
        """ calculate inertia matrix with respect to the center of gravity """
        
        self.inertia_matrix = np.zeros((3,3))
        # influence of motors on inertia
        for i in range(self.n_motors):
            tm_i = self.motor_translations_com[i]
            mm_i = self.motor_masses[i]
            self.inertia_matrix += mm_i*(tm_i.T@tm_i*np.eye(3) - np.outer(tm_i,tm_i))
        # influence of sensors on inertia
        if self.n_sensors > 0:
            for i in range(self.n_sensors):
                ts_i = self.sensor_translations_com[i]
                ms_i = self.sensor_masses[i]
                self.inertia_matrix += ms_i*(ts_i.T@ts_i*np.eye(3) - np.outer(ts_i,ts_i))
        self.inertia_matrix += self.frame_mass*((-self.r_com).T@(-self.r_com)*np.eye(3) - np.outer(-self.r_com,-self.r_com))
        
    def calc_actuator_mixing_matrix(self):
        """ calculate actuator mixing matrix with respect to the center of gravity """
        self.actuator_mixing_matrix = np.zeros((6,self.n_motors))
        for i in range(self.n_motors):
            vec_z_mf = np.array([0,0,1]) # motor frame z axis
            R_i = self.motor_orientations_com[i]
            t_i = self.motor_translations_com[i]
            vec_z_bf = R_i @ vec_z_mf # motor axis in body frame
            
            alpha_i = self.motor_directions[i]
            
            # forces
            self.actuator_mixing_matrix[:3,i] = vec_z_bf
            # torques
            self.actuator_mixing_matrix[3:,i] = np.cross(t_i, vec_z_bf) - alpha_i*self.cq * vec_z_bf
        
    def prepare_torch(self,device):
        self.motor_directions = torch.tensor(self.motor_directions,dtype=torch.float32,device = device)
        self.actuator_mixing_matrix = torch.tensor(self.actuator_mixing_matrix,
                                                   dtype=torch.float32,
                                                   device = device)
        self.inertia_matrix = torch.tensor(self.inertia_matrix, dtype=torch.float32,device = device)
        self.inv_inertia_matrix = torch.linalg.inv(self.inertia_matrix)
        self.pinv_allocation_matrix = torch.tensor(self.pinv_allocation_matrix, 
                                                   dtype=torch.float32,
                                                   device = device)
        self.g = torch.tensor(self.g, dtype=torch.float32,device = device)
        self.r_com = torch.tensor(self.r_com, dtype=torch.float32,device = device)
    
    def step(self, state , motor_thrust, dt):
        """ Performs a nonlinear step of the system dynamics (euler angles as orientation representation)
            with explicit euler integration for a single system.
        Args:
            state (torch tensor (12)): robot state 
            motor_thrust (torch tensor (#motors)): motor thrusts
            dt (torch tensor (1)): time step size
        Returns:
            state_next (torch tensor (12)): step at time step t+dt
        """
        
        pos = state[:3]
        theta = state[3:6]
        vel = state[6:9]
        omega = state[9:]
        
        J = self.inertia_matrix
        J_inv = self.inv_inertia_matrix
        F = self.actuator_mixing_matrix
        
        eta = F@motor_thrust
        f_u = eta[:3]
        tau_u = eta[3:]

        R = euler_angles_to_matrix(theta, "XYZ")
        acc = -self.g + R@f_u/self.total_mass
        
        ang_acc = J_inv @ (tau_u - torch.cross(omega, J @ omega))
        
        pos_next = pos + vel * dt
        theta_next = theta + omega * dt
        vel_next = vel + acc * dt
        omega_next = omega + ang_acc * dt

        #state_next = torch.round(torch.cat((pos_next, theta_next, vel_next, omega_next)), decimals=7)
        state_next = torch.cat((pos_next, theta_next, vel_next, omega_next))
        
        return state_next
        
    def linearize_system(self, step_name = "step_eulAng"):
        """ Returns a function that takes in the same inputs as the defined step function 
            and returns the Jacobian of the step function with respect to the robot state 
            and inputs (system and input matrix of the linearized system).
        Args:
            step_name (str, optional): Name of the step function which should be used. 
                                       Defaults to "step_eulAng".
        Returns:
            A (func): Jacobian of the step function with respect to the robot state
            B (func): Jacobian of the step function with respect to the input
        """
        
        if step_name == "step_eulAng":
            A = torch.func.jacfwd(self.step_eulAng, 0)
            B = torch.func.jacfwd(self.step_eulAng, 1)
        else:
            raise ValueError(step_name,"this step function is not implemented")
            
        return A, B
        
class RobotParameter:
    """ Robot parameters needed to define a robot model."""
    def __init__(self):
        self.config_name = None # name of the robot configuration
        self.cq = None # torque constant [1/m]
        self.frame_mass = None # mass of the frame [kg]  
        self.motor_masses = None # masses of the motors [kg]
        self.motor_orientations = None # orientations of the motors [Rotation matrizes, quaternions (x,y,z,w) or euler angles in deg (x,y,z) ]
        self.motor_translations = None # translations of the motors [m]
        self.motor_directions = None 
        self.sensor_masses = []
        self.sensor_orientations = [] # orientations of the motors [Rotation matrizes, quaternions (x,y,z,w) or euler angles in deg (x,y,z) ]
        self.sensor_translations = [] # translations of the motors [m]
        self.max_u = None # maximum control input [N]
        self.min_u = None # minimum control input [N]