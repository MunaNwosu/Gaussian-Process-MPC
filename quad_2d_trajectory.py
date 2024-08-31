from time import time
import casadi as ca
import numpy as np
from casadi import sin, cos, pi
import matplotlib.pyplot as plt
import pickle




# setting matrix_weights' variables
Q_x = 100
Q_xd = 0.1
Q_z = 80
Q_zd = 0.1
Q_theta = 0.01
Q_theta_d = 0.01

R1 = 0.05
R2 = 0.05


step_horizon = 0.05  # time between steps in seconds
N = 30            # number of look ahead steps

sim_time = 7      # simulation time
tau_theta_mpc = 0.4
k_theta_mpc = 1.4
tau_theta_step = 0.1
k_theta_step = 1


grav = 9.8

# initial specs
x_init = 0.0
x_d_init = 0.0
z_init = 0.0
z_d_init = 0.0
theta_init = 0.0
theta_d_init = 0.0

#target
x_target = 3.0
x_d_target = 0.0
z_target = 4.0
z_d_target = 0.0
theta_target = 0.0
theta_d_target = 0.0

#input bounds
theta_max = ca.pi/4
theta_min = -ca.pi/4
thrust_max = 30.0
thrust_min = 0.0


#Just MPC
def shift_timestep_mpc(step_horizon, t0, state_init, u, f, time_arr):
    f_value = f(state_init, u[:, 0])
    next_state = ca.DM.full(state_init + (step_horizon * f_value))
    t0 = t0 + step_horizon
    time_arr.append(t0)
    
    u0 = ca.horzcat(
        u[:, 1:],
        ca.reshape(u[:, -1], -1, 1)
    )
    
    return t0, next_state, u0, time_arr

#Added error
def shift_timestep(step_horizon, t0, state_init, u, f_step, f_mpc, time_arr):
    f_value = f_step(state_init, u[:, 0])
    st = state_init
    con = u[:,0]
    k1 = f_mpc(st, con)
    k2 = f_mpc(st + step_horizon/2*k1, con)
    k3 = f_mpc(st + step_horizon/2*k2, con)
    k4 = f_mpc(st + step_horizon * k3, con)
    st_next_RK4 = st + (step_horizon / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
    
    next_state = ca.DM.full(state_init + (step_horizon * f_value))
    state_error = next_state - st_next_RK4
    t0 = t0 + step_horizon
    time_arr.append(t0)
    
    
    u0 = ca.horzcat(
        u[:, 1:],
        ca.reshape(u[:, -1], -1, 1)
    )

    return t0, next_state, u0, time_arr,state_error

def DM2Arr(dm):
    return np.array(dm.full())

def RMSE(x_ref,z_ref,x_init_arr,z_init_arr):  
    #mse  = (np.square(np.subtract(x_init_arr,x_ref)) + \
    #        np.square(np.subtract(z_init_arr,z_ref))).mean()
    #rmse1 = math.sqrt(mse)
    
    rmse = np.sqrt(np.mean(((x_init_arr-x_ref)**2)+(z_init_arr-z_ref)**2))
    return rmse

# state symbolic variables
x = ca.SX.sym('x')
x_d = ca.SX.sym('x_d')
z = ca.SX.sym('z')
z_d = ca.SX.sym('z_d')
theta = ca.SX.sym('theta')
theta_d =ca.SX.sym('theta_d')
states = ca.vertcat(
    x,
    z,
    x_d,
    z_d,
    theta,
    theta_d
)
n_states = states.numel()

# control symbolic variables
theta_c = ca.SX.sym('theta_c')
thrust_c = ca.SX.sym('thrust_c')

controls = ca.vertcat(
    theta_c,
    thrust_c
)
n_controls = controls.numel()

# matrix containing all states over all time steps +1 (each column is a state vector)
X = ca.SX.sym('X', n_states, N + 1)

# matrix containing all control actions over all time steps (each column is an action vector)
U = ca.SX.sym('U', n_controls, N)

# coloumn vector for storing initial state, x and u ref for all predicted Horizons
P = ca.SX.sym('P', n_states + (N+1)*(n_states))

# state weights matrix
Q = ca.diagcat(Q_x, Q_z, Q_xd,  Q_zd, Q_theta, Q_theta_d)

# controls weights matrix
R = ca.diagcat(R1,R2)

RHS_MPC = ca.vertcat(x_d, z_d, thrust_c*ca.sin(theta),thrust_c*ca.cos(theta)-grav,
                 theta_d, 1/tau_theta_mpc*(k_theta_mpc*theta_c-theta))

RHS_STEP = ca.vertcat(x_d, z_d, thrust_c*ca.sin(theta),thrust_c*ca.cos(theta)-grav,
                 theta_d, 1/tau_theta_step*(k_theta_step*theta_c-theta))
# maps controls from inputs to outputs
f_mpc = ca.Function('f_mpc', [states, controls], [RHS_MPC])
f_step = ca.Function('f_step', [states, controls], [RHS_STEP])


cost_fn = 0  # cost function
g = X[:, 0] - P[:n_states]  # constraints in the equation
ref_states = n_states


numruns =  0

try:
    numruns = pickle.load(open("numruns.pickle", "rb"))
    numruns += 1
    pickle.dump(numruns, open("numruns.pickle", "wb"))
except (IOError) as e:
    numruns = 0
    pickle.dump(numruns, open("numruns.pickle", "wb"))

"""path = './numruns.txt'
check_file = os.path.isfile(path)"""

if numruns > 0:
    print("\nnumruns:", numruns)
    Bd = np.array([0,0,0,0,0,1]) # unit vector for extracting only gaussian av error prediction
    import gaus_mpc_2d_copy as gaus

else:
    print("numruns:", numruns)
    print("\nFirst run\n")
    
for k in range(N):
 # calculate costarray_totorch
    st = X[:, k]
    con = U[:, k]
    cost_fn = cost_fn \
        + (st - P[ref_states*k+6:ref_states*k+12]).T @ Q @ (st - P[ref_states*k+6:ref_states*k+12]) \
        + con.T @ R @ con
    st_next = X[:, k+1]
 # runge kutta   
    k1 = f_mpc(st, con)
    k2 = f_mpc(st + step_horizon/2*k1, con)
    k3 = f_mpc(st + step_horizon/2*k2, con)
    k4 = f_mpc(st + step_horizon * k3, con)
    st_next_RK4 = st + (step_horizon / 6) * (k1 + 2 * k2 + 2 * k3 + k4) 
    if numruns > 0:
        z = ca.vertcat(st,con)
        g = ca.vertcat(g, st_next - (st_next_RK4 + Bd @ gaus.casadi_predict(z=z.T)['mean']))
    else:
        g = ca.vertcat(g, st_next - st_next_RK4)
    
cost_array = []

OPT_variables = ca.vertcat(
    X.reshape((-1, 1)),   # Example: 3x11 ---> 33x1 where 3=states, 11=N+1
    U.reshape((-1, 1))
)
nlp_prob = {
    'f': cost_fn,
    'x': OPT_variables,
    'g': g,
    'p': P
}

opts = {
    'ipopt': {
        'max_iter': 2000,
        'print_level': 0,
        'acceptable_tol': 1e-8,
        'acceptable_obj_change_tol': 1e-6
    },
    'print_time': 0
}

solver = ca.nlpsol('solver', 'ipopt', nlp_prob, opts)


lbx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1))
ubx = ca.DM.zeros((n_states*(N+1) + n_controls*N, 1))

lbx[0: n_states*(N+1): n_states] = -ca.inf     # x lower bound
lbx[1: n_states*(N+1): n_states] = -ca.inf     # x_d lower bound
lbx[2: n_states*(N+1): n_states] = -ca.inf     # z lower bound
lbx[3: n_states*(N+1): n_states] = -ca.inf     # z_d lower bound
lbx[4: n_states*(N+1): n_states] = -ca.pi/4    # theta lower bound
lbx[5: n_states*(N+1): n_states] = -ca.inf     # theta_d lower bound

ubx[0: n_states*(N+1): n_states] = ca.inf      # x upper bound
ubx[1: n_states*(N+1): n_states] = ca.inf      # x_d upper bound
ubx[2: n_states*(N+1): n_states] = ca.inf      # z lower bound
ubx[3: n_states*(N+1): n_states] = ca.inf      # z_d lower bound
ubx[4: n_states*(N+1): n_states] = ca.pi/4     # theta upper bound
ubx[5: n_states*(N+1): n_states] = ca.inf      # theta_d lower bound



lbx[n_states*(N+1):: n_controls] = theta_min       # lower bound for theta
ubx[n_states*(N+1):: n_controls] = theta_max       # upper bound for theta
lbx[n_states*(N+1)+1:: n_controls] = thrust_min    # lower bound for thrust
ubx[n_states*(N+1)+1:: n_controls] = thrust_max    # upper bound for thrust

#print("lbx shape", ubx[n_states*(N+1)+1::n_controls].shape)
#print("thrust max", ubx[n_states*(N+1)+1::n_controls])

args = {
    'lbg': ca.DM.zeros((n_states*(N+1), 1)),  # constraints lower bound
    'ubg': ca.DM.zeros((n_states*(N+1), 1)),  # constraints upper bound 
    'lbx': lbx,
    'ubx': ubx,
}


t0 = 0
time_arr = [t0]
time_arr_first = [t0]

state_init = ca.DM([x_init, z_init, x_d_init,  z_d_init, theta_init,theta_d_init])        # initial state
state_target = ca.DM([x_target, z_target, x_d_target,  z_d_target, theta_target, theta_d_target])  # target state
state_init_arr  = ca.DM([x_init, z_init, x_d_init,  z_d_init, theta_init, theta_d_init])
state_error_arr = state_init_arr

state_init_first = ca.DM([x_init, z_init, x_d_init,  z_d_init, theta_init,theta_d_init]) 

# xx = DM(state_init)
t = ca.DM(t0)

u0 = ca.DM.zeros((n_controls, N))  # initial control
X0 = ca.repmat(state_init, 1, N+1) # initial state full

mpc_iter = 0
cat_states = DM2Arr(X0)
cat_controls = DM2Arr(u0[:, 0])
times = np.array([[0]])
t1 = []
input_array = cat_controls
var_freq = (2*pi)/10
###############################################################################

if __name__ == '__main__':
    main_loop = time()  # return time in sec
    state_ref = np.zeros([6,int(sim_time/step_horizon)+1])
    while (mpc_iter * step_horizon < sim_time):

        t1 = time()
        
        """
        args['p'] = ca.vertcat(
            state_init,    # current state
            state_target   # target state
        )
        """
        args['p'] = state_init
        """
        if mpc_iter * step_horizon < 2:
            var_freq = (2*pi)/8
        elif mpc_iter * step_horizon < 4:
            var_freq = (2*pi)/7
        elif mpc_iter * step_horizon < 6:
            var_freq = (2*pi)/6
        """ 
        for k in range(N+1):
            t_predict = t0 + (k-1)*step_horizon
            x_target = 0.02*cos(4*t_predict)
            z_target = 0.09*sin(4*t_predict)
            
            state_target = np.array([x_target, z_target, 0, 0, 0, 0])
        
            #P[ref_states*k+6:ref_states*k+12] = state_target
            args['p'] = ca.vertcat(
                args['p'],
                state_target   # target state
            )
            
        t_predict = t0
        x_target = 0.02*cos(4*t_predict)
        z_target = 0.09*sin(4*t_predict)
        state_target = np.array([x_target, z_target, 0, 0, 0, 0])
        state_ref[:,mpc_iter+1] = state_target
        
        #args['p'] = P
         
        # optimization variable current state
        args['x0'] = ca.vertcat(
            ca.reshape(X0, n_states*(N+1), 1),

            ca.reshape(u0, n_controls*N, 1)
        )

        sol = solver(
            x0=args['x0'],
            lbx=args['lbx'],
            ubx=args['ubx'],
            lbg=args['lbg'],
            ubg=args['ubg'],
            p=args['p']
        )
        

        cost_array.append(DM2Arr(sol['f']).squeeze())
        
        u = ca.reshape(sol['x'][n_states * (N + 1):], n_controls, N)
        X0 = ca.reshape(sol['x'][: n_states * (N+1)], n_states, N+1)
       
        cat_states = np.dstack((
            cat_states,
            DM2Arr(X0)
        ))

        cat_controls = np.vstack((
            cat_controls,
            DM2Arr(u[:, 0])
        ))
        t = np.vstack((
            t,
            t0
        ))
    
         
        if numruns < 1:
            t0, state_init, u0,time_arr,state_error  = shift_timestep(step_horizon, t0, state_init, u, f_step, f_mpc, time_arr)
            state_init_arr = ca.vertcat(state_init_arr,state_init)
            state_error_arr = ca.vertcat(state_error_arr,state_error)
        else:
            t0, state_init,u0,time_arr = shift_timestep_mpc(step_horizon, t0, state_init, u, f_mpc, time_arr)
            state_init_arr = ca.vertcat(state_init_arr,state_init)
        
        input_array = ca.vertcat(input_array,u0[:,0])
        
        X0 = ca.horzcat(
            X0[:, 1:],
            ca.reshape(X0[:, -1], -1, 1)
        )
        

        t2 = time()
        print("mpc iter: ",mpc_iter)
        print(t2-t1)

        times = np.vstack((
            times,
            t2-t1
        ))

        mpc_iter = mpc_iter + 1
        
    main_loop_time = time()
    
    rmse = RMSE(state_ref[0],state_ref[1],state_init_arr[0::6],state_init_arr[1::6])
    
    print("RMSE: ", rmse)
    print('\n\n')
    print('Total time: ', main_loop_time - main_loop)
    print('avg iteration time: ', np.array(times).mean() * 1000, 'ms')

    #print("cat shape: ",cat_controls.shape)
    
    state_path = 'state_init_arr.pickle'
    control_path = 'cat_controls.pickle'
    error_path =  'state_error_arr.pickle'
    if numruns == 0:
    #1 : write the controls states and error to pickle
        with open(state_path, 'wb') as state_file:
            pickle.dump(state_init_arr, state_file)
        with open(control_path, 'wb') as control_file:
            pickle.dump(cat_controls, control_file)
        with open(error_path, 'wb') as error_file:
            pickle.dump(state_error_arr, error_file)
        print("All the data has been loaded")
    
        """                          First Run                        """
    #loading pickle variables
    load_state_data = None
    load_control_data = None
    load_error_data = None  
    
    # Open the file in binary mode
    with open(state_path, 'rb') as state_file:
        load_state_data  = pickle.load(state_file)
    with open(control_path, 'rb') as control_file:
        load_control_data = pickle.load(control_file)
    with open(error_path, 'rb') as error_file:
        load_error_data =  pickle.load(error_file) 
    print("All the data has been Saved")


    fig,ax = plt.subplots(4,2)
    fig2,ax2 = plt.subplots()
    fig3,ax3 = plt.subplots()
    
    ax[0,0].set_xlabel('Time step(s)')
    ax[0,0].set_ylabel('Theta control(radian)')
    ax[0,0].plot(load_control_data[0::2], label= 'Input', color = 'r')
    ax[0,0].legend(loc='upper right',bbox_to_anchor=(1,0.5))
    
    ax[0,1].set_xlabel('Time step(s)')
    ax[0,1].set_ylabel('Input Thrust(N)')
    ax[0,1].plot(load_control_data[1::2],label= 'Input', color = 'r')
    ax[0,1].legend(loc='upper right',bbox_to_anchor=(1,0.5))
    
    ax[1,0].plot(time_arr, load_state_data[0::6], label = 'state', color ='b')
    ax[1,0].set_xlabel('Time(s)')
    ax[1,0].set_ylabel('X Position(m)')
    ax[1,0].plot(time_arr, state_ref[0], label = 'Reference',color = 'g')
    ax[1,0].legend(loc='upper right')
    
    ax[1,1].plot(time_arr, load_state_data[1::6],label = 'State', color ='b')
    ax[1,1].set_xlabel('Time(s)')
    ax[1,1].set_ylabel('Z position(m)')
    ax[1,1].plot(time_arr, state_ref[1],label = 'Reference',color = 'g')
    ax[1,1].legend(loc='upper right')

    #x and z
    ax2.plot(load_state_data[0::6],load_state_data[1::6])
    #ax2.plot(load_state_data[0],load_state_data[1],label ='Starting position','og')
    #ax2.plot(load_state_data[-1],load_state_data[-2],label ='End position','or')
    ax2.set_xlabel('X position')
    ax2.set_ylabel('Z position')
    fig2.suptitle("MPC Quadrotor Trajectory")
    
    #reference trajectory
    ax2.plot(state_ref[0],state_ref[1])
    #ax3.set_xlabel('X position')
    #ax3.set_ylabel('Z position')
    fig3.suptitle("Quadrotor Reference Trajectory")

    ax[2,0].plot(time_arr, load_state_data[2::6],label = 'State', color ='b')
    ax[2,0].set_ylabel('X velocity(m/s)')
    ax[2,0].legend(loc='upper right')

    ax[2,1].plot(time_arr, load_state_data[3::6],label = 'State', color ='b')
    ax[2,1].set_xlabel('Time(s)')
    ax[2,1].set_ylabel('Z velocity(m/s)')
    ax[2,1].legend(loc='upper right')

    ax[3,0].plot(time_arr, load_state_data[4::6],label = 'State', color ='b')
    ax[3,0].set_xlabel('Time(s)')
    ax[3,0].set_ylabel('Theta')
    ax[3,0].legend(loc='upper right')

    ax[3,1].plot(time_arr, load_state_data[5::6],label = 'State', color ='b')
    ax[3,1].set_xlabel('Time(s)')
    ax[3,1].set_ylabel('Angular velocity(rad/sec)')
    ax[3,1].legend(loc='upper right')
    
    fig.subplots_adjust(wspace=0.5,hspace=0.5)
    fig.suptitle("MPC")

    
    """                          After learned GP error                        """
    if numruns > 0:
        fig,ax= plt.subplots(4,2)
        fig2,ax2 = plt.subplots()
        
        ax[0,0].set_xlabel('Time step(s)')
        ax[0,0].set_ylabel('Theta control(radian)')
        ax[0,0].plot(cat_controls[0::2],label= 'Input', color = 'r')
        ax[0,0].legend(loc='upper right', bbox_to_anchor=(1,0.5))
        
        ax[0,1].set_xlabel('Time step(s)')
        ax[0,1].set_ylabel('Input Thrust(N)')
        ax[0,1].plot(cat_controls[1::2],label= 'Input', color = 'r')
        ax[0,1].legend(loc='upper right', bbox_to_anchor=(1,0.5))
        
        ax[1,0].plot(time_arr, state_init_arr[0::6], label = 'state', color ='b')
        ax[1,0].set_xlabel('Time(s)')
        ax[1,0].set_ylabel('X Position(m)')
        ax[1,0].plot(time_arr,state_ref[0], label = 'reference',color = 'g')
        ax[1,0].legend(loc='upper right')

        ax[1,1].plot(time_arr , state_init_arr[1::6], label = 'State', color ='b')
        ax[1,1].set_xlabel('Time(s)')
        ax[1,1].set_ylabel('Z position(m)')
        ax[1,1].plot(time_arr,state_ref[1],label = 'reference',color = 'g')
        ax[1,1].legend(loc='upper right')
        
        ax2.plot(state_init_arr[0::6],state_init_arr[1::6])
        ax2.plot(state_ref[0],state_ref[1]) # reference trejectory
        ax2.set_xlabel('X position')
        ax2.set_ylabel('Z position')
        fig2.suptitle("GPMPC Quadrotor Trajectory")
        
        ax[2,0].plot(time_arr , state_init_arr[2::6], label = 'State', color ='b')
        ax[2,0].set_xlabel('Time(s)')
        ax[2,0].set_ylabel('X velocity(m/s)')
        ax[2,0].legend(loc='upper right')


        ax[2,1].plot(time_arr , state_init_arr[3::6], label = 'State', color ='b')
        ax[2,1].set_xlabel('Time(s)')
        ax[2,1].set_ylabel('Z velocity(m/s)')
        ax[2,1].legend(loc='upper right')
        

        ax[3,0].plot(time_arr , state_init_arr[4::6], label = 'State', color ='b')
        ax[3,0].set_xlabel('Time(s)')
        ax[3,0].set_ylabel('Theta')
        ax[3,0].legend(loc='upper right')


        ax[3,1].plot(time_arr , state_init_arr[5::6], label = 'State', color ='b')
        ax[3,1].set_xlabel('Time(s)')
        ax[3,1].set_ylabel('Angular velocity')
        ax[3,1].legend(loc='upper right')
        
        fig.subplots_adjust(wspace=0.5,hspace=0.5)
        fig.suptitle("GPMPC")

    # Current data
    state_path_curr = 'state_init_arr_curr.pickle'
    control_path_curr = 'cat_controls_curr.pickle'
    error_path_curr =  'state_error_curr.pickle'
    if numruns == 0:
    #1 : write the controls states and error to pickle
        with open(state_path_curr, 'wb') as state_file:
            pickle.dump(state_init_arr, state_file)
        with open(control_path_curr, 'wb') as control_file:
            pickle.dump(cat_controls, control_file)
        with open(error_path_curr, 'wb') as error_file:
            pickle.dump(state_error_arr, error_file)
        print("New data has been loaded")

    
    
    plt.show()