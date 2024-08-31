
import torch
import gpytorch
import casadi as ca
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import numpy as np
import pickle

"""
def open_files():
    
    with open('input_array.txt',"r") as f:
        input_array = f.read()
        input_array = input_array.replace("[", "")
        input_array = input_array.replace("]", "")
        input_array = input_array.replace(",", "")
        input_array = [float(i) for i in input_array.split(' ')] 
    with open('state_array.txt',"r") as f:
        state_init_step = f.read()
        state_init_step = state_init_step.replace("[", "")
        state_init_step = state_init_step.replace("]", "")
        state_init_step = state_init_step.replace(",", "")
        state_init_step = [float(i) for i in state_init_step.split(' ')] 
        
    with open('state_error.txt',"r") as f:
        state_error = f.read()
        state_error = state_error.replace("[", "")
        state_error = state_error.replace("]", "")
        state_error = state_error.replace(",", "")
        state_error = [float(i) for i in state_error.split(' ')] 
        
    return input_array, state_init_step, state_error
"""
def array_totorch(array_2numpy):
    numpy_array = np.array(array_2numpy)
    pytorch_tensor = torch.from_numpy(numpy_array)
    return pytorch_tensor.flatten()

def covSEard(x,
             z,
             ell,
             sf2
             ):
    #sf2: output_scale
    dist = ca.sum1((x - z)**2 / ell**2)
    return sf2 * ca.SX.exp(-.5 * dist)

def Matern5_2(x,z,ell,sf2):
    mat_func = ( 1 + ca.sum1(((5**0.5)*ca.SX.fabs(x-z))/ell) + 
     ca.sum1((5*(x-z)**2)/(3*(ell**2))) ) \
        * ca.sum1(ca.SX.exp((-(5**0.5)* ca.SX.fabs(x-z))/ell))
    return sf2*mat_func

       
def make_casadi_prediction_func(model, train_inputs, train_targets):
    '''Assumes train_inputs and train_targets are already masked.'''
    K_lazy = model.covar_module(train_inputs.double())
    K_lazy_plus_noise = K_lazy.add_diag(model.likelihood.noise)
    n_samples = train_inputs.shape[0]
    model.K_plus_noise = K_lazy_plus_noise.matmul(torch.eye(n_samples).double())
    model.K_plus_noise_inv = K_lazy_plus_noise.inv_matmul(torch.eye(n_samples).double())
    
    train_inputs = train_inputs.numpy()
    train_targets = train_targets.numpy()
    lengthscale = model.covar_module.base_kernel.lengthscale.detach().numpy()
    output_scale = model.covar_module.outputscale.detach().numpy()
    Nx = 8  # number of controls + states
    z = ca.SX.sym('z', Nx)
    K_z_ztrain = ca.Function('k_z_ztrain',
                                [z],
                                [Matern5_2(z, train_inputs.T, lengthscale.T,output_scale)],
                                ['z'],
                                ['K'])
    predict = ca.Function('pred',
                            [z],
                            [K_z_ztrain(z=z)['K'] @ model.K_plus_noise_inv.detach().numpy() @ train_targets],
                            ['z'],
                            ['mean'])
    return predict


state_path_curr = 'state_init_arr_curr.pickle'
control_path_curr = 'cat_controls_curr.pickle'
error_path_curr =  'state_error_curr.pickle'

load_state_data = None
load_control_data = None
load_error_data = None  

# Open the file in binary mode
with open(state_path_curr , 'rb') as state_file:
    load_state_data  = pickle.load(state_file)
with open(control_path_curr, 'rb') as control_file:
    load_control_data = pickle.load(control_file)
with open(error_path_curr, 'rb') as error_file:
    load_error_data =  pickle.load(error_file) 
print("All the data has been Saved") 

theta_input = array_totorch(load_control_data[0::2])
thrust_input = array_totorch(load_control_data[1::2])
x_position = array_totorch(load_state_data[0::6])
z_position = array_totorch(load_state_data[1::6])
x_velocity = array_totorch(load_state_data[2::6])
z_velocity = array_totorch(load_state_data[3::6])
theta = array_totorch(load_state_data[4::6])
angular_velocity = array_totorch(load_state_data[5::6])

x_position_error =  array_totorch(load_error_data[0::6])
z_position_error =  array_totorch(load_error_data[1::6])
x_velocity_error =  array_totorch(load_error_data[2::6])
z_velocity_error =  array_totorch(load_error_data[3::6])
theta_error =  array_totorch(load_error_data[4::6])
angular_velocity_error =  array_totorch(load_error_data[5::6])

"""
## add inputs
theta_input = input_state_errordiff[0]
thrust_input = input_state_errordiff[1]
x_position = input_state_errordiff[2]
z_position = input_state_errordiff[3]
x_velocity = input_state_errordiff[4]
z_velocity = input_state_errordiff[5]
theta = input_state_errordiff[6]
angular_velocity = input_state_errordiff[7]
x_position_error = input_state_errordiff[8]
z_position_error = input_state_errordiff[9]
x_velocity_error = input_state_errordiff[10]
z_velocity_error = input_state_errordiff[11]
theta_error = input_state_errordiff[12]
angular_velocity_error = input_state_errordiff[13]
"""
#splitting the test and training data for the states and errors
train_xp, test_xp, train_xp_error, test_xp_error = train_test_split(x_position,x_position_error, \
    test_size=0.2,random_state=42)
train_zp, test_zp, train_zp_error, test_zp_error = train_test_split(z_position,z_position_error, \
    test_size=0.2,random_state=42)
train_xv, test_xv, train_xv_error, test_xv_error = train_test_split(x_velocity,x_velocity_error, \
    test_size=0.2,random_state=42)
train_zv, test_zv, train_zv_error, test_zv_error = train_test_split(z_velocity,z_velocity_error, \
    test_size=0.2,random_state=42)
train_theta, test_theta, train_theta_error, test_theta_error = train_test_split(theta,theta_error, \
    test_size=0.2,random_state=42)
train_av, test_av, train_av_error, test_av_error = train_test_split(angular_velocity,angular_velocity_error, \
    test_size=0.2,random_state=42)

train_theta_input,test_theta_input = train_test_split(theta_input,test_size = 0.2, random_state = 42)
train_thrust_input,test_thrust_input = train_test_split(thrust_input,test_size = 0.2, random_state = 42)

training_data = torch.stack((train_xp,train_zp,train_xv,train_zv, train_theta,train_av,train_theta_input,train_thrust_input), dim=-1)
testing_data = torch.stack((test_xp,test_zp,test_xv,test_zv,test_theta,test_av,test_theta_input,test_thrust_input), dim=-1)

training_data = training_data + torch.randn_like(training_data)

class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ZeroMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(
        #    gpytorch.kernels.MaternKernel(nu=2.5, ard_num_dims=train_x.shape[1]),
        #    ard_num_dims=train_x.shape[1]
            gpytorch.kernels.RBFKernel(ard_num_dims=train_x.shape[1]),
            ard_num_dims=train_x.shape[1]
           )
        
    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        #print(mean_x,"\n")
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
     


# initialize likelihoods and models
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(training_data, train_av_error, likelihood)
#likelihood2 = gpytorch.likelihoods.GaussianLikelihood()
#model2 = ExactGPModel(training_data, train_theta_error, likelihood2)

"""
likelihood2 = gpytorch.likelihoods.GaussianLikelihood()
model2 = ExactGPModel(train_zp, train_zp_error, likelihood2)

likelihood3 = gpytorch.likelihoods.GaussianLikelihood()
model3 = ExactGPModel(train_xv,train_xv_error,likelihood3)

likelihood4 = gpytorch.likelihoods.GaussianLikelihood()
model4 = ExactGPModel(train_zv,train_zv_error,likelihood4)

likelihood5 = gpytorch.likelihoods.GaussianLikelihood()
model5 = ExactGPModel(train_theta,train_theta_error,likelihood5)

likelihood6 = gpytorch.likelihoods.GaussianLikelihood()
model6 = ExactGPModel(train_av,train_av_error,likelihood6)

"""

# this is for running the notebook in our testing framework
import os
smoke_test = ('CI' in os.environ)
training_iter = 2 if smoke_test else 200


# Find optimal model hyperparameters
model.train()
likelihood.train()
#model2.train()
#likelihood2.train()


# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters
#optimizer2 = torch.optim.Adam(model2.parameters(),lr=0.1)
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)
#mll2 = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood2,model2)

#Angular velocity error training
for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(training_data)
    # Calc loss and backprop gradients
    loss = -mll(output, train_av_error)
    loss.backward()
    
    print('Iter %d/%d - Loss: %.3f   lengthscale: %s   noise: %s' % (
        i + 1, training_iter, loss.item(),
        model.covar_module.base_kernel.lengthscale,
        model.likelihood.noise.item()
        ))
    optimizer.step()

"""
#Theta error Training 
for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer2.zero_grad()
    # Output from model
    output2 = model2(training_data)
    # Calc loss and backprop gradients
    loss2 = -mll2(output2, train_theta_error)
    loss2.backward()
    
    print('Iter %d/%d - Loss: %.3f   lengthscale: %s   noise: %s' % (
        i + 1, training_iter, loss2.item(),
        model2.covar_module.base_kernel.lengthscale,
        model2.likelihood.noise.item()
        ))

    optimizer2.step()    
"""   
    
# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()
#model2.eval()
#likelihood2.eval()


# Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(model(testing_data))
    #observed_pred2 = likelihood2(model2(testing_data))

with torch.no_grad():
    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    #lower2, upper2 = observed_pred2.confidence_region()
    
casadi_predict = make_casadi_prediction_func(model,training_data,train_av_error)
predict_av_error = casadi_predict(z=testing_data.numpy().T)['mean'].toarray()
print("\nPredict Error:",predict_av_error)


####### Angular Velocity error
    #surface of training points
fig = plt.figure(figsize= (20,30))
ax = fig.add_subplot(1,2,1,projection='3d')

ax.plot3D(train_xv.numpy(),train_zv.numpy(),train_av_error.numpy(),'b.',label ='Training Data') 

#test data
ax.plot3D(test_xv.numpy(),test_zv.numpy(), observed_pred.mean.numpy(), 'r.',label ='Observed Prediction')


##Plotting error bars
res = torch.stack((test_xv, test_zv, upper,lower), dim=1)
for i in np.arange(0, len(test_xv)-1):
    for temp_test_xv, temp_test_zv, temp_tupper, temp_lower in res :
        ax.plot([temp_test_xv, temp_test_xv], [temp_test_zv, temp_test_zv], [temp_tupper, temp_lower], marker="_", color='k')

plt.title("Angular/XZ velocity Error Prediction")
ax.set_xlabel('X Velocity')
ax.set_ylabel('Z Velocity')
ax.set_zlabel('AV error Prediction')
ax.zaxis.labelpad = 0.5


plt.legend()
plt.show()





