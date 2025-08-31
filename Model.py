import jax
import jax.numpy as jnp
from flax import linen as nn
from flax.training import train_state
from flax.core import FrozenDict
from typing import Callable, Sequence, Dict, Any, Tuple
import optax
import chex
from MLPNet import MLP
from Optimizers import get_optimizer
from OptimalTransport import network_grad_time
import numpy as np
import itertools
import torch
'''For more detailed and efficient code see the paper "Learning diffusion at lightspeed" by Terpin et al. https://arxiv.org/pdf/2406.12616'''
'''JKONet-Star Quadratic
JKOQuadratic (disclaimer this was not included in Learning Diffusion at Lightspeed) but the idea still comes from their loss function with
the quadratic potential 1/2*x.T@theta@x
So the grad_potential = (theta+theta.T)@x
We use torch instead of the preferred jax because jax's torchmin equivalent has less options than torchmin
'''
class JKOQuadratic:
    def __init__(self,data_dim,tau,config):
        self.data_dim=data_dim
        self.tau=tau
        self.config=config['energy']['quadratic']['args']
        self.reg = config['energy']['regulariser']
    
    #Enforce a matrix condition: we admit there could be a larger variety of matrix choices but we wanted to see how these general ones fared up

    def obtain_matrix(self,params):
        if 'Unrestricted' in self.config:
            mat=params[:self.data_dim**2].view(self.data_dim,self.data_dim)
        if 'Posdef' in self.config:
            U=params[:self.data_dim**2].view(self.data_dim,self.data_dim)
            mat=U@U.T
        if 'L1-pos' in self.config:
            U=params[:self.data_dim**2].view(self.data_dim,self.data_dim)
            mat=U*U
        return mat
    
    #Choose regularisation depending on the matrix

    def reg_term(self,params):
        if 'L2' in self.reg:
            L2param=self.reg['L2']
            if 'Unrestricted' in self.config:
                regterm=L2param*torch.sum(params**2)
            if 'Posdef' in self.config:
                regterm=2*L2param*torch.sum(params**2)
            if 'L1-pos' in self.config:
                regterm=2*L2param*torch.sum(params**2)
        if 'L1' in self.reg:
            L1param=self.reg['L1']
            regterm=L1param*torch.norm(torch.abs(params))
        return regterm

    #Loss and training step obtaining optimal parameter(s) theta

    def loss_term(self,matrix,xs,ys,t,ws):
        matrix=1/2*(matrix+matrix.T)
        grad_potential1=torch.einsum('ij,kj->ik',ys,matrix)
        return torch.sum(ws*torch.sum(((grad_potential1)+1/self.tau*(ys-xs))**2,axis=1))
    def train_step(self,theta,sample):
        xs,ys,t,ws=sample
        theta=theta.reshape(self.data_dim,self.data_dim)
        Lossterm=self.loss_term(theta,xs,ys,t,ws)
        regterm=self.reg_term(theta)
        TotLoss=Lossterm+regterm
        return TotLoss
    
    #Function is what is put into the torchmin minimise function then choose the optimizer in our case L-FBGS

    def Function(self,params,sample):
        theta=self.obtain_matrix(params)
        loss=self.train_step(theta,sample)
        return loss





'''JKONet-Star Quadratic Linear
JKOQuadratic Linear (disclaimer this was not included in Learning Diffusion at Lightspeed) but the idea still comes from their loss function with
the quadratic linear potential 1/2*x.T@theta@x+b.T@x
So the grad_potential = 1/2*(theta+theta.T)@x+b
'''
class JKOQuadraticLinear:
    def __init__(self,data_dim,tau,config):
        self.data_dim=data_dim
        self.tau=tau
        self.config=config['energy']['quadratic']['args']
        self.reg = config['energy']['regulariser']

    #Enforce a matrix condition: we admit there could be a larger variety of matrix choices but we wanted to see how these general ones fared up
    
    def obtain_matrix(self,params):
        if 'Unrestricted' in self.config:
            mat=params[:self.data_dim**2].view(self.data_dim,self.data_dim)
            bvec=params[self.data_dim**2:].view(self.data_dim)
        if 'Posdef' in self.config:
            U=params[:self.data_dim**2].view(self.data_dim,self.data_dim)
            bvec=params[self.data_dim**2:].view(self.data_dim)
            mat=U@U.T
        if 'L1-pos' in self.config:
            U=params[:self.data_dim**2].view(self.data_dim,self.data_dim)
            bvec=params[self.data_dim**2:].view(self.data_dim)
            mat=U*U
        return mat,bvec
    #Choose regularisation depending on the matrix
    
    def reg_term(self,params):
        if 'L2' in self.reg:
            L2param=self.reg['L2']
            if 'Unrestricted' in self.config:
                regterm=L2param*torch.sum(params**2)
            if 'Posdef' in self.config:
                regterm=2*L2param*torch.sum(params**2)
            if 'L1-pos' in self.config:
                regterm=2*L2param*torch.sum(params**2)
        if 'L1' in self.reg:
            L1param=self.reg['L1']
            regterm=L1param*torch.norm(torch.abs(params))
        return regterm

    #Loss and training step obtaining optimal parameter(s) theta and bvec

    def loss_term(self,matrix,bvec,xs,ys,t,ws):
        matrix=1/2*(matrix+matrix.T)
        grad_potential1=torch.einsum('ij,kj->ik',ys,matrix)
        return torch.sum(ws*torch.sum(((grad_potential1+bvec)+1/self.tau*(ys-xs))**2,axis=1))
    def train_step(self,theta,bvec,sample):
        xs,ys,t,ws=sample
        theta=theta.reshape(self.data_dim,self.data_dim)
        Lossterm=self.loss_term(theta,bvec,xs,ys,t,ws)
        regterm=self.reg_term(1/2*(theta+theta.T))
        TotLoss=Lossterm+regterm
        return TotLoss
    
    #Function is what is put into the torchmin minimise function then choose the optimizer in our case L-FBGS

    def Function(self,params,sample):
        theta,bvec=self.obtain_matrix(params)
        loss=self.train_step(theta,bvec,sample)
        return loss



'''JKONet-Star Linear Parameterisation
JKOLinearModified gives back the solution to the linear system of the loss function when we have potential modelled by the linear parameter
V=theta.T phi(x)
Where phi(x) is the feature functions. Original code from https://arxiv.org/pdf/2406.12616 by Terpin et al.
''' 
def l2loss(reg,params):
    return reg*sum([jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(params)])
class JKOLinearModified:
    def __init__(self,config,data_dim,tau):
        super().__init__()
        self.tau = tau
        self.data_dim = data_dim
        self.config = config['energy']['linear']['features']
        self.reg = config['energy']['linear']['reg']
        self.fns=[]
# Define the feature choices: note that Learning diffusion at lightspeed had more rbfs than we do.
# We note that this method while fine in lower dimensions becomes problematic in higher due to the rbfs requiring that the domain be [-c,c]^d 
# where d is the number of features and c is the constant chosen for the domain. So the number of features must be increased for higher dimensions
# This unfortunately causes issues in the higher dimensions

        if 'polynomials' in self.config:
            exps = [a for a in itertools.product(range(self.config['polynomials']['degree']+1),repeat=self.data_dim) if sum(a)>0]
            for a in exps:
                self.fns.append(self.create_polynomial_fn(a))
                if self.config['polynomials']['sines']:
                    self.fns.append(self.create_sine_fn(a))
                if self.config['polynomials']['cosines']:
                    self.fns.append(self.create_cosine_fn(a))
        if 'rbfs' in self.config:
            domain=self.config['rbfs']['domain']
            n_centers=self.config['rbfs']['n_centers_per_dim']
            sigma=self.config['rbfs']['sigma']
            centers=[jnp.asarray(c) for c in itertools.product(np.linspace(domain[0],domain[1],n_centers),repeat=self.data_dim)]
            for c in centers:
                self.fns.append(self.create_rbf_fn(c,sigma))
        
        _features_grad = jax.vmap(self.features_grad)
        self.yt1 = _features_grad
        self.theta_dim = self.features_dim 

    def create_polynomial_fn(self,exp):
        expa=jnp.array(exp)
        def fn(x):
            return jnp.prod(x**expa,axis=-1)
        return fn
    def create_sine_fn(self,exp):
        expa=jnp.array(exp)
        def fn(x):
            return jnp.prod(jnp.sin(x**expa),axis=-1)
        return fn
    def create_cosine_fn(self,exp):
        expa=jnp.array(exp)
        def fn(x):
            return jnp.prod(jnp.cos(x**expa),axis=-1)
        return fn
    def create_rbf_fn(self,center,sigma):
        def fn(x):
            return jnp.exp((-jnp.sum((x-center)**2,axis=-1))/sigma)
        return fn
    def features(self, x):
        return jnp.asarray([f(x) for f in self.fns])
    def features_grad(self,x):
        return jnp.stack([jax.grad(f)(x) for f in self.fns],axis=1)
    @property
    def features_dim(self):
        if not hasattr(self, '_features_dim_cache'):
            self._features_dim_cache = self.features(jnp.ones((self.data_dim,))).shape[0]
        return self._features_dim_cache
    def loss(self,theta,xs,ys,ws):
        grad_potential=self.yt1(ys)
        grad_potential=jnp.sum(theta*grad_potential,axis=1)
        return jnp.sum(ws*jnp.sum((self.tau*grad_potential+ys-xs)**2,axis=1))

    def get_potential(self,theta):
        return lambda x: jnp.sum(theta*self.features(x))
    def get_grad_potential(self,theta):
        return lambda x: jnp.sum(theta*self.features_grad(x),axis=1)


'''JKONet-Star Neural Network potential
Just_the_Pot_time is the model which learns the potential from an MLP of shape [64,64,1] and then predicts using the time-implicit Euler scheme
'''
# def l1loss(reg,params):
#     u=jax.random.normal(jax.random.key(42),params.shape)
#     v=params/u
#     return reg*1/2*(jnp.sum(u)**2+jnp.sum(v)**2)
def l2loss(reg,params):
    return reg*sum([jnp.sum(jnp.square(x)) for x in jax.tree_util.tree_leaves(params)])
class Just_the_Pot_time:
    def __init__(self,config,data_dim,tau):
        self.tau = tau
        self.data_dim = data_dim
        self.config = config
        self.config_optimizer = config['energy']['optim']
        self.layers = config['energy']['model']['layers']
        self.reg=config['energy']['regulariser']['l2reg']
        self.model_potential = MLP(self.layers)
    def create_state(self,rng):
        optimizer = get_optimizer(self.config_optimizer)
        def cts(rng,model,optimizer,input_shape):
            params = model.init(rng, jnp.ones((1, input_shape)))['params']
            return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
        return cts(rng,self.model_potential,optimizer,self.data_dim+1)
    def create_state_from_params(self,potential_params):
        optimizer = get_optimizer(self.config_optimizer)
        def ctsp(model,params,optimizer):
            return train_state.TrainState.create(apply_fn=model.apply, params=params, tx=optimizer)
        return ctsp(self.model_potential,potential_params,optimizer)
    def loss_potential_term(self,params,xt1s,t):
        grad_fn = network_grad_time(self.model_potential,params)
        xt1=jnp.concatenate((xt1s,t[:,None]),axis=1)
        return grad_fn(xt1)
    def loss(self,potential_params,xts,xt1s,t,wts):
        grad_potential=self.loss_potential_term(potential_params,xt1s,t)
        L2error=l2loss(self.reg,potential_params)
        return jnp.sum(wts*jnp.sum((self.tau*grad_potential+xt1s-xts)**2,axis=1))+self.tau**2*(L2error)
    def within_train_step(self,state,xts,xt1s,t,wts):
        loss,grads=jax.value_and_grad(self.loss,argnums=0)(state.params,xts,xt1s,t,wts)
        state=state.apply_gradients(grads=grads)
        return loss, state
    def train_step(self,state,sample):
        xts,xt1s,t,wts = sample
        return self.within_train_step(state,xts,xt1s,t,wts)
    def get_params(self, state):
        return state.params
    def get_potential(self,state):
        return lambda x: state.apply_fn({'params':state.params},x)
