import jax
import jax.numpy as jnp
import ot
from scipy.special import logsumexp
import numpy as np
import ott
from ott.geometry import pointcloud
from ott.problems.linear import linear_problem
from ott.solvers.linear import sinkhorn
import torch
from torch.utils.data import DataLoader

def minr(H,reg,r):
    return reg*np.log(r)-reg*logsumexp(H/reg,1)
def minc(H,reg,c):
    return reg*np.log(c)-reg*logsumexp(H/reg,0)
def SINK5(C,r,c,reg,tol,M):
    f=np.zeros_like(r)
    g=np.zeros_like(c)
    Error=np.zeros(M)
    for i in range(M):
        fold=f.copy()
        gold=g.copy()
        f=minr(g[None,:]-C,reg,r)
        g=minc(f[:,None]-C,reg,c)
        if np.linalg.norm(fold-f)<tol and np.linalg.norm(gold-g)<tol:
            break
        P=np.exp((f[:,None]+g[None,:]-C)/reg)
        Error[i]=np.linalg.norm(np.sum(P,0)-c,1)+np.linalg.norm(np.sum(P,1)-r,1)
    min_cost=np.sum(P*C)
    return P,min_cost,Error

def SINK1(C,r,c,reg,tol,M):
    K=np.exp(-C/np.max(C)/reg)
    u=np.ones_like(r)
    v=np.ones_like(c)
    Error=np.zeros(M)
    for i in range(M):
        uold=u.copy()
        vold=v.copy()
        u=r/(K@v)
        v=c/(K.T@u)
        if np.linalg.norm(uold-u)<tol and np.linalg.norm(vold-v)<tol:
            break
        P=np.diag(u)@(K@np.diag(v))
        Error[i]=np.linalg.norm(np.sum(P,0)-c,1)+np.linalg.norm(np.sum(P,1)-r,1)
    min_cost=np.sum(P*C)
    return P,min_cost,Error
def BaryProj(TP,x,y):
    return np.dot(TP,y)/(np.ones_like(x)/x.shape[0])

def SINK6(C,r,c,reg,tol,M):
    f=np.zeros_like(r)
    g=np.zeros_like(c)
    Error=np.zeros(M)
    for i in range(M):
        fold=f.copy()
        gold=g.copy()
        f=reg*np.log(r)+np.min((C-g[None,:]))-reg*logsumexp(((g[None,:]-C)+np.min(C-g[None,:]))/reg,axis=1)
        g=reg*np.log(c)+np.min((C-f[:,None]))-reg*logsumexp(((f[:,None]-C)+np.min(C-f[:,None]))/reg,axis=0)
        if np.linalg.norm(fold-f)<tol and np.linalg.norm(gold-g)<tol:
            break
        P=np.exp((f[:,None]+g[None,:]-C)/reg)
        Error[i]=np.linalg.norm(np.sum(P,0)-c,1)+np.linalg.norm(np.sum(P,1)-r,1)
    min_cost=np.sum(P*C)
    return P,min_cost,Error

def OTplan(x0s,x1s):
    mu=jnp.ones(x0s.shape[0])/x0s.shape[0]
    nu=jnp.ones(x1s.shape[0])/x1s.shape[0]
    M=ot.dist(x0s,x1s)
    return ot.emd(mu,nu,M,numItermax=10000000)

def SINKOTplan(x0s,x1s,reg):
    mu=jnp.ones(x0s.shape[0])/x0s.shape[0]
    nu=jnp.ones(x1s.shape[0])/x1s.shape[0]
    geom=pointcloud.PointCloud(x0s,x1s,epsilon=reg)
    problem=linear_problem.LinearProblem(geom,mu,nu)
    solver=sinkhorn.Sinkhorn()
    OT=solver(problem)
    return OT.matrix

def SINKOTUnbalancedplan(x0s,x1s,reg,tau_a,tau_b):
    mu=jnp.ones(x0s.shape[0])/x0s.shape[0]
    nu=jnp.ones(x1s.shape[0])/x1s.shape[0]
    geom=pointcloud.PointCloud(x0s,x1s,epsilon=reg)
    problem=linear_problem.LinearProblem(geom,mu,nu,tau_a=tau_a,tau_b=tau_b)
    solver=sinkhorn.Sinkhorn()
    OT=solver(problem)
    return OT.matrix


def Computecoupling(X0,X1,t,d):
    weight=OTplan(X0,X1)
    idx,idy=jnp.meshgrid(jnp.arange(X0.shape[0]),jnp.arange(X1.shape[0]),indexing='ij')
    couplings=jnp.column_stack((X0[idx.flatten()],X1[idy.flatten()],jnp.full_like(weight.flatten(),t),weight.flatten()))
    minprob=1/(10*max(X0.shape[0],X1.shape[0]))
    coupling=[]
    while len(coupling)==0:
        coupling=couplings[couplings[:,-1]>minprob]
        minprob/=2
    xs=coupling[:,:d]
    ys=coupling[:,d:2*d]
    time=coupling[:,-2]
    ws=coupling[:,-1]
    return xs,ys,time,ws


def SinkComputecoupling(X0,X1,t,d,reg):
    weight=SINKOTplan(X0,X1,reg)
    idx,idy=jnp.meshgrid(jnp.arange(X0.shape[0]),jnp.arange(X1.shape[0]),indexing='ij')
    couplings=jnp.column_stack((X0[idx.flatten()],X1[idy.flatten()],jnp.full_like(weight.flatten(),t),weight.flatten()))
    minprob=1/(10*max(X0.shape[0],X1.shape[0]))
    coupling=[]
    while len(coupling)==0:
        coupling=couplings[couplings[:,-1]>minprob]
        minprob/=2
    xs=coupling[:,:d]
    ys=coupling[:,d:2*d]
    time=coupling[:,-2]
    ws=coupling[:,-1]
    return xs,ys,time,ws

def SinkUnbalancedcoupling(X0,X1,t,d,reg,tau_a,tau_b):
    weight=SINKOTUnbalancedplan(X0,X1,reg,tau_a,tau_b)
    idx,idy=jnp.meshgrid(jnp.arange(X0.shape[0]),jnp.arange(X1.shape[0]),indexing='ij')
    couplings=jnp.column_stack((X0[idx.flatten()],X1[idy.flatten()],jnp.full_like(weight.flatten(),t),weight.flatten()))
    minprob=1/(10*max(X0.shape[0],X1.shape[0]))
    coupling=[]
    while len(coupling)==0:
        coupling=couplings[couplings[:,-1]>minprob]
        minprob/=2
    xs=coupling[:,:d]
    ys=coupling[:,d:2*d]
    time=coupling[:,-2]
    ws=coupling[:,-1]
    return xs,ys,time,ws


def network_grad_time(model,params):
    def forward(y):
        partialy=y[:-1]
        def loss(partialy):
            fullinpit=jnp.concatenate([partialy,y[-1:]],axis=-1)
            return model.apply({'params':params},fullinpit)
        return jax.grad(loss)(partialy)
    return jax.vmap(forward,in_axes=0)

def multi_network_grad_time(model,params,taskid):
    def forward(y):
        partialy=y[:-1]
        def loss(partialy):
            fullinpit=jnp.concatenate([partialy,y[-1:]],axis=-1)
            return model.apply({'params':params},fullinpit,taskid)
        return jax.grad(loss)(partialy)
    return jax.vmap(forward,in_axes=0)

'''Data Loader'''
# newdata=TensorDataset(xs,ys,t,ws)

def dataload(traindata,valdata,batch_size):
    batchedtrain=DataLoader(traindata,batch_size=batch_size,shuffle=True)
    batchedval=DataLoader(valdata,batch_size=batch_size,shuffle=False)
    sample1=[]
    for xs,ys,t,ws in batchedtrain:
        xs=jnp.array(xs)
        ys=jnp.array(ys)
        t=jnp.array(t)
        ws=jnp.array(ws)
        sample3=(xs,ys,t,ws)
    sample1.append(sample3)
    sample2=[]
    for xs,ys,t,ws in batchedval:
        xs=jnp.array(xs)
        ys=jnp.array(ys)
        t=jnp.array(t)
        ws=jnp.array(ws)
        sample4=(xs,ys,t,ws)
        sample2.append(sample4)
    return sample1,sample2

