# -*- coding: utf-8 -*-
""" Toolbox for quantum metrology
    Calculate Holevo bound using semidefinite programing
    Calculate SLD and RLD bounds
    Author: Le Bin Ho @ 2022
"""

#__all__ = ['sld_qfim',
#           'sld_bound',
#           'rld_qfim',
#           'rld_bound',
#           'cfim',
#           'cfim_bound']

import qiskit
import vqa.circuits
import vqa.constants
import vqa.vqm

import numpy as np
from autograd.numpy.linalg import inv, multi_dot, eigh, matrix_rank, norm
from scipy.linalg import sqrtm, solve_sylvester
import copy


"""
def holevo(W):
    # calculate the Holevol bound using semidefinite programing
    # input: W - random or identity matrix
    #        n - numbner of parameters
    # output: Optimal Holevo value and optimal matrix V
    
    p = 3 # number of conditions - We fixed
    n = W.shape[0]
    W = np.array(W)
    X = []
    b = []
    for i in range(p):
        X.append(np.random.randn(n, n))
        b.append(np.random.randn())

    # Create a symmetric matrix variable.
    V = cp.Variable((n,n), symmetric=True)
    # The operator >> denotes matrix inequality.
    constraints = [V >> 0]
    constraints += [cp.trace(X[i] @ V) == b[i] for i in range(p)]
    prob = cp.Problem(cp.Minimize(cp.trace(W @ V)),constraints)
    prob.solve()

    # return the optimal Holevol value and matrix V
    return prob.value, V.value

"""

def sld_qfim(qc: qiskit.QuantumCircuit,cirs,coefs,params):
    
    """ calculate the QFIM bases SLD 
    
    Args:
        - qc : initial circuit
        - cirs: set of circuits (we always qc_add as model, no call here)
        - params: phase encode
        - coefs: coefs

    Returns:
       - qfim 
    """ 
    
    cir = vqa.vqm.qc_add(qc.copy(), cirs, coefs, params)
    rho = vqa.circuits.state_density(cir.copy()) #rho after evolve
    grho = _grad_rho(qc.copy(), cirs, coefs, params)
    
    d = len(grho) # number of paramaters
    H = np.zeros((d,d), dtype = complex)
    invM = _inv_M(rho)
    
    vec_grho = []  
    for i in range(d):
        vec_grho.append(_vectorize(grho[i]))

    for i in range(d):
        for j in range(d): 
            H[i,j] = 2*multi_dot([np.conjugate(vec_grho[i]).T, invM, vec_grho[j]]) 
    return np.real(H)  

 
def sld_bound(qc: qiskit.QuantumCircuit,cirs,coefs,params):
    """ return the SLD bound 
    
    Args:
        - qc
        - cirs
        - params
        - coefs

    Returns:
       - sld bound
    """
    
    list2str = list(map(lambda f: f.__name__, cirs)) 
    idx = list2str.index('u_phase')
    
    d = len(params[idx])
    W = np.identity(d)
    
    sld = sld_qfim(qc.copy(), cirs, coefs, params)
    return np.real(np.trace(W @ inv(sld + np.eye(len(sld)) * 10e-10)))


def rld_qfim(qc: qiskit.QuantumCircuit,cirs,coefs,params): 
    
    """ calculate the QFIM bases RLD 
    
    Analytical:
        H_ij = tr(rho*L_i*L_j^\dag)
               = tr(L_j^\dag*rho*L_i)
               
        submit rho*L_i = der_i(rho) : this is rld
        we get: H_ij = tr(L_j^\dag*dev_i(rho))
        
        apply tr(A^dag*B) = vec(A)^dag*vec(B), 
        we have: H_ij = vec(L_j)^dag*vec[dev_i(rho)]
        here: vec[L_j)] = (I x rho)^{-1)*vec(L_j)
        
    Args:
        - qc
        - cirs
        - params
        - coefs

    Returns:
       - qfim via rld   
    """ 
    qc_func = vqa.vqm.qc_add(qc.copy(),cirs, coefs, params)
    rho = vqa.circuits.state_density(qc_func.copy())
    grho = _grad_rho(qc.copy(),cirs, coefs, params)
    
    d = len(grho) # number of estimated parameters  
    R = np.zeros((d,d), dtype = complex)
    IR = _i_R(rho)
    
    vec_grho = []  
    for i in range(d):
        vec_grho.append(_vectorize(grho[i]))
    
    for i in range(d):
        for j in range(d): 
            vecLj = np.conjugate(multi_dot([IR, vec_grho[j]])).T
            R[i,j] = multi_dot([vecLj, vec_grho[i]])  
    """
    
    #for solveing by sylvester equation
    zeros = np.zeros((len(rho),len(rho)))
    L = sylvester(rho,zeros,grho)
    
    for i in range(d):
        for j in range(d): 
            R[i,j] = np.trace(multi_dot([rho, L[i], np.conj(L[j]).T]))
    """
    return R


def rld_bound(qc: qiskit.QuantumCircuit,cirs,coefs,params):
    """ return the SLD bound 
    
    Args:
        - qc
        - cirs
        - params
        - coefs
        - W: weight matrix

    Returns:
       - rld bound
    """
    
    list2str = list(map(lambda f: f.__name__, cirs)) 
    idx = list2str.index('u_phase')
    
    d = len(params[idx])
    W = np.identity(d)
        
    rld = rld_qfim(qc,cirs, coefs, params)
    invrld = inv(rld + np.eye(len(rld)) * 10e-10)
    R1 = np.trace(W @ np.real(invrld))
    R2 = norm(multi_dot([sqrtm(W), np.imag(invrld),sqrtm(W)])) 
    
    return R1 + R2


def cfim(qc: qiskit.QuantumCircuit,cirs,coefs,params):
    """ return the classical fisher information matrix
    
    Args:
        - qc
        - cirs
        - params
        - coefs
    Returns:
        - cfim
    """
    # measurements
    qc_copy = vqa.vqm.qc_add(qc.copy(),cirs, coefs, params)
    pro = vqa.circuits.measure_born(qc_copy)
            
    list2str = list(map(lambda f: f.__name__, cirs)) 
    idx = list2str.index('u_phase')  
    
    dpro = []
    d = len(params[idx])
    s = vqa.constants.step_size
    
    #remove zeros indexs
    idxs = np.where(pro<=10e-18)
    pro = np.delete(pro, idxs)

    
    for i in range(0, d):
        # We use the parameter-shift rule explicitly
        # to compute the derivatives
        params1, params2 = copy.deepcopy(params), copy.deepcopy(params)    
        params1[idx][i] += s 
        params2[idx][i] -= s
        
        plus = vqa.vqm.qc_add(qc.copy(), cirs, coefs, params1) 
        minus = vqa.vqm.qc_add(qc.copy(), cirs, coefs, params2)
        gr = (vqa.circuits.measure_born(plus)-vqa.circuits.measure_born(minus))/(2*s)
        gr = np.delete(gr, idxs)
        dpro.append(np.array(gr))
    
    matrix = np.zeros((d,d), dtype = float)
    for i in range(d):
        for j in range(d):      
            matrix[i,j] = np.sum(dpro[i] * dpro[j] / pro)

    return matrix


def cls_bound(qc: qiskit.QuantumCircuit,cirs,coefs,params):
    """ return the classical bound 
    
    Args:
        - qc
        - create_circuit_func
        - params
        - coefs
        - W: weight matrix

    Returns:
       - rld bound
    """
    
    #list2str = list(map(lambda f: f.__name__, cirs)) 
    #idx = list2str.index('u_phase') 
    #d = len(params[idx])
        
    clf = cfim(qc, cirs, coefs, params)
    W = np.identity(len(clf))
    
    return np.trace(W @ inv(clf + np.eye(len(clf)) * 10e-10))


def _vectorize(rho):
    # return a vectorized of rho
    # rho: a matrices (data)
    vec_rho = np.reshape(rho, (len(rho)**2,1), order='F')
    return vec_rho


def _grad_rho(qc,cirs,coefs,params):
    """ calculate drho by parameter-shift rule
    
    Args:
        - qc: initial cuircuit (we need to reset for rho(+) and rho(-)
        - qc_func: circuit after apply a shift
        - cirs: circuits
        - params: params
        - coefs: coefs
        
    Return:
        - gradient of state density w.r.t. all phases
    """
    
    dp = [] #array of matrices rho
    s = vqa.constants.step_size

    
    list2str = list(map(lambda f: f.__name__, cirs)) 
    idx = list2str.index('u_phase')    
  
    for i in range(0, len(params[idx])):
        params1, params2 = copy.deepcopy(params), copy.deepcopy(params)       
        params1[idx][i] += s
        params2[idx][i] -= s
        
        plus = vqa.vqm.qc_add(qc.copy(), cirs, coefs, params1) 
        minus = vqa.vqm.qc_add(qc.copy(), cirs, coefs, params2)       
        dp.append((vqa.circuits.state_density(plus)-vqa.circuits.state_density(minus))/(2*s))
        
    return dp


def _inv_M(rho, epsilon = 10e-10): 
    """ return inverse matrix M 
        M = rho.conj()*I + I*rho.conj()
    
    Args:
        - quantum state rho (data)

    Returns:
       - inverse matrix M 
    """
    
    d = len(rho)
    M = np.kron(np.conj(rho), np.identity(d)) + np.kron(np.identity(d), rho)
    return inv(M + np.eye(len(M)) * epsilon)


def _i_R(rho, epsilon = 10e-10):
    """ return inverse of matrix R 
        R = I*rho.conj()
    
    Args:
        - quantum state rho (data)

    Returns:
       - inverse matrix R 
    """    
    d = len(rho)
    R = np.kron(np.identity(d), rho)
    return inv(R) #inv(R + np.eye(len(R)) * epsilon)


def sylvester(A,B,C):
    """ solve the sylvester function:
        AX + XB = C
        here A,B = rho for SLD
             A = rho, B = 0 for RLD
             C = drho
    Args:
        - rho: input quantum state
        - drho: input derivative of quantum state
    Returns:
        - X operator
    """
    
    lenC = len(C) #we have several drho (gradient of rho)
    X = []
    
    for i in range(lenC):
        L = solve_sylvester(A, B, C[i])
        X.append(L)
        
    #print(X)  
    return X
        
        
