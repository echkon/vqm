# -*- coding: utf-8 -*-
""" Variational quantum metrology
    Author: Le Bin Ho @ 2023
"""

#__all__ = []

import qiskit
import numpy as np
import vqa.fitting 

from qiskit.extensions import UnitaryGate
from qiskit import QuantumRegister, ClassicalRegister, QuantumCircuit

#model can dua params vao

def training(qc: qiskit.QuantumCircuit,
             cirs,
             coefs,
             params,
             which_train,
             cost_func,
             grad_func,
             opt_func,
             num_steps):
    
    """ train a vqm model
    
    Args:
        - qc (QuantumCircuit): Fitting circuit
        - cirs: set of circuits functions
        - coefs: coefs
        - params: set of paramaters
        - which_train: training option
        - coss_func (FunctionType): coss function
        - grad_func (FunctionType): Gradient function
        - opt_func (FunctionType): Otimizer function
        - num_steps (Int): number of iterations
        
    Returns:
        - optimal parameters
    """
    

    return vqa.fitting.fit(qc,cirs,coefs,params,which_train,cost_func,grad_func,opt_func,num_steps)
    

def qc_add(qc: qiskit.QuantumCircuit,cirs,coefs,params):
    
    """ create a full circuit from qc_func
    
    Args:
        - qc: initial circuit
        - cirs: list of circuits
        - params: parameters
        - coefs: coefs includes num_layers, time, gamma
        
    Returns:
        - qc: final circuit
    """
    
    cirq = qc.copy()    
    for i in range (len(params)):
        cirq += cirs[i](qc.copy(), coefs[i], params[i])
        
    return cirq
 
#
# custom unitary for phase and noises
#

def u_phase(qc: qiskit.QuantumCircuit, t, params):
    
    """Add phase model to the circuit
    
    Args:
        - qc (qiskit.QuantumCircuit): quantumcircuit
        - t: time coefs
        - params  (number of parameters)
        
    Return
        - qc
    """
    n = qc.num_qubits
    
    for i in range(n):        
        qc.append(_u_gate(t, params),[i])
    
    return qc


def _u_gate(t, params):
    """ return arbitary gates
    
    Args:
        - t: time coefs
        - params: phases
    
    Method:
        H = x*sigma_x + y*sigma_y +  z*sigma_z
        U = exp(-i*t*H)
        t: time (use for time-dephasing)
    """
        
    x,y,z = params[0],params[1],params[2]
    p2 = np.sqrt(x*x + y*y + z*z)
    tp2 = t * p2
    
    u11 = np.cos(tp2) - 1j*z*np.sin(tp2)/p2
    u12 = (-1j*x - y)*np.sin(tp2)/p2
    u21 = (-1j*x + y)*np.sin(tp2)/p2
    u22 = np.cos(tp2) + 1j*z*np.sin(tp2)/p2
    
    u = [[u11, u12],[u21,u22]]
    gate = UnitaryGate(u, 'Phase')

    return gate


from qiskit.quantum_info import Kraus, SuperOp, Choi
from qiskit.providers.aer.utils import approximate_quantum_error
import qiskit.quantum_info as qi
from qiskit.providers.aer.noise import QuantumError


def dephasing(qc: qiskit.QuantumCircuit, t, y):
    """Add dephasing to the circuit

    Args:
        - qc (qiskit.QuantumCircuit): quantumcircuit
        - t: time coefs
        - y: gamma in params
    
    Return
        - qc
    """ 
    y = y[0] if type(y) == list else y
    lamb = 1 - np.exp(-t*y)

    # kraus operators
    k1 = np.array([[np.sqrt(1 - lamb), 0],[0, 1]])
    k2 = np.array([[np.sqrt(lamb), 0],[0, 0]])
    
    noise_ops = Kraus([k1,k2])
    kraus_to_error = QuantumError(noise_ops) 
    
    for i in range(qc.num_qubits):
        qc.append(kraus_to_error,[i])
        
    return qc


def bit_flip(qc: qiskit.QuantumCircuit, t, y):
    """Add bit flip to the circuit

    Args:
        - qc (qiskit.QuantumCircuit): quantumcircuit
        - t: time
        - y: gamma
    
    Return
        - qc
    """ 
    y = y[0] if type(y) == list else y
    lamb = 1 - np.exp(-t*y)
    
    noise_ops = Kraus([np.sqrt(1-lamb) * np.array([[1, 0], [0, 1]]),
                   np.sqrt(lamb) * np.array([[0, 1], [1, 0]])])
    kraus_to_error = QuantumError(noise_ops)
    
    for i in range(qc.num_qubits):
        qc.append(kraus_to_error,[i])
        
    return qc


def markovian(qc: qiskit.QuantumCircuit, t, y):
    """Add bit flip to the circuit

    Args:
        - qc (qiskit.QuantumCircuit): quantumcircuit
        - t:  time
        - y: gamma
    
    Return
        - qc
    """ 
    y = y[0] if type(y) == list else y
    qt = 1 - np.exp(-y*t) #y = gamma = 0.1
    
    k1 = np.array([[np.sqrt(1-qt) ,0], [0,1]])
    k2 = np.array([[np.sqrt(qt),0], [0,0]])
    
    noise_ops = Kraus([k1,k2])
    kraus_to_error = QuantumError(noise_ops)  
    
    for i in range(qc.num_qubits):
        qc.append(kraus_to_error,[i])
        
    return qc


def non_markovian(qc: qiskit.QuantumCircuit, t, y):
    """Add bit flip to the circuit

    Args:
        - qc (qiskit.QuantumCircuit): quantumcircuit
        - t:  time
        - y: gamma
    
    Return
        - qc
    """ 
    y = y[0] if type(y) == list else y
    tc = 20.0
    qt = 1 - np.exp(-y*t**2/(2*tc)) #gamma = 0.1
    
    k1 = np.array([[np.sqrt(1-qt) ,0], [0,1]])
    k2 = np.array([[np.sqrt(qt),0], [0,0]])
    
    noise_ops = Kraus([k1,k2])
    kraus_to_error = QuantumError(noise_ops)  
    
    for i in range(qc.num_qubits):
        qc.append(kraus_to_error,[i])
        
    return qc
 