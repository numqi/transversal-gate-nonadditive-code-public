import os
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import torch
import numqi
import opt_einsum
import scipy.linalg

from zzz233 import to_pickle_wrapper, from_pickle_wrapper
to_pickle = to_pickle_wrapper('623a.pkl')
from_pickle = from_pickle_wrapper('623a.pkl')

if torch.get_num_threads() != 1:
    torch.set_num_threads(1)

np_rng = np.random.default_rng()


class QECC623C10TransversalGateModel(torch.nn.Module):
    def __init__(self, tag_phase:bool=False):
        super().__init__()
        self.gate_list = torch.tensor(numqi.qec.get_su2_finite_subgroup_generator('C10'), dtype=torch.complex128)
        num_qubit = 6
        self.num_qubit = num_qubit
        self.error_torch = numqi.qec.make_pauli_error_list_sparse(num_qubit, distance=3, kind='torch-csr01')[1].to(torch.complex128)
        # self.manifold = numqi.manifold.Stiefel(2**num_qubit, rank=2, dtype=torch.complex128)
        self.manifold = numqi.manifold.Sphere(12, batch_size=2, dtype=torch.complex128)
        ind0 = np.array([5,  9, 14, 17, 22, 26, 33, 38, 42, 50, 60, 63], dtype=np.int64)
        ind1 = np.array([ 0,  3, 13, 21, 25, 30, 37, 41, 46, 49, 54, 58], dtype=np.int64)
        basis0 = np.zeros((len(ind0),2**6), dtype=np.complex128)
        basis0[np.arange(len(ind0)), ind0] = 1
        basis1 = np.zeros((len(ind1),2**6), dtype=np.complex128)
        basis1[np.arange(len(ind1)), ind1] = 1
        self.basis0 = torch.tensor(basis0, dtype=torch.complex128)
        self.basis1 = torch.tensor(basis1, dtype=torch.complex128)

        X,Y,Z,I = numqi.gate.X, numqi.gate.Y, numqi.gate.Z, numqi.gate.I
        self.su2_fixZ = torch.tensor(np.stack([1j*X, 1j*Z, 1j*Z, I, I, I], axis=0), dtype=torch.complex128)
        # tmp0 = gate_list.shape[0] * num_qubit
        # self.theta_su2 = torch.nn.Parameter(torch.randn(6, dtype=torch.float64))
        tmp0 = [numqi.gate.rz(x) for x in [2*np.pi/5, 2*np.pi/5, 2*np.pi/5, 2*np.pi/5, 4*np.pi/5, -4*np.pi/5]]
        self.gate_su2 = torch.tensor(np.stack(tmp0, axis=0), dtype=torch.complex128)
        self.mask = torch.tensor(np.array([[1, 1, 0, 1, 0, 0, 1, 0, 0, 0, 1, 1], [1, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1]]), dtype=torch.float64)

        N0 = num_qubit
        tmp0 = [y for x in range(N0) for y in [(2,2), (N0+x,x)]]
        self.contract_expr = opt_einsum.contract_expression([2]*(N0+1), list(range(N0))+[2*N0], [2]*(N0+1), list(range(N0,2*N0))+[2*N0+1], *tmp0, [2*N0+1,2*N0])

    def set_lambda_target(self, x):
        if (x is None) or (x=='min') or (x=='max'):
            self.lambda_target = x
        else:
            self.lambda_target = torch.tensor(float(x), dtype=torch.float64)

    def forward(self, return_info:bool=False):
        # q0 = self.manifold()
        coeff = self.manifold()
        q0 = torch.stack([coeff[0] @ self.basis0, coeff[1] @ self.basis1], axis=1)
        lambda_aij = numqi.qec.knill_laflamme_hermite_mul(self.error_torch, q0)
        # lambda_aij = q0.T.conj() @ (self.error_torch @ q0).reshape(-1, *q0.shape)
        # su2 = numqi.gate.rz(self.theta_su2)
        su2 = self.gate_su2
        tmp1 = q0.reshape([2]*(self.num_qubit+1))
        logicalU = self.contract_expr(tmp1, tmp1.conj(), *su2)
        constraint = [
            # coeff[:,:2].imag,
            (coeff.imag*self.mask),
            coeff[:,2].imag,
            # torch.angle(coeff[0,2])-torch.angle(coeff[1,2]),
            logicalU - self.gate_list*(-1),
            lambda_aij[:,0,1],
            lambda_aij[:,0,0].real - lambda_aij[:,1,1].real,
        ]
        if return_info:
            info = dict(lambda_aij=lambda_aij.detach().numpy(), q0=q0.detach().numpy(), logicalU=logicalU.detach().numpy(), su2=su2.detach().numpy())
        loss = sum([torch.vdot(x.reshape(-1), x.reshape(-1)).real for x in constraint])
        ret = (loss,info) if return_info else loss
        return ret


trans_gate_list = numqi.qec.get_su2_finite_subgroup_generator('C10')
model = QECC623C10TransversalGateModel()
theta_optim = numqi.optimize.minimize(model, 'uniform', num_repeat=40, tol=1e-9, early_stop_threshold=1e-4)
if theta_optim.fun < 1e-4:
    theta_optim = numqi.optimize.minimize(model, theta_optim.x, num_repeat=1, tol=1e-20)
info = model(return_info=True)[1]
coeff = model.manifold().detach().numpy()
coeff *= np.exp(-1j*np.angle(coeff[:,:1]))
print(np.abs(coeff))
print(np.around(np.angle(coeff)/np.pi, 4))

'''
[[0.3162 0.3162 0.2582 0.3162 0.2582 0.2582 0.3162 0.2582 0.2582 0.2582 0.3162 0.3162]
 [0.3162 0.3162 0.2582 0.2582 0.2582 0.3162 0.2582 0.2582 0.3162 0.2582 0.3162 0.3162]]
[[ 2.7933e-18 -5.3685e-02  8.8864e-01 -8.6767e-01  7.4132e-01 -6.4570e-01  7.8192e-01 -9.4242e-01  3.3723e-01  8.5657e-01  2.6651e-01 -4.8410e-01]
 [ 8.4341e-19 -7.5061e-01 -3.4068e-01  1.7867e-01 -5.4168e-01 -2.6602e-01 -8.3841e-01 -2.2543e-01 -6.1643e-01 -3.7275e-01  5.6958e-01  5.1590e-01]]
'''


hfb = lambda x: bin(x)[2:].rjust(6,'0')
ind0 = np.array([5,  9, 14, 17, 22, 26, 33, 38, 42, 50, 60, 63], dtype=np.int64)
ind1 = np.array([ 0,  3, 13, 21, 25, 30, 37, 41, 46, 49, 54, 58], dtype=np.int64)
# 000101 001001 (001110) 010001 (010110) (011010) 100001   (100110) (101010) (110010) 111100 111111
# 000000 000011 (001101) (010101) (011001) 011110 (100101) (101001) 101110   (110001) 110110 111010

a = np.sqrt(1/10) #0.3162
b = np.sqrt(1/15) #0.2582
p3 = np.exp(1j*np.pi/3)
p6 = p3*p3
code = np.zeros((2,64), dtype=np.complex128)
code[0, [5,9,14,17,22,26,33,38,42,50,60,63]] = np.array([a,a,b,a,b/p6,b*p6, -a,b/p3,b*p3,-b,-a,a])
code[1, [0,3,13,21,25,30,37,41,46,49,54,58]] = np.array([a,-a,b,b*p6,b/p6,a, b*p3,b/p3,-a,-b,-a,-a])
error_str,error_scipy = numqi.qec.make_pauli_error_list_sparse(6, distance=3, kind='scipy-csr01')
z0 = code.conj() @ (error_scipy @ code.T).reshape(-1,64,2)
assert np.abs(z0[:,0,1]).max() < 1e-10
assert np.abs(z0[:,1,0]).max() < 1e-10
assert np.abs(z0[:,0,0]-z0[:,1,1]).max() < 1e-10
# qweA = numqi.qec.get_weight_enumerator(code, tagB=False) #[1, 0, 0.84, 0, 11.64, 15.36, 3.16]
import functools
hf_kron = lambda *x: functools.reduce(np.kron, x)
tmp0 = [numqi.gate.rz(x) for x in [2*np.pi/5, 2*np.pi/5, 2*np.pi/5, 2*np.pi/5, 4*np.pi/5, -4*np.pi/5]]
np1 = code.conj() @ hf_kron(*tmp0) @ code.T
assert np.abs(np1 + numqi.gate.rz(2*np.pi/5)).max() < 1e-10

