import collections
import itertools
import numpy as np
import torch
import opt_einsum

import numqi

from utils import hf_kron


if torch.get_num_threads() != 1:
    torch.set_num_threads(1)


class QECCn23BDnModel(torch.nn.Module):
    def __init__(self, veca, BD2m_m:int, logicalX:str, ind_logical0, tag_real:bool=False):
        super().__init__()
        num_qubit = len(veca)
        error_str,error_torch = numqi.qec.make_pauli_error_list_sparse(num_qubit, distance=3, kind='torch-csr01')
        self.num_qubit = num_qubit
        self.error_torch = error_torch.clone().to(torch.complex128)
        basis0 = np.zeros((len(ind_logical0), 2**num_qubit), dtype=np.complex128)
        basis0[np.arange(len(ind_logical0)), ind_logical0] = 1
        assert np.abs(basis0.conj() @ basis0.T - np.eye(len(basis0))).max() < 1e-12
        basis1 = (numqi.qec.hf_pauli(logicalX) @ basis0.T).T
        self.manifold = numqi.manifold.Sphere(len(basis0), dtype=(torch.float64 if tag_real else torch.complex128))
        self.basis0 = torch.tensor(basis0, dtype=torch.complex128).T.to_sparse_csr()
        self.basis1 = torch.tensor(basis1, dtype=torch.complex128).T.to_sparse_csr()
        self.logical_gate = torch.tensor(numqi.gate.rz(-2*np.pi/BD2m_m), dtype=torch.complex128)
        # self.theta_su2 = torch.nn.Parameter(torch.randn(num_qubit, dtype=torch.float64))
        self.theta_su2 = torch.tensor(np.array(veca)*2*np.pi/BD2m_m, dtype=torch.float64)

        N = num_qubit
        tmp0 = [y for x in range(N) for y in [(2,2), (N+x,x)]]
        self.contract_expr = opt_einsum.contract_expression([2]*(N+1), list(range(N))+[2*N], [2]*(N+1), list(range(N,2*N))+[2*N+1], *tmp0, [2*N+1,2*N])
        self.mask_lambda = None

    def forward(self, return_info:bool=False):
        coeff = self.manifold().to(torch.complex128)
        # coeff = torch.concat([coeff[:2]*1j, coeff[2:]])
        # coeff = torch.concat([coeff[:2]*1j, coeff[4:]*0], axis=0) + coeff[2:]
        q0 = torch.stack([self.basis0@coeff, self.basis1@coeff], axis=1)
        lambda_aij = numqi.qec.knill_laflamme_hermite_mul(self.error_torch, q0)
        if hasattr(self, 'manifold_su2'):
            assert not hasattr(self, 'theta_su2')
            su2 = self.manifold_su2()
        elif hasattr(self, 'theta_su2'):
            su2 = numqi.gate.rz(self.theta_su2)
        else:
            su2 = self.su2
        tmp1 = q0.reshape([2]*(self.num_qubit+1))
        logicalU = self.contract_expr(tmp1, tmp1.conj(), *su2)
        constraint = [
            torch.vdot(q0[:,0], q0[:,1]),
            lambda_aij[:,0,1],
            lambda_aij[:,0,0].real-lambda_aij[:,1,1].real,
            logicalU - self.logical_gate,
        ]
        if return_info:
            info = dict(lambda_aij=lambda_aij.detach().numpy(), q0=q0.detach().numpy(), logicalU=logicalU.detach().numpy(), su2=su2.detach().numpy(), coeff=coeff.detach().numpy())
        loss = sum([torch.vdot(x.reshape(-1), x.reshape(-1)).real for x in constraint])
        ret = (loss,info) if return_info else loss
        return ret


def check_BD2m_transversal_zm(veca, m:int, logicalX:str, zero_eps=1e-10):
    num_qubit = len(veca)
    assert (len(logicalX)==num_qubit) and (set(logicalX)<={'I','X'})
    assert (veca.sum()+1)%m==0
    ind0 = numqi.qec.get_BD2m_submultiset(m, veca)
    basis0 = np.zeros((len(ind0),2**num_qubit), dtype=np.float64)
    basis0[np.arange(len(ind0)), ind0] = 1
    basis1 = (numqi.qec.hf_pauli(logicalX) @ basis0.T).T
    su2 = numqi.gate.rz(veca*2*np.pi/m)
    tmp1 = np.stack([basis0,basis1], axis=0)
    z1 = np.einsum(tmp1, [0,1,2], tmp1, [3,4,5], hf_kron(*su2), [2,5], [0,3,1,4], optimize=True)
    assert np.abs(numqi.gate.rz(-2*np.pi/m).reshape(2,2,1,1) * np.eye(z1.shape[2]) - z1).max() < zero_eps


num_qubit = 8
BD2m_m = 19
logicalX = 'X'*num_qubit #not support other options yet

veca = np.array([2,3,3,4,4,5,7,9])
ind0 = numqi.qec.get_BD2m_submultiset(BD2m_m, veca)

check_BD2m_transversal_zm(veca, BD2m_m, 'XXXXXXXX')

error_str,error_scipy = numqi.qec.make_pauli_error_list_sparse(num_qubit, distance=3, kind='scipy-csr01')

ind0 = [0, 35, 46, 54, 67, 78, 86, 105, 113, 124, 153, 165, 197, 234, 242]
basis0 = np.zeros((len(ind0),2**num_qubit), dtype=np.float64)
basis0[np.arange(len(ind0)), ind0] = 1
basis0 = np.stack([basis0,basis0*1j], axis=1).reshape(-1, 2**num_qubit)
tmp0 = set(list(range(len(basis0))))
ind_mask = set([2,4,6,8,10,12,15,17,19,21,23,25,27,29])
basis0 = basis0[sorted(tmp0-ind_mask)]
basis1 = (numqi.qec.hf_pauli('X'*num_qubit) @ basis0.T).T


# ind0 = [0, 35, 46, 54, 67, 78, 86, 105, 113, 124, 153, 165, 197, 234, 242]
# ind_mask = [2,4,6,8,10,12,15,17,19,21,23,25,27,29]
# tmp0 = sorted(set(range(len(ind0)*2))-set(ind_mask))
# ind_real = [ind0[x//2] for x in tmp0 if x%2==0]
# ind_imag = [ind0[x//2] for x in tmp0 if x%2==1]


# ind_logical0 = numqi.qec.get_BD2m_submultiset(BD2m_m, veca)
# model = QECCn23BDnModel(veca, BD2m_m=BD2m_m, logicalX='XXXXXXXX', ind_logical0=ind_logical0, tag_real=False)

# su2 = numqi.gate.rz(veca*2*np.pi/m)
# tmp1 = np.stack([basis0,basis1], axis=0)
# z1 = np.einsum(tmp1, [0,1,2], tmp1, [3,4,5], hf_kron(*su2), [2,5], [0,3,1,4], optimize=True)
# assert np.abs(numqi.gate.rz(-2*np.pi/BD2m_m).reshape(2,2,1,1) * np.eye(z1.shape[2]) - z1).max() < 1e-10

## TODO add 823 BD38 to unittest
## TODO 723 wired code (1,0.6)
## TODO ((n,2,3))+BD2m -> ((n+1,2,3))+BD4m
## TODO veca class

N0 = len(basis0)
tmp0 = basis0.conj() @ (error_scipy @ basis1.T).reshape(-1,2**num_qubit,N0)
tmp1 = (basis0.conj() @ (error_scipy @ basis0.T).reshape(-1,2**num_qubit,N0)
         - basis1.conj() @ (error_scipy @ basis1.T).reshape(-1,2**num_qubit,N0))
z0 = np.concatenate([tmp0, tmp1], axis=0)
z0 = (z0 + z0.transpose(0,2,1)) # real coefficient only
z1 = numqi.qec.pick_indenpendent_vector(z0.reshape(-1,N0*N0), tag_pure_imag=True).reshape(-1,N0,N0)
z1[np.abs(z1)<1e-10] = 0
print(z1.shape)

i0,i1,i2 = np.nonzero((z1*(1-np.eye(z1.shape[1]))))
tmp0 = sorted(collections.Counter(i1.tolist()).items(), key=lambda x: -x[1])
print(tmp0)
tmp1 = [x[0] for x in tmp0 if x[1]==tmp0[0][1]]
print(tmp1)
tmp2 = sorted(set(range(30)) - ind_mask)
print([tmp2[x] for x in tmp1])


zc0 = np.around(np.diagonal(z1, axis1=1, axis2=2), 3).astype(np.int64)//4

import cvxpy
cvxX = cvxpy.Variable(zc0.shape[1])
constraint = [
    cvxX>=0,
    zc0 @ cvxX==0,
    cvxpy.sum(cvxX)==1,
]
prob = cvxpy.Problem(cvxpy.Minimize(0), constraint)
prob.solve()

basis0 = np.zeros((15, 2**8), dtype=np.complex128)
basis0[np.arange(7), [0, 35, 46, 54, 67, 78, 86]] = 1
basis0[np.arange(7,15), [105, 113, 124, 153, 165, 197, 234, 242]] = 1j
basis1 = (numqi.qec.hf_pauli('X'*num_qubit) @ basis0.T).T

N1 = 8
error_str,error_scipy = numqi.qec.make_pauli_error_list_sparse(N1, distance=3, kind='scipy-csr01')
N0 = len(basis0)
tmp0 = basis0.conj() @ (error_scipy @ basis1.T).reshape(-1,2**N1,N0)
tmp1 = basis0.conj() @ (error_scipy @ basis0.T).reshape(-1,2**N1,N0) - basis1.conj() @ (error_scipy @ basis1.T).reshape(-1,2**N1,N0)
z0 = np.concatenate([tmp0, tmp1], axis=0)
z0 = (z0 + z0.transpose(0,2,1)) # real coefficient only
z1 = numqi.qec.pick_indenpendent_vector(z0.reshape(-1,N0*N0), tag_pure_imag=True).reshape(-1,N0,N0)
z1[np.abs(z1)<1e-10] = 0
assert np.abs(z1 - np.around(z1).astype(np.int64)).max() < 1e-10
z1 = np.around(z1).astype(np.int64)
assert np.all(z1%4==0)
x0 = np.diagonal(z1//4, axis1=1, axis2=2)
