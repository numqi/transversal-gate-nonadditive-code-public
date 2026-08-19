import os
import collections
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
import numpy as np
import scipy.linalg
import torch
import numqi
import opt_einsum

from utils import SpecialUnitary2XZManifold
from utils01 import rotate_index, hf_theta_round, hfb

from zzz233 import to_pickle_wrapper, from_pickle_wrapper
to_pickle = to_pickle_wrapper('823_BDn.pkl')
from_pickle = from_pickle_wrapper('823_BDn.pkl')

if torch.get_num_threads() != 1:
    torch.set_num_threads(1)

np_rng = np.random.default_rng()

# def hf_build_basis(i0, i1):
#     assert (len(i0) + len(i1))>=1
#     ret = np.zeros((1,128), dtype=np.float64)
#     a = 1/np.sqrt(len(i0)+len(i1))
#     if len(i0)>0:
#         ret[0,i0] = a
#     if len(i1)>0:
#         ret[0,i1] = -a
#     return ret

def hf_build_basis(ind, value, n=7):
    ind = np.asarray(ind, dtype=np.int64)
    value = np.asarray(value)
    ret = np.zeros(2**n, dtype=np.complex128)
    ret[ind] = value/np.linalg.norm(value)
    return ret.reshape(1,-1)



class QECCn23BD2mModel(torch.nn.Module):
    def __init__(self, num_qubit:int, logicalX:str, BD2m_m:int, tag_real:bool):
        super().__init__()
        error_str,error_torch = numqi.qec.make_pauli_error_list_sparse(num_qubit, distance=3, kind='torch-csr01')
        self.num_qubit = num_qubit
        self.error_torch = error_torch.clone().to(torch.complex128)
        # ind_real = np.array([0, 105, 113, 124, 153, 165, 197, 234, 242], dtype=np.int64)
        # ind_imag = np.array([0, 35, 46, 54, 67, 78, 86], dtype=np.int64)
        # tmp0 = np.zeros((len(ind_real), 2**num_qubit), dtype=np.complex128)
        # tmp0[np.arange(len(ind_real)), ind_real] = 1
        # tmp1 = np.zeros((len(ind_imag), 2**num_qubit), dtype=np.complex128)
        # tmp1[np.arange(len(ind_imag)), ind_imag] = 1j
        # basis0 = np.concatenate([tmp0, tmp1], axis=0)
        # tmp2 = np.zeros((1,128), dtype=np.complex128)
        # s = np.sqrt
        # tmp2[0,[57,108]] = np.array([1,-1])/s(2) #[1,-1,-1]
        # basis0 = np.concatenate([basis0,tmp2], axis=0)
        # ind0 = [0,14,25,37,58,67,124,150,233]
        # basis0 = np.zeros((len(ind0), 2**num_qubit), dtype=np.complex128)
        # basis0[np.arange(len(ind0)), ind0] = 1
        basis0 = np.eye(2**num_qubit)
        basis1 = (numqi.qec.hf_pauli(logicalX) @ basis0.T).T
        assert np.abs(basis0.conj() @ basis0.T - np.eye(len(basis0))).max() < 1e-12
        self.manifold = numqi.manifold.Sphere(len(basis0), dtype=(torch.float64 if tag_real else torch.complex128))
        # self.manifold = numqi.manifold.Stiefel(2**num_qubit, rank=2, dtype=(torch.float64 if tag_real_stiefel else torch.complex128))
        self.BD2m_m = BD2m_m
        self.basis0 = torch.tensor(basis0, dtype=torch.complex128).T.to_sparse_csr()
        self.basis1 = torch.tensor(basis1, dtype=torch.complex128).T.to_sparse_csr()
        # self.manifold_su2 = numqi.manifold.SpecialOrthogonal(2, batch_size=num_qubit, dtype=torch.complex128)
        # self.logical_gate = torch.tensor(numqi.qec.get_su2_finite_subgroup_generator(f'BD{2*BD2m_m}')[1], dtype=torch.complex128)
        self.logical_gate = torch.tensor(numqi.gate.rz(-2*np.pi/BD2m_m), dtype=torch.complex128) #BD16
        self.theta_su2 = torch.nn.Parameter(torch.randn(num_qubit, dtype=torch.float64))
        # self.theta_su2 = torch.tensor(np.array([1,2,4,7,8,11,13,17])*2*np.pi/BD2m_m, dtype=torch.float64)
        # BD64 [2,4,7,8,9,10,12,15]
        # BD72 [3,5,6,8,10,12,13,14] [0,13,60,90,102,150,163,236,241]
        # BD74 [4,6,7,9,10,11,12,14] [0,7,60,90,105,142,153,163,244]
        # BD76 [2,4,7,9,10,12,15,16] [0,13,35,60,90,102,150,234,241]
        # BD78 [3,5,8,9,10,12,14,16] [0,19,60,102,105,142,165,220,242]
        # BD80 [2,5,6,8,11,14,15,18] [0,14,21,58,102,105,165,195,220]

        N = num_qubit
        tmp0 = [y for x in range(N) for y in [(2,2), (N+x,x)]]
        self.contract_expr = opt_einsum.contract_expression([2]*(N+1), list(range(N))+[2*N], [2]*(N+1), list(range(N,2*N))+[2*N+1], *tmp0, [2*N+1,2*N])

    def forward(self, return_info:bool=False):
        coeff = self.manifold().to(torch.complex128)
        q0 = torch.stack([self.basis0@coeff, self.basis1@coeff], axis=1)
        lambda_aij = numqi.qec.knill_laflamme_hermite_mul(self.error_torch, q0)
        # tmp0 = (lambda_aij[:,0,0] + lambda_aij[:,1,1]).real/2
        # norm2 = torch.dot(tmp0, tmp0)
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
            info = dict(lambda_aij=lambda_aij.detach().numpy(), q0=q0.detach().numpy(), logicalU=logicalU.detach().numpy(), su2=su2.detach().numpy())
        loss = sum([torch.vdot(x.reshape(-1), x.reshape(-1)).real for x in constraint])
        ret = (loss,info) if return_info else loss
        return ret

model = QECCn23BD2mModel(num_qubit=8, logicalX='XXXXXXXX', BD2m_m=41, tag_real=False)
early_stop_threshold = 1e-7
theta_optim = numqi.optimize.minimize(model, 'uniform', num_repeat=1000, tol=1e-10,
                        early_stop_threshold=early_stop_threshold, print_freq=0)
if theta_optim.fun < early_stop_threshold:
    theta_optim = numqi.optimize.minimize(model, theta_optim.x, num_repeat=1, tol=1e-20, print_freq=500)


if theta_optim.fun < 1e-10:
    coeff = model.manifold().detach().numpy().copy()
    info = model(return_info=True)[1]
    code = info['q0'].T
    if np.abs(coeff.imag).max() > 1e-10:
        coeff = coeff * np.exp(-1j*np.angle(coeff[np.argmax(np.abs(coeff))]))
    if np.abs(code.imag).max() < 1e-10:
        code = code.real
    else:
        print(np.around(coeff, 4))
        code[0] *= np.exp(-1j*np.angle(code[0,0]))
    hf0a = lambda x,eps: np.nonzero(np.abs(code[0]-x)<eps)[0].tolist()
    hf0 = lambda x,eps=1e-3: (hf0a(x,eps), hf0a(-x,eps),hf0a(1j*x,eps),hf0a(-1j*x,eps))
    print(np.sort(np.abs(coeff)))

    # qweA,qweB = numqi.qec.get_weight_enumerator(code, tagB=True)
    # print(qweA, qweB, sep='\n')


# basis0 = np.concatenate([
#     hf_build_basis([1, 2, 4, 88], [49, 50, 52, 104]),
#     hf_build_basis([8, 81, 82, 84, 56, 97, 98, 100], []),
#     hf_build_basis([79, 127], []),
#     hf_build_basis([31], [47]),
# ], axis=0)
# basis1 = (numqi.qec.hf_pauli('X'*7) @ basis0.T).T
# I,X,Y,Z = numqi.gate.I, numqi.gate.X, numqi.gate.Y, numqi.gate.Z
# su2_list = np.stack([I, 1j*Y, -1j*Y, numqi.gate.rz(-np.pi/3), numqi.gate.rz(2*np.pi/3), numqi.gate.rz(2*np.pi/3), numqi.gate.rz(2*np.pi/3)], axis=0)

# N0 = len(basis0)
# error_scipy = numqi.qec.make_pauli_error_list_sparse(7, distance=3, kind='scipy-csr01')[1]
# tmp0 = basis0.conj() @ (error_scipy @ basis1.T).reshape(-1,2**7,N0)
# tmp1 = basis0.conj() @ (error_scipy @ basis0.T).reshape(-1,2**7,N0) - basis1.conj() @ (error_scipy @ basis1.T).reshape(-1,2**7,N0)
# z0 = np.concatenate([tmp0, tmp1], axis=0)
# z0 = (z0 + z0.transpose(0,2,1)) # real coefficient only
# z1 = numqi.qec.pick_indenpendent_vector(z0.reshape(-1,N0*N0), tag_pure_imag=True).reshape(-1,N0,N0)

# ((14,2,3)) BD544


# basis0 = model.basis0.to_dense().T.numpy()
# basis1 = model.basis1.to_dense().T.numpy()
# N = numqi.utils.hf_num_state_to_num_qubit(basis0.shape[1])
# error_str,error_scipy = numqi.qec.make_pauli_error_list_sparse(N, distance=3, kind='scipy-csr01')
# N0 = len(basis0)
# tmp0 = basis0.conj() @ (error_scipy @ basis1.T).reshape(-1,2**N,N0)
# tmp1 = basis0.conj() @ (error_scipy @ basis0.T).reshape(-1,2**N,N0) - basis1.conj() @ (error_scipy @ basis1.T).reshape(-1,2**N,N0)
# z0 = np.concatenate([tmp0, tmp1], axis=0)
# # z0 = (z0 + z0.transpose(0,2,1)) # real coefficient only
# z1 = numqi.qec.pick_indenpendent_vector(z0.reshape(-1,N0*N0), tag_pure_imag=True).reshape(-1,N0,N0)
# z1[np.abs(z1)<1e-10] = 0
# x0 = np.diagonal(z1, axis1=1, axis2=2)

# N = 8
# ind0 = [13,  35,  60,  86, 113, 128, 153, 165, 234]
# basis0 = np.zeros((len(ind0), 2**N), dtype=np.complex128)
# basis0[np.arange(len(ind0)), ind0] = 1
# basis0 = np.stack([basis0, basis0*1j], axis=1).reshape(-1, 2**N)
# ind_mask = [1,3,7,12,14,16]
# ind_kept = np.array(sorted(set(range(18)) - set(ind_mask)))
# basis0 = basis0[ind_kept]
# basis1 = (numqi.qec.hf_pauli('X'*N) @ basis0.T).T
# error_str,error_scipy = numqi.qec.make_pauli_error_list_sparse(N, distance=3, kind='scipy-csr01')
# N0 = len(basis0)
# tmp0 = basis0.conj() @ (error_scipy @ basis1.T).reshape(-1,2**N,N0)
# tmp1 = basis0.conj() @ (error_scipy @ basis0.T).reshape(-1,2**N,N0) - basis1.conj() @ (error_scipy @ basis1.T).reshape(-1,2**N,N0)
# z0 = np.concatenate([tmp0, tmp1], axis=0)
# z0 = (z0 + z0.transpose(0,2,1)) # real coefficient only
# z1 = numqi.qec.pick_indenpendent_vector(z0.reshape(-1,N0*N0), tag_pure_imag=True).reshape(-1,N0,N0)
# z1[np.abs(z1)<1e-10] = 0

# _,i1,i2 = np.nonzero(z1*(1-np.eye(z1.shape[1])))
# i1 = ind_kept[i1]
# i2 = ind_kept[i2]
# print(i1,i2,sep='\n')
# tmp0 = sorted(collections.Counter(i1.tolist()).items(), key=lambda x: -x[1])
# print([x[0] for x in tmp0 if x[1]==tmp0[0][1]])
