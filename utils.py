import functools
import itertools
import numpy as np
import scipy.special
import scipy.linalg
import scipy.optimize
import torch
import opt_einsum
import numqi
import cvxpy

hfb = lambda x,n=7: bin(x)[2:].rjust(n,'0') #TODO
hf_kron = lambda *x: functools.reduce(np.kron, x)


def get_rotation_axis(np0, kind='rad', zero_eps=1e-10):
    assert (np0.ndim in {2,3}) and (np0.shape[-1]==2) and (np0.shape[-2]==2)
    isone = np0.ndim==2
    if isone:
        np0 = np0.reshape(1,2,2)
    tmp0 = np0[:,0,0]*np0[:,1,1] - np0[:,0,1]*np0[:,1,0]
    assert np.abs(np.abs(tmp0)-1).max() < zero_eps
    np0 = np0 * np.exp(-1j*np.angle(tmp0)/2).reshape(-1,1,1)
    theta = 2*np.arccos((np0[:,0,0]+np0[:,1,1]).real/2) #[0,2pi)
    tmp0 = np.stack([-(np0[:,1,0]+np0[:,0,1]).imag, (np0[:,1,0]-np0[:,0,1]).real, -(np0[:,0,0]-np0[:,1,1]).imag], axis=1)
    tmp1 = np.linalg.norm(tmp0, axis=1)
    ind0 = tmp1<zero_eps
    axis = np.zeros((np0.shape[0], 3), dtype=np.float64)
    if np.any(ind0):
        axis[ind0] = np.array([1,0,0])
    ind0 = np.logical_not(ind0)
    if np.any(ind0):
        axis[ind0] = tmp0[ind0] / tmp1[ind0].reshape(-1,1)
    if kind=='deg':
        theta = np.rad2deg(theta) #[0,360)
    if isone:
        axis = axis[0]
        theta = theta[0]
    return axis,theta


class QECCn23TransversalGateModel(torch.nn.Module):
    def __init__(self, num_qubit:int, gate_list:np.ndarray, tag_phase:bool=False, tag_same_su2:bool=False, tag_real_stiefel:bool=False):
        super().__init__()
        if gate_list.ndim==2:
            gate_list = gate_list.reshape(1,2,2)
        assert (gate_list.shape[1]==2) and (gate_list.shape[2]==2)
        assert np.abs(gate_list @ gate_list.transpose(0,2,1).conj() - np.eye(2)).max() < 1e-10
        tmp0 = gate_list[:,0,0]*gate_list[:,1,1]-gate_list[:,0,1]*gate_list[:,1,0]
        gate_list = np.exp(-1j*np.angle(tmp0)/2).reshape(-1,1,1) * gate_list #SU(2) remove global phase
        self.num_qubit = num_qubit
        self.error_torch = numqi.qec.make_pauli_error_list_sparse(num_qubit, distance=3, kind='torch-csr01')[1].to(torch.complex128)
        self.manifold = numqi.manifold.Stiefel(2**num_qubit, rank=2, dtype=(torch.float64 if tag_real_stiefel else torch.complex128))
        self.tag_same_su2 = tag_same_su2
        tmp0 = gate_list.shape[0] * (1 if tag_same_su2 else num_qubit)
        self.manifold_su2 = numqi.manifold.SpecialOrthogonal(2, batch_size=tmp0, dtype=torch.complex128)
        self.gate_list = torch.tensor(gate_list, dtype=torch.complex128)
        self.tag_phase = tag_phase
        if tag_phase:
            self.theta_phase = torch.nn.Parameter(torch.randn(gate_list.shape[0], dtype=torch.float64))

        N0,N1 = num_qubit, gate_list.shape[0]
        tmp0 = [y for x in range(N0) for y in [(N1,2,2), (2*N0+2,N0+x,x)]]
        self.contract_expr = opt_einsum.contract_expression([2]*(N0+1), list(range(N0))+[2*N0], [2]*(N0+1), list(range(N0,2*N0))+[2*N0+1], *tmp0, [2*N0+2,2*N0+1,2*N0])
        self.lambda_target = None

    def set_lambda_target(self, x):
        if (x is None) or (x=='min') or (x=='max'):
            self.lambda_target = x
        else:
            self.lambda_target = torch.tensor(float(x), dtype=torch.float64)

    def forward(self, return_info:bool=False):
        q0 = self.manifold().to(torch.complex128)
        lambda_aij = numqi.qec.knill_laflamme_hermite_mul(self.error_torch, q0)
        # lambda_aij = q0.T.conj() @ (self.error_torch @ q0).reshape(-1, *q0.shape)
        tmp0 = self.manifold_su2()
        su2 = [tmp0.reshape(-1,2,2)]*self.num_qubit if self.tag_same_su2 else tmp0.reshape(self.num_qubit,-1,2,2)
        tmp1 = q0.reshape([2]*(self.num_qubit+1))
        logicalU = self.contract_expr(tmp1, tmp1.conj(), *su2)
        tmp0 = (lambda_aij[:,0,0].real + lambda_aij[:,1,1].real)/2
        norm2 = torch.dot(tmp0, tmp0)
        constraint = [
            lambda_aij[:,0,1],
            lambda_aij[:,0,0].real-lambda_aij[:,1,1].real,
            logicalU - (self.gate_list*torch.exp(1j*self.theta_phase).reshape(-1,1,1) if self.tag_phase else self.gate_list),
        ]
        if return_info:
            tmp0 = (su2[0] if self.tag_same_su2 else su2).detach().numpy()
            info = dict(lambda_aij=lambda_aij.detach().numpy(), q0=q0.detach().numpy(), logicalU=logicalU.detach().numpy(), su2=tmp0)
        if (self.lambda_target is None) or isinstance(self.lambda_target, torch.Tensor):
            loss = sum([torch.vdot(x.reshape(-1), x.reshape(-1)).real for x in constraint])
            if self.lambda_target is not None:
                loss = loss + (norm2-self.lambda_target)**2
            ret = (loss,info) if return_info else loss
        else:
            assert self.lambda_target in {'min','max'}
            loss = norm2 if (self.lambda_target=='min') else -norm2
            ret = (loss, constraint, info) if return_info else (loss, constraint)
        return ret


def rand_local_unitary(psi, dim=2, seed=None):
    np_rng = numqi.random.get_numpy_rng(seed)
    shape = psi.shape
    N0 = round(np.log(shape[-1]) / np.log(dim))
    assert shape[-1]==dim**N0
    psi = psi.reshape(-1, shape[-1])
    matU = list(numqi.random.rand_haar_unitary(dim, batch_size=N0, seed=np_rng))
    tmp0 = [y for i,x in enumerate(matU) for y in [x,[i+1,N0+i+1]]]
    tmp1 = psi.reshape([psi.shape[0]] + [dim]*N0)
    ret = np.einsum(tmp1, list(range(N0+1)), *tmp0, [0]+list(range(N0+1,2*N0+1)), optimize=True).reshape(shape)
    return ret


def get_su2_euler_formula(axis, theta):
    sx = np.array([[0,1],[1,0]])
    sy = np.array([[0,-1j],[1j,0]])
    sz = np.array([[1,0],[0,-1]])
    axis = np.asarray(axis)
    assert axis.shape[-1] == 3
    axis = axis / np.linalg.norm(axis, axis=-1, keepdims=True)
    theta = np.asarray(theta)
    isone = (theta.ndim==0)
    nx,ny,nz = axis.T.reshape(3,-1,1,1)
    ret = scipy.linalg.expm(-(1j*theta.reshape(-1,1,1)/2)*(nx*sx+ny*sy+nz*sz))
    if isone:
        ret = ret[0]
    return ret


class SpecialUnitary2XZManifold(torch.nn.Module):
    def __init__(self, batch_size:int):
        super().__init__()
        self.sphere = numqi.manifold.Sphere(2, batch_size=batch_size, dtype=torch.float64)
        self.theta = torch.nn.Parameter(torch.randn(batch_size, dtype=torch.float64))
        self.torchXi = torch.tensor(numqi.gate.X*1j, dtype=torch.complex128)
        self.torchZi = torch.tensor(numqi.gate.Z*1j, dtype=torch.complex128)
        self.torchI = torch.eye(2, dtype=torch.complex128)

    def forward(self):
        ct = torch.cos(self.theta)
        st = torch.sin(self.theta)
        xy = self.sphere()
        tmp0 = xy[:,0].reshape(-1,1,1)*self.torchXi + xy[:,1].reshape(-1,1,1)*self.torchZi
        ret = ct.reshape(-1,1,1)*self.torchI + st.reshape(-1,1,1)*tmp0
        return ret


## cvxpy is slow in this case
# import functools
# import cvxpy
# @functools.lru_cache
# def _get_cvxpy_LP_prob(n:int, lenBsi:int):
#     cvxX = cvxpy.Variable(lenBsi)
#     cvxBsi = cvxpy.Parameter((lenBsi, n))
#     constraint = [
#         cvxX>=0,
#         cvxpy.sum(cvxX)==1,
#         cvxX @ cvxBsi==1/2,
#     ]
#     prob = cvxpy.Problem(cvxpy.Minimize(0), constraint)
#     return cvxBsi, prob

def _BD_group_LP(bsi:np.ndarray):
    s,n = bsi.shape
    tmp0 = np.concatenate([bsi.T, np.ones((1, s))], axis=0)
    tmp1 = np.ones(n+1)
    tmp1[:-1] *= 0.5
    tmp2 = [(0, 1)]*s
    res = scipy.optimize.linprog(np.zeros(s), A_eq=tmp0, b_eq=tmp1, bounds=tmp2, options={'disp': False})
    return (res.success and res.status == 0)


# def _search_veca_BD_group_recursion(a, idx, min_val, rem, _constant):
#     ret = []
#     n = _constant['n']
#     if idx==(n-1):
#         assert rem>=min_val
#         ind0 = _constant['ind0']
#         a[idx] = rem
#         ind1 = (ind0 @ np.array(a))%_constant['m']==0
#         if (ind1.sum() >= _constant['min_term']) and _BD_group_LP(ind0[ind1]):
#             ret = [a.copy()]
#             if _constant['tag_print']:
#                 print(a)
#     else:
#         for v in range(int(rem / (n-idx)), min_val-1, -1):
#             a[idx] = v
#             ret += _search_veca_BD_group_recursion(a, idx+1, v, rem - v, _constant)
#     return ret


# def search_veca_BD_group(n:int, m:int, tag_print=True, min_value:int=0, sum_value:int|None=None, min_term=None):
#     """
#     non-decreasing integer lists a of length n summing to 2*m-1 pass BD group
#     """
#     if sum_value is None:
#         sum_value = 2*m-1
#     if min_term is None:
#         min_term = n+1
#     ind0 = np.array(list(itertools.product([0,1], repeat=n)))
#     _constant = {'ind0': ind0, 'n':n, 'm':m, 'tag_print':tag_print, 'min_term':min_term}
#     ret = _search_veca_BD_group_recursion([0]*n, 0, min_value, sum_value, _constant)
#     return ret


def search_veca_BD_group(n:int, m:int, tag_print=True, min_value:int=0, k:int|None=2, min_term=None):
    if min_term is None:
        min_term = n
    assert min_term>=2
    ind0 = np.array(list(itertools.product([0,1], repeat=n)), dtype=np.int32)
    if k is None:
        tmp0 = (x for x in itertools.combinations_with_replacement(range(min_value, m), n) if (sum(x)+1)%m==0)
    else:
        assert k>=2
        tmp0 = (x for x in itertools.combinations_with_replacement(range(min_value, m), n) if sum(x)==(k*m-1))
    ret = []
    for a in tmp0:
        ind1 = (ind0 @ np.array(a))%m==0
        if (ind1.sum()>=min_term) and _BD_group_LP(ind0[ind1]): #TODO, this results requires bsi has rank n
            ret.append(a)
            if tag_print:
                print(a)
    return ret

def _C_group_LP(bsi0, bsij0, bsi1, bsij1):
    s,n = bsi0.shape
    s1,_ = bsi1.shape
    tmp0 = np.concatenate([bsi0.T, -bsi1.T], axis=1)
    tmp0b = np.zeros(n)
    tmp1 = np.concatenate([bsij0.T, -bsij1.T], axis=1)
    tmp1b = np.zeros(bsij0.shape[1])
    tmp2 = np.stack([np.concatenate([np.ones(s),np.zeros(s1)], axis=0), np.concatenate([np.zeros(s),np.ones(s1)], axis=0)], axis=0)
    tmp2b = np.ones(2)
    tmp3 = np.concatenate([tmp0, tmp1, tmp2], axis=0)
    tmp3b = np.concatenate([tmp0b, tmp1b, tmp2b], axis=0)
    tmp4 = [(0,1)]*(s+s1)
    res = scipy.optimize.linprog(np.zeros(s+s1), A_eq=tmp3, b_eq=tmp3b, bounds=tmp4, method='highs', options={'disp': False, 'presolve':True})
    return (res.success and res.status == 0)


def search_veca_C_group(n:int, m:int, tag_print=True, min_value:int=0, min_term=None):
    if min_term is None:
        min_term = n
    assert min_term>=2
    ind0 = np.array(list(itertools.product([0,1], repeat=n)), dtype=np.int32)
    i0,i1 = np.triu_indices(n,1)
    ind0ij = ind0[:,i0]*ind0[:,i1]
    ret = []
    for a in itertools.combinations_with_replacement(range(min_value, m), n):
        tmp0 = (ind0 @ np.array(a, dtype=np.int32))%m
        ind1 = tmp0==0
        ind2 = tmp0==(m-1)
        if (ind1.sum() >= min_term) and (ind2.sum() >= min_term) and _C_group_LP(ind0[ind1], ind0ij[ind1], ind0[ind2], ind0ij[ind2]):
            ret.append(tuple(a))
            if tag_print:
                print(a)
    return ret
