import itertools
import numpy as np
import torch
import scipy.sparse

import numqi

def greedy_independent_row_indices(A, tol=1e-10, rtol=None):
    """
    Greedy earliest-row selection of linearly independent rows.

    Scans rows in increasing index order; keeps a row iff its component
    orthogonal to the span of previously kept rows has norm > threshold.

    Parameters
    ----------
    A : array_like, shape (m, n)
    tol : float
        Absolute tolerance on residual norm (used if rtol is None).
    rtol : float or None
        Relative tolerance; if provided, threshold = rtol * ||A||_F.

    Returns
    -------
    idx : np.ndarray (dtype=int)
        Indices of selected rows (in increasing order).
    """
    A = np.asarray(A)
    if A.ndim != 2:
        raise ValueError("A must be a 2D array")
    m, n = A.shape

    # Scale-aware threshold (recommended)
    thresh = (rtol * np.linalg.norm(A, ord="fro")) if (rtol is not None) else tol

    # Orthonormal basis vectors spanning selected rows (stored as columns), shape (n, k)
    Q = np.empty((n, 0), dtype=A.dtype)
    idx = []

    for i in range(m):
        v = A[i].astype(A.dtype, copy=True)  # working vector in R^n / C^n

        # Project out components along existing basis
        if Q.shape[1]:
            v = v - Q @ (Q.conj().T @ v)
            v = v - Q @ (Q.conj().T @ v) #do one re-orthogonalization pass for better stability.

        r = np.linalg.norm(v)
        if r > thresh:
            # Accept: normalize and append to basis
            q = (v / r).reshape(n, 1)
            Q = np.hstack((Q, q))
            idx.append(i)

            # Early stop: can't exceed n independent rows
            if Q.shape[1] == n:
                break

    return np.array(idx, dtype=int)


def scipy_csr_to_torch_sparse_matrix(x0:scipy.sparse.csr_matrix, dtype):
    tmp0 = torch.tensor(x0.indptr, dtype=torch.int64)
    tmp1 = torch.tensor(x0.indices, dtype=torch.int64)
    tmp2 = torch.tensor(x0.data, dtype=dtype)
    ret = torch.sparse_csr_tensor(tmp0, tmp1, tmp2, size=x0.shape, dtype=dtype)
    return ret


def to_full_code_basis(psi_list:list[np.ndarray], n_split_list:list[int])->np.ndarray:
    basis_list = [numqi.dicke.get_dicke_basis(x,dim=2)[::-1] for x in n_split_list]
    ret = []
    n = len(n_split_list)
    for psi in psi_list:
        assert psi.shape==tuple(x+1 for x in n_split_list)
        tmp0 = [y for i,x in enumerate(basis_list) for y in (x, (i,i+n))]
        ret.append(np.einsum(psi, tuple(range(n)), *tmp0, tuple(range(n,2*n)), optimize=True).reshape(-1))
    return ret


class KnillLaflammeTorchOpReal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sym, real):
        tmp0 = (sym @ real).reshape(-1,real.shape[0]) #S*r
        ret = tmp0 @ real #r*S*r
        ctx.save_for_backward(tmp0)
        return ret

    @staticmethod
    def backward(ctx, grad):
        Sr, = ctx.saved_tensors
        grad_real = 2*(grad @ Sr)
        return None, grad_real


class KnillLaflammeTorchOp(torch.autograd.Function):
    @staticmethod
    def forward(ctx, sym, antisym, real, imag):
        n = real.shape[0]
        tmp0 = sym @ real, sym @ imag, antisym @ real, antisym @ imag
        tmp0 = tuple(x.reshape(-1,n) for x in tmp0)
        ret0 = tmp0[0] @ real + tmp0[1] @ imag #r*S*r + i*S*i
        ret1 = tmp0[2] @ imag #i*A*r
        ctx.save_for_backward(*tmp0)
        return ret0,ret1

    @staticmethod
    def backward(ctx, grad0, grad1):
        Sr,Si,Ar,Ai = ctx.saved_tensors
        grad_real = 2*(grad0 @ Sr) - grad1 @ Ai
        grad_imag = 2*(grad0 @ Si) + grad1 @ Ar
        return None, None, grad_real, grad_imag


def to_sparse_csr(np0, zero_eps=1e-5):
    np0 = np0.reshape(-1, np0.shape[-1])
    assert np0.ndim==2
    mask = np.abs(np0)>=zero_eps
    assert np.abs(np0[~mask]).max() < 1e-10
    ind0,ind1 = np.nonzero(mask)
    ret = scipy.sparse.csr_matrix((np0[ind0,ind1], (ind0, ind1)), shape=np0.shape)
    return ret


def split_n_into_partition(n:int, m:int)->list[list[int]]:
    """return all b in Z_+^m with sum(b) = n"""
    # choose positions for the m-1 bars among n + m - 1 slots
    hf0 = lambda x: tuple(a-b-1 for a,b in zip(x[1:],x))
    ret = [hf0((-1,)+divs+(n+m-1,)) for divs in itertools.combinations(range(n + m - 1), m - 1)]
    return ret


def reduce_matrix_subspace(np0:np.ndarray):
    assert np0.ndim==3
    np0 = np0[np.abs(np0).max(axis=(1,2))>1e-10]
    np0 = np0[np.argsort((np.abs(np0)>1e-10).sum(axis=(1,2)))]
    np0 = np0[greedy_independent_row_indices(np0.reshape(np0.shape[0],-1))]
    # np0 = numqi.matrix_space.reduce_vector_space(np0.reshape(np0.shape[0],-1), zero_eps=1e-10).reshape(-1,np0.shape[1],np0.shape[1])
    return np0
