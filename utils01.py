import numpy as np
import scipy.linalg
import numqi

from utils import get_rotation_axis, hfb, hf_kron


def hfb(x:int|list[int], n=7):
    if not hasattr(x,'__len__'):
        ret = bin(x)[2:].rjust(n,'0')
    else:
        ret = ' '.join([bin(x)[2:].rjust(n,'0') for x in x])
    return ret


def hf_theta_round(x, eps=1e-2):
    x0 = np.around(x).astype(np.int64)
    assert np.abs(x0-x).max() < eps
    ret = np.sort(np.abs(x0))
    return ret


def rotate_index(z0, ind0):
    ind0 = tuple(int(x) for x in ind0)
    num_qubit = len(ind0)
    ret = []
    if any((x==-1) for x in ind0):
        assert all((x==1 or x==-1) for x in ind0)
        hff = lambda a: ('1' if a=='0' else '0')
        for x in z0:
            tmp0 = hfb(x, num_qubit)
            ret.append(int(''.join([(hff(y0) if (y1==-1) else y0) for y0,y1 in zip(tmp0,ind0)]), base=2))
    else:
        assert set(ind0)==set(range(num_qubit))
        for x in z0:
            tmp0 = hfb(x, num_qubit)
            ret.append(int(''.join(tmp0[y] for y in ind0), base=2))
    print(','.join(str(x) for x in ret))
    return ret


def search_2I_code_transversalR(code):
    I,X,Y,Z = numqi.gate.I, numqi.gate.X, numqi.gate.Y, numqi.gate.Z
    hfR = lambda a,b,t=1: I*np.cos(t*np.pi/5) + 1j*np.sin(t*np.pi/5)/np.sqrt(5) * (a*Y + b*Z)
    model = numqi.qec.SearchTransversalGateModel(code)
    transR = hfR(-2,1)
    tmp0 = dict(theta0='uniform', num_repeat=50, tol=1e-7, early_stop_threshold=1e-4, print_every_round=0)
    model.set_target_gate(transR)
    theta_optim = numqi.optimize.minimize(model, **tmp0)
    if theta_optim.fun > 1e-7:
        transR = hfR(2,1)
        model.set_target_gate(transR)
        theta_optim = numqi.optimize.minimize(model, **tmp0)
        assert theta_optim.fun < 1e-7
        print('transR: R(2,1)')
    else:
        print('transR: R(-2,1)')
    theta_optim = numqi.optimize.minimize(model, theta_optim.x, num_repeat=1, tol=1e-30, print_every_round=0)
    assert theta_optim.fun < 1e-12
    su2 = model(return_info=True)[1]
    assert np.abs(np.trace(su2 @ X, axis1=1, axis2=2)).max() < 1e-5
    ct = np.trace(su2, axis1=1, axis2=2).real/2
    tmp0 = np.arccos(ct)/np.pi*5
    assert np.abs(tmp0-np.around(tmp0,3).astype(np.int64)).max() < 1e-5
    tmp0 = np.around(tmp0,3).astype(np.int64)
    a = np.trace(su2 @ Y, axis1=1, axis2=2).imag/2 / np.sin(tmp0*np.pi/5) * np.sqrt(5)
    b = np.trace(su2 @ Z, axis1=1, axis2=2).imag/2 / np.sin(tmp0*np.pi/5) * np.sqrt(5)
    assert np.abs(a-np.around(a,3).astype(np.int64)).max() < 1e-5
    assert np.abs(b-np.around(b,3).astype(np.int64)).max() < 1e-5
    a = np.around(a,3).astype(np.int64)
    b = np.around(b,3).astype(np.int64)
    print(list(zip(a.tolist(), b.tolist(),tmp0.tolist())))
