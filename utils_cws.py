import itertools
import numpy as np
import numqi


def get_partition(n:int, start:int=1, min_:int=1):
    # n=a+b+c+... where start<=a<=b<=c<=...
    assert (start>=1) and (n>=start) and (min_ in {1,2})
    ret = []
    if n>=(3*start):
        ret += [(x,)+y for x in range(start,n//3+1) for y in get_partition(n-x,x,2)]
    if (n>=(2*start)) and min_<=2:
        ret += [(x,n-x) for x in range(start,n//2+1)]
    if min_==1:
        ret += [(n,)]
    return ret

# https://www.ii.uib.no/~larsed/entanglement/
# https://arxiv.org/abs/quant-ph/0602096 table V
_inequivalent_graph_dict = {
    1: [[]], #No. 0
    2: [[(0,1)]], #No. 1
    3: [[(0,1),(0,2)]], #No. 2
    4: [
        [(0,1),(0,2),(0,3)], #No. 3
        [(0,1),(1,2),(2,3)],
    ],
    5: [
        [(0,1),(0,2),(0,3),(0,4)], #No. 5
        [(0,1),(1,2),(1,4),(2,3)],
        [(0,1),(1,2),(2,3),(3,4)],
        [(0,1),(1,2),(2,3),(3,4),(4,0)],
    ],
    6: [
        [(0,1),(0,2),(0,3),(0,4),(0,5)], #n=6, No. 9
        [(0,5),(1,5),(2,5),(3,4),(4,5)], #n=6, No. 10
        [(0,5),(1,5),(2,4),(3,4),(4,5)],
        [(0,1),(1,2),(1,5),(2,3),(3,4)],
        [(0,1),(1,2),(2,3),(2,5),(3,4)],
        [(0,1),(1,2),(2,3),(3,4),(4,5)],
        [(0,5),(1,3),(2,3),(2,5),(3,4),(4,5)], #n=6, No. 15
        [(0,1),(1,2),(1,3),(2,3),(2,5),(3,4)],
        [(0,1),(0,4),(0,5),(1,2),(2,3),(3,4)],
        [(0,1),(0,5),(1,2),(2,3),(3,4),(4,5)],
        [(0,1),(0,2),(0,5),(1,2),(1,4),(2,3),(3,4),(3,5),(4,5)],
    ],
    7: [
        [(0,1),(0,2),(0,3),(0,4),(0,5),(0,6)], #n=7, No. 20
        [(0,6),(1,6),(2,6),(3,6),(4,5),(5,6)],
        [(0,6),(1,6),(2,6),(3,5),(4,5),(5,6)],
        [(0,6),(1,6),(2,6),(3,4),(4,5),(5,6)],
        [(0,6),(1,6),(2,4),(3,4),(4,5),(5,6)],
        [(0,1),(0,6),(2,6),(3,6),(4,5),(5,6)], #n=7, No. 25
        [(0,6),(1,6),(2,5),(3,4),(4,5),(5,6)],
        [(0,1),(1,2),(1,6),(2,3),(3,4),(4,5)],
        [(0,1),(1,2),(2,3),(2,4),(4,5),(5,6)],
        [(0,1),(1,2),(2,3),(2,5),(3,4),(5,6)],

        [(0,1),(1,2),(2,3),(3,4),(4,5),(5,6)], #n=7, No. 30
        [(0,2),(1,2),(2,3),(2,5),(3,4),(4,5),(4,6)],
        [(0,6),(1,6),(2,5),(3,4),(4,5),(4,6),(5,6)],
        [(0,2),(1,2),(2,3),(2,6),(3,4),(4,5),(5,6)],
        [(0,3),(1,2),(2,3),(2,5),(3,4),(4,5),(5,6)],
        [(0,5),(1,2),(2,3),(2,6),(3,4),(4,5),(5,6)], #n=7, No. 35
        [(0,1),(1,2),(2,3),(2,4),(3,4),(3,6),(4,5)],
        [(0,6),(1,2),(2,3),(2,6),(3,4),(4,5),(5,6)],
        [(0,1),(0,5),(1,2),(2,3),(3,4),(4,5),(5,6)],
        [(0,1),(0,4),(1,2),(2,3),(3,4),(4,5),(5,6)],

        [(0,1),(0,6),(1,2),(2,3),(3,4),(4,5),(5,6)], #n=7, No. 40
        [(0,1),(0,4),(0,5),(1,2),(2,3),(3,4),(4,5),(5,6)],
        [(0,2),(0,6),(1,2),(1,5),(2,3),(3,4),(4,5),(5,6)],
        [(0,1),(0,3),(0,6),(1,2),(2,3),(2,5),(3,4),(4,5),(4,6)],
        [(0,3),(0,6),(1,2),(1,6),(2,3),(2,4),(3,4),(4,5),(5,6)],
        [(0,1),(1,2),(1,4),(1,6),(2,3),(2,6),(3,4),(3,5),(4,5),(5,6)], #n=7, No. 45
    ],
}


def get_all_CWS_K2_code(num_qubit:int, distance:int):
    # https://arxiv.org/abs/0803.3232v1
    assert (2<=num_qubit<=7) and (distance>=2)
    all_graph_list = dict()
    for partition in get_partition(num_qubit):
        tmp0 = [range(len(_inequivalent_graph_dict[x])) for x in partition]
        for ind0 in list(itertools.product(*tmp0)):
            edge = []
            for ind1 in range(len(partition)):
                x0 = sum(partition[:ind1])
                edge += [(y0+x0,y1+x0) for y0,y1 in _inequivalent_graph_dict[partition[ind1]][ind0[ind1]]]
            all_graph_list[partition,ind0] = edge
    op_list = numqi.qec.make_pauli_error_list_sparse(num_qubit=num_qubit, distance=distance, kind='scipy-csr01')[1]
    ret = []
    for (partition,index_graph),edge in all_graph_list.items():
        adj_matrix = np.zeros((sum(partition),sum(partition)), dtype=np.uint8)
        for x,y in edge:
            adj_matrix[x,y] = 1
            adj_matrix[y,x] = 1
        code0 = numqi.sim.build_graph_state(adj_matrix)

        for indz in range(1,2**num_qubit):
            indz = format(indz, f'0{num_qubit}b')
            tmp0 = [i for i,x in enumerate(indz) if x=='1']
            code1 = code0
            for x in tmp0:
                code1 = numqi.sim.state.apply_gate(code1, numqi.gate.Z, x)
            code = np.stack([code0, code1], axis=0)
            z0 = code.conj() @ (op_list @ code.T).reshape(-1,2**num_qubit,2)
            if (np.abs(z0[:,0,1]).max() < 1e-12) and (np.abs(z0[:,0,0] - z0[:,1,1]).max() < 1e-12):
                ret.append(dict(adj_matrix=adj_matrix, codeword=indz, index_graph=list(zip(partition,index_graph)),
                        lambda_ai=z0[:,0,0].real, code=code))
    return ret


def demo_cws_transversal_group():
    z0 = get_all_CWS_K2_code(7, 3)
    z1 = []
    for ind0 in range(len(z0)):
        print(ind0)
        code = z0[ind0]['code']
        # logical_list = numqi.qec.get_transversal_group(code, num_round=100, tag_print=False)
        # info1 = numqi.qec.get_transversal_group_info([x[0] for x in logical_list])
        info1 = None
        qweA,qweB = numqi.qec.get_weight_enumerator(code, tagB=True)
        if qweA[1].max() < 1e-4:
            a = qweA[2]
            if np.abs(qweA-np.array([1,0,a,0,21-2*a,0,42+a,0])).max() > 1e-10:
                print(ind0, 'ERROR')
            else:
                print(ind0, 'OK')
        # print(np.around(qweA, 3))
        z1.append(dict(ind0=ind0, code=code, info1=info1, qweA=qweA, qweB=qweB))
