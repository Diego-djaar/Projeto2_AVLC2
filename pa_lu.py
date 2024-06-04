import numpy as np
import numpy.linalg as la
from copy import deepcopy
from enum import Enum

class row_op_kind(Enum):
    RowAdd = 1
    RowSwap = 2
    
class row_op:
    output_row: int
    scalar: float
    input_row: int
    kind: row_op_kind
    
    def __repr__(self):
        # TODO: repr
        return f"({self.input_row}, {self.output_row}, {self.kind}, {self.scalar})"
    
    def __init__(self, output_row, input_row, kind, scalar = None):
        self.output_row = output_row
        self.input_row = input_row
        self.kind = kind
        self.scalar = scalar
        
# Decompor A em PA = LU
def pa_lu(dim, A: np.ndarray):
    L = np.identity(dim)
    P = np.identity(dim)
    U = deepcopy(A)
    ops_list: list[row_op] = []
    for i in range(0,dim):
        # Selecionar o valor na célula Ui,i para verificar se pode ser o pivô
        if U[i,i] == 0:
            # Nesse caso, tentar trocar com outra fileira seguinte
            for j in range(i+1, dim):
                if U[j,i] != 0:
                    ops_list.append(row_op(i, j, row_op_kind.RowSwap))
                    
                    # Atualiza as operações de adicionar fileiras anteriores
                    # para corresponder a troca de fileiras
                    for op in ops_list:
                        if op.kind == row_op_kind.RowAdd:
                            if op.input_row == i:
                                op.input_row = j
                            elif op.input_row == j:
                                op.input_row = i
                            if op.output_row == i:
                                op.output_row = j
                            elif op.output_row == j:
                                op.output_row = i
                    
                    # Troca as fileiras
                    temp = deepcopy(U[i])
                    U[i] = U[j]
                    U[j] = temp
                    break
            # Agora, ou essa fileira tem um pivô, ou é zero
            # Se for, se trata de um sistema LD. Pode-se ignorar a fileira
            if U[i,i] == 0:
                continue
        # Faz eliminação gaussiana das fileiras seguintes, adicionando na lista de operações
        for j in range(i+1, dim):
            if U[j,i] != 0:
                scalar_factor = -U[j,i]/U[i,i]
                ops_list.append(row_op(j, i, row_op_kind.RowAdd, scalar_factor))
                
                # Adiciona a fileira na outra
                U[j] += scalar_factor * U[i]   
    # Aplica as operações de swap em P e de adicionar em L, ao contrário
    for op in ops_list:
        if op.kind == row_op_kind.RowSwap:
            i = op.input_row
            j = op.output_row
            temp = deepcopy(P[i])
            P[i] = P[j]
            P[j] = temp
    # TODO: consertar isso
    for op in reversed(ops_list):
        if op.kind == row_op_kind.RowAdd:
            i = op.input_row
            j = op.output_row
            scalar = op.scalar
            L[j] -= scalar * L[i]
            
    #test code
    PA = np.matmul(P,A)
    LU = np.matmul(L,U)
    
    
    # print("P = ", P)
    # print("A = ", A)
    # print("PA = ", PA)
    # print("LU = ", LU)
    # print("L = ", L)
    # print("U = ", U)
    
    for i in range(0, dim):
        for j in range(0,dim):
            assert abs(PA[i,j] - LU[i,j]) < 0.001, (PA[i,j],LU[i,j])
    
    return (P,A,L,U)