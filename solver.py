from pa_lu import pa_lu
import numpy as np
import numpy.linalg as la
from copy import deepcopy
from random import randint

    
# solucionar A*x = d
# No caso fechado, d (demand_vector) é igual a 0 e, no aberto, é um input do problema
def solve_system(dim, consume_matrix: np.ndarray, demand_vector = None):
    if demand_vector is None:
        demand_vector = np.array([0.0]*dim)
    A = np.identity(dim) - consume_matrix
    
    # 1) Decompor em PA = LU
    (P,A,L,U) = pa_lu(dim, A)
    
    # 2) Solucionar LU * x = P * d, que possui a mesma solução
    Pd = np.matmul(P, demand_vector)
    
    # 2.1) Solucionar L * y = Pd
    # Note que L é sempre li
    y = []
    for i in range(0,dim):
        pivot = L[i,i]
        sum = 0
        for j in range(0,i):
            sum += L[i, j]*y[j]
        y.append((Pd[i] - sum)/pivot)
        
    # test code
    # print("P = ", P)
    # print("LU = ", np.matmul(L,U))
    # print("L = ", L)
    # print("U = ", U)
    # print("y = ", y)
    Ly = np.matmul(L,y)
    for i in range(0, dim):
        assert abs(Ly[i] - Pd[i]) < 0.001, (Ly[i], Pd[i])
        
    # 2.2) Solucionar U * x = y
    x = [0.0]*dim
    # soluções para cada célula de x, em listas
    # ex: solutions[2] = [1,2,3] é o mesmo que x[2] = 1 + 2*a + 3*b
    solutions = dict()
    # quantidade de parâmetros já criados na solução
    param_count = 0
    for i in reversed(range(0,dim)):
        pivot = U[i,i]
        # Considerar valores próximos de zero como parâmetros livres
        if pivot < 0.0000001 and pivot > -0.0000001:
            param_count += 1
            solutions[i] = [0.0]*(dim+1)
            solutions[i][param_count] = 1.0
        else:
            current_sum = [0]*(dim+1)
            # Calcula a fórmula para cada variável, considerando as
            # parametrizações das anteriores
            for j in range(i+1, dim):
                add: list[float] = solutions[j]
                for k in range(0, len(add)):
                    current_sum[k] += -U[i,j]*add[k]/pivot
            # Adicionar fator do y, se diferente de zero
            current_sum[0] += y[i]/pivot
            
            # Atualizar dicionário de soluções para o próximo pivô
            solutions[i] = current_sum
            x[i] = current_sum[0]
            
    # Parametrização (sistema LD)
    count = 0
    lds = []
    for i in range(1, param_count+1):
        vec = np.array([0.0]*dim)
        for j in solutions.keys():
            vec[j] = solutions[j][i]
        lds.append(np.array(vec))
        count += 1
        
    # Testar
    # print("x = ",x)
    # print(lds)
    _d = np.matmul(A,x)
    # print("d = ", demand_vector)
    # print("_d = ",_d)
    for i in range(0, dim):
        # Isso vai falhar se o sistema não tiver solução
         assert abs(demand_vector[i] - _d[i]) < 0.001, (demand_vector[i], _d[i])
    x2 = np.array(deepcopy(x))
    for ld in lds:
        x2 += randint(-50,50)*ld
    # print("x2 = ", x2)
    _d2 = np.matmul(A,x2)
    # print(_d2)
    for i in range(0, dim):
         assert abs(demand_vector[i] - _d2[i]) < 0.001, (demand_vector[i], _d2[i])
         
    return(x, lds)
    
            
        
    