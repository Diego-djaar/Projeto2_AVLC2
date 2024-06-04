import numpy as np
import numpy.linalg as la
from pa_lu import pa_lu
from solver import solve_system

def np_format_matrix(matrix_string: str, demand_string: str):
    # Recebe uma matrix KxK em que cada entrada não negativa Kij representa a fração
    # da produção total da j-ésima indústria que é comprada pela i-ésima indústria.

    # Ou seja, os valores de cada coluna somam 1, representando o total da produção da
    # respectiva indústria (No caso do modelo fechado, isto é, o produto é trocado 
    # apenas entre as indústrias)
    lines = matrix_string.split('\n')
    matrix = list(map(lambda y: list(map(lambda z: float(z),y)), map(lambda x: x.split(' '), lines)))
    if demand_string == "":
        return (np.array(matrix), None)
    
    demand = list(map(lambda x : float(x),demand_string.split(' ')))
    return(np.array(matrix), demand)

#Deve-se montar o sistema linear (PA=LU) que encontra o vetor-preço de tal forma que
#as indústrias igualem seus gastos aos seus ganhos, e exibe este vetor. 

# É preciso solucionar a equação (para o sistema fechado):
# (I - E)*p = 0
# Sendo E a matriz de entrada e p o vetor-preço que se quer encontrar

# Cada valor de p é a receita de cada indústria, de tal forma que é igualado a seu
# gasto, resultando, portanto, no equilíbrio do sistema
def main():
    file = open("input_matrix", "r")
    input_string = file.read()
    file2 = open("demand_vector", "r")
    input2_string = file2.read()
    (np_matrix, demand) = np_format_matrix(input_string, input2_string)
    dim = len(np_matrix[0])
    
    # Prototype solution
    (x, lds) = solve_system(dim, np_matrix, demand)
    print(f"solução geral: ")
    print(f"{x}", end='')
    count = 0
    for ld in lds:
        print(f" + {chr(ord('a')+count)}*{ld}", end='')
        count += 1
    print()


if __name__ == "__main__":
    main()