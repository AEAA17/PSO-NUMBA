import numpy as np
from numba import njit, objmode

@njit
def pso_numba(func, lb, ub, swarmsize=45, omega0=1.0, omegaf=0.5, phip=1.5, phig=1.5, maxiter=100):
    """
    Otimizador Estocástico Particle Swarm Optimization (PSO)
    Acelerado via LLVM/C com Numba.
    """
    S = swarmsize
    D = len(lb)
    
    x = np.zeros((S, D))
    v = np.zeros((S, D))
    p = np.zeros((S, D))
    fx = np.zeros(S)
    
    fp = np.ones(S) * 1e15
    g = np.zeros(D)
    fg = 1e15
    fg_anterior = 1e15
    
    vhigh = np.abs(ub - lb)
    vlow = -vhigh
    
    # Inicialização das Partículas
    for i in range(S):
        for d in range(D):
            x[i, d] = lb[d] + np.random.rand() * (ub[d] - lb[d])
            v[i, d] = vlow[d] + np.random.rand() * (vhigh[d] - vlow[d])
            p[i, d] = x[i, d]
            
    # Primeira Avaliação
    for i in range(S):
        fx[i] = func(x[i, :])
        if fx[i] < fp[i]:
            fp[i] = fx[i]
            if fp[i] < fg:
                fg = fp[i]
                for d in range(D):
                    g[d] = p[i, d]
    
    contador_estagnacao = 0
    
    # Loop Principal
    for it in range(1, maxiter + 1):
        omega = omega0 + it * (omegaf - omega0) / maxiter
        
        for i in range(S):
            for d in range(D):
                rp = np.random.rand()
                rg = np.random.rand()
                
                v[i, d] = omega * v[i, d] + phip * rp * (p[i, d] - x[i, d]) + phig * rg * (g[d] - x[i, d])
                x[i, d] = x[i, d] + v[i, d]
                
                # Correção de Limites
                if x[i, d] < lb[d]:
                    x[i, d] = lb[d]
                    v[i, d] = 0.0
                elif x[i, d] > ub[d]:
                    x[i, d] = ub[d]
                    v[i, d] = 0.0
            
            fx[i] = func(x[i, :])
            
            # Atualiza Personal e Global Best
            if fx[i] < fp[i]:
                fp[i] = fx[i]
                for d in range(D):
                    p[i, d] = x[i, d]
                    
                if fp[i] < fg:
                    fg = fp[i]
                    for d in range(D):
                        g[d] = p[i, d]
        
        # Log nativo protegido
        with objmode():
            print("Iteracao:", it, "| Melhor Fobj:", fg)

        # Lógica de Parada Baseada EXCLUSIVAMENTE na estagnação
        if fg < fg_anterior - 1e-4:
            fg_anterior = fg
            contador_estagnacao = 0
        else:
            contador_estagnacao += 1
            
        # Deixa o PSO explorar por pelo menos 30 iterações travadas antes de abortar
        if contador_estagnacao >= 30: 
            with objmode():
                print("Busca encerrada: Nenhuma melhora significativa nas ultimas 30 iteracoes.")
            break

    return g, fg