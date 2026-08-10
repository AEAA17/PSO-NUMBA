# PSO_NUMBA
A função `pso_numba` implementa o algoritmo de **Otimização por Enxame de Partículas (PSO - Particle Swarm Optimization)**. O código foi inteiramente estruturado e tipado para ser compilado *Just-In-Time* (JIT) pela biblioteca Numba, operando em nível de máquina (LLVM/C). Isso remove o *overhead* do Python puro, acelerando drasticamente a busca global em superfícies de erro complexas ou sistemas de equações diferenciais rígidas.

### Parâmetros de Entrada (Inputs)

| Parâmetro | Tipo | Padrão | Descrição |
| --- | --- | --- | --- |
| **`func`** | `function` | - | Função objetivo a ser minimizada. Deve ser compilada em Numba (`@njit`) ou utilizar a ponte `objmode` para rotinas em Python puro.
| **`lb`** | `np.array` | - | Limites inferiores (*lower bounds*) do espaço de busca.
| **`ub`** | `np.array` | - | Limites superiores (*upper bounds*) do espaço de busca.
| **`swarmsize`** | `int` | 45 | Número de partículas exploradoras no enxame.
| **`omega0`** | `float` | 1.0 | Peso de inércia inicial (favorece a exploração global do mapa).
| **`omegaf`** | `float` | 0.5 | Peso de inércia final (favorece a exploração local/refinamento ao fim do loop).
| **`phip`** | `float` | 1.5 | Coeficiente de atração cognitiva (tendência da partícula voltar ao seu próprio melhor histórico).
| **`phig`** | `float` | 1.5 | Coeficiente de atração social (tendência da partícula seguir o líder do enxame).
| **`maxiter`** | `int` | 100 | Número máximo de iterações permitidas.


### Retornos (Outputs)

| Retorno | Tipo | Descrição |
| --- | --- | --- |
| **`g`** | `np.array` | O vetor contendo os melhores parâmetros globais encontrados pelo enxame (coordenadas do ótimo).
| **`fg`** | `float` | O menor valor da função objetivo associado aos parâmetros encontrados.

### Mecânica de Funcionamento

O algoritmo opera através de iterativas atualizações matemáticas do estado das partículas, divididas nas seguintes etapas principais:

**1. Pré-Alocação e Escudo Numérico:**
Para maximizar a compatibilidade com a flag `fastmath=True` do compilador, variáveis de rastreamento de erro (`fp`, `fg`, `fg_anterior`) não são iniciadas com infinitos (`np.inf`), mas sim com um teto colossal absoluto ($10^{15}$). Isso impede quebras lógicas durante as comparações matemáticas em baixo nível.

**2. Decaimento Dinâmico de Inércia:**
A inércia $\omega$ dita o momento da partícula. Ela decai linearmente a cada iteração $it$, permitindo uma transição suave da busca global (início) para a convergência local (fim):


$$\omega = \omega_0 + it \cdot \frac{\omega_f - \omega_0}{maxiter}$$

**3. Atualização Cinemática:**
A cada ciclo, a nova velocidade $v_{i,d}$ e a nova posição $x_{i,d}$ de cada dimensão $d$ da partícula $i$ são calculadas usando fatores aleatórios uniformes ($r_p, r_g \in [0, 1]$):


$$v_{i,d} = \omega \cdot v_{i,d} + \phi_p \cdot r_p \cdot (p_{i,d} - x_{i,d}) + \phi_g \cdot r_g \cdot (g_d - x_{i,d})$$

$$x_{i,d} = x_{i,d} + v_{i,d}$$


Onde $p$ é o melhor histórico pessoal da partícula e $g$ é o melhor histórico de todo o enxame.

**4. Controle de Bordas (Boundary Conditions):**
Se uma partícula ultrapassa os limites dimensionais impostos (`lb`, `ub`), sua posição é grampeada (*clipped*) no limite da borda. Crucialmente, **sua velocidade escalar é zerada ($v_{i,d} = 0.0$)**. Isso evita o "efeito mola", onde as partículas acumulariam momento falso empurrando contra a restrição paramétrica.

**5. Critério de Estagnação (Early Stopping):**
Para evitar interrupções prematuras (comuns em funções muito íngremes ou "vales estreitos"), o algoritmo só encerra o processo precocemente se a melhora na função objetivo for insignificante (menor que $10^{-4}$) por **30 iterações consecutivas**. O progresso visual do log do enxame é protegido subindo o escopo para o Python através da instrução `with objmode():`.
