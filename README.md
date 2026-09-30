# Caos no pêndulo forçado e amortecido: notebooks do artigo

Códigos que geram todas as figuras do artigo:

> **Caracterização do caos no pêndulo forçado e amortecido: instabilidade, informação e geometria do espaço de fase**

Cada figura tem um notebook próprio, com o código comentado, a explicação das equações e dos métodos numéricos e as saídas já calculadas. O leitor pode acompanhar o artigo e abrir o notebook da figura correspondente sem instalar nada.

## Como usar

**Só ler.** Clique em um notebook da tabela abaixo. O GitHub mostra o texto, o código, os números impressos e as figuras exatamente como saíram da execução.

**Rodar no navegador.** Clique no botão *Open in Colab* do notebook. O Google Colab abre uma cópia que pode ser executada e modificada. Use *Ambiente de execução > Executar tudo*; nada é instalado no seu computador e o original não é alterado.

**Rodar no computador.**

```bash
git clone https://github.com/SEU-USUARIO/pendulo-caotico.git
cd pendulo-caotico
pip install -r requirements.txt
jupyter lab
```

## Notebooks

| Figura | Seção do artigo | Notebook | O que calcula | Tempo* |
|:---:|---|---|---|:---:|
| 1 | III | [01_retrato_de_fase](01_retrato_de_fase.ipynb) <br>[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SEU-USUARIO/pendulo-caotico/blob/main/01_retrato_de_fase.ipynb) | Retrato de fase sem forçamento; equilíbrios e sua estabilidade | ~5 s |
| 2 | III | [02_secoes_de_poincare_e_lyapunov](02_secoes_de_poincare_e_lyapunov.ipynb) <br>[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SEU-USUARIO/pendulo-caotico/blob/main/02_secoes_de_poincare_e_lyapunov.ipynb) | Seções de Poincaré e convergência de $\lambda_{\max}$ em $f = 0{,}9$ e $1{,}2$ | ~10 s |
| 3 | III | [03_diagrama_de_bifurcacao](03_diagrama_de_bifurcacao.ipynb) <br>[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SEU-USUARIO/pendulo-caotico/blob/main/03_diagrama_de_bifurcacao.ipynb) | Diagrama de bifurcação e $\lambda_{\max}(f)$ em $f \in [0{,}9;\ 1{,}5]$ | ~10 s |
| 4 | IV | [04_metodo_de_melnikov](04_metodo_de_melnikov.ipynb) <br>[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SEU-USUARIO/pendulo-caotico/blob/main/04_metodo_de_melnikov.ipynb) | Função de Melnikov e limiar analítico do caos $f_c$; variedades invariantes da sela, tangência exata, fronteiras de bacia, e **ferradura de Smale** na seção de Poincaré | ~17 min |
| 5 | V | [05_dimensao_fractal](05_dimensao_fractal.ipynb) <br>[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SEU-USUARIO/pendulo-caotico/blob/main/05_dimensao_fractal.ipynb) | Box-counting $D_0$ e dimensão de Kaplan-Yorke | ~15 s |
| 6 | VI | [06_descritores_lagrangianos](06_descritores_lagrangianos.ipynb) <br>[![Abrir no Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/SEU-USUARIO/pendulo-caotico/blob/main/06_descritores_lagrangianos.ipynb) | Descritores lagrangianos $L^+$ e $\lVert\nabla L^+\rVert$ | ~20 s |

\*Tempo de execução medido em uma máquina com 2 núcleos, semelhante à do Colab gratuito. Inclui a compilação do Numba na primeira execução.

Os notebooks são independentes: cada um define o seu próprio integrador e pode ser aberto e executado sozinho, em qualquer ordem.

## Modelo e parâmetros comuns

Todos os notebooks integram a Eq. (3) do artigo,

$$\dot\theta = \omega, \qquad \dot\omega = -q\,\omega - \sin\theta + f\cos(\Omega t),$$

com Runge-Kutta de 4ª ordem de passo fixo.

| Símbolo no artigo | Nome no código | Valor |
|---|---|---|
| $q$ (amortecimento) | `beta` | 0,5 (0,25 na Figura 1) |
| $\Omega$ (frequência do forçamento) | `Omega` | 2/3 |
| $T_d = 2\pi/\Omega$ (período do forçamento) | `T_D` | $3\pi \approx 9{,}42$ |
| $f$ (amplitude do forçamento) | `f`, `F_PER`, `F_CAO` | 0,9 (periódico) e 1,2 (caótico); domínio $[0{,}9;\ 1{,}5]$ |
| passos por período | `spp`, `SPP` | 80 a 200, conforme o notebook |

## Organização de cada notebook

1. **Cabeçalho:** figura e seção do artigo, o que o notebook calcula e o tempo de execução.
2. **Equações:** as equações do artigo usadas no cálculo, com a numeração do texto.
3. **Código em blocos:** cada bloco é precedido de uma explicação do que faz e por quê.
4. **Verificações:** testes internos, como autovalores dos equilíbrios, multiplicadores de Floquet e a regra da soma dos expoentes.
5. **Leitura dos resultados:** os números obtidos comparados com os valores citados no artigo.
6. **Para explorar:** sugestões de modificações, na linha do projeto proposto no Apêndice A do artigo.

Os notebooks seguem a organização do artigo: **instabilidade** (02 e 03), **mecanismo e limiar do caos** (04), **geometria** (05) e **estrutura do espaço de fases** (06).

## Figuras

As figuras geradas ficam na pasta [`figuras/`](figuras).

## Estrutura

```
pendulo-caotico/
├── README.md
├── requirements.txt
├── manifolds.py           # Biblioteca de cálculo das variedades invariantes (importado por 04)
├── horseshoe.py           # Visualização da ferradura de Smale (importado por 04)
├── 01_retrato_de_fase.ipynb
├── 02_secoes_de_poincare_e_lyapunov.ipynb
├── 03_diagrama_de_bifurcacao.ipynb
├── 04_metodo_de_melnikov.ipynb
├── 05_dimensao_fractal.ipynb
├── 06_descritores_lagrangianos.ipynb
└── figuras/
```

## Dependências

Python 3.9 ou mais recente, com `numpy`, `matplotlib` e `numba`; o Notebook 04 usa também `sympy` e `mpmath` (ver `requirements.txt`). O Numba compila os laços de integração e é o que torna viáveis os milhões de passos de RK4 de cada figura. Na primeira execução de cada notebook há alguns segundos de compilação.
