"""
Visualização da ferradura de Smale para o pêndulo forçado amortecido.

O mapa da ferradura emerge do emaranhado homoclínico entre W^u e W^s da sela.
Aqui integramos condições iniciais na variedade instável e visualizamos
sua evolução sob o mapa de Poincaré estroboscópico.
"""

import math
import numpy as np
try:
    from .manifolds import saddle_orbit, iterate_points, wrap_for_plot
except ImportError:
    from manifolds import saddle_orbit, iterate_points, wrap_for_plot


def horseshoe_manifold(t0, q, f, Om, spp, nper=6, h=1e-6, N_per_ramo=400):
    """
    Computa pontos da variedade instável W^u da sela.

    Parâmetros:
    -----------
    t0, q, f, Om, spp : configuração do sistema
    nper : número de períodos para integrar para frente
    h : tamanho inicial do passo na direção do vetor tangente
    N_per_ramo : número de pontos parametrizados em cada ramo de W^u

    Retorna:
    --------
    dict com:
        - 'manifold_pts': (nper+1, N_per_ramo*2, 2) array de trajetórias
        - 'saddle': ponto da sela
        - 'mu_u', 'mu_s': multiplicadores de Floquet
        - 'nper': número de períodos
    """

    # Órbita periódica da sela
    x, mu_u, vu, mu_s, vs = saddle_orbit(t0, q, f, Om, spp)

    T = 2*math.pi/Om
    lam = abs(mu_u)  # multiplicador instável

    # Dois ramos da variedade instável (orientações opostas de vu)
    ramos = []
    for branch_sign in [+1, -1]:
        v = vu if vu[0] > 0 else -vu
        v = branch_sign * v

        # Parametrização exponencial: distância h*exp(s) onde s varia logaritmicamente
        s = h * np.exp(np.linspace(0.0, math.log(lam), N_per_ramo))
        P0 = x[None, :] + s[:, None] * v[None, :]

        # Itera para frente: sgn=+1
        iteradas = iterate_points(P0, t0, nper, spp, q, f, Om, sgn=1)
        ramos.append(iteradas)

    # Empilha os dois ramos
    manifold_pts = np.concatenate(ramos, axis=1)

    return {
        'manifold_pts': manifold_pts,
        'saddle': x,
        'mu_u': mu_u,
        'mu_s': mu_s,
        'vu': vu,
        'vs': vs,
        'nper': nper,
        't0': t0,
        'q': q,
        'f': f,
        'Om': Om,
        'spp': spp,
    }


def extract_poincare_sections(horseshoe_data, unwrap=True):
    """
    Extrai seções de Poincaré da estrutura da ferradura.

    Retorna:
    --------
    sections : list de len(nper+1) com arrays (N, 2) de pontos em cada seção
    """
    pts = horseshoe_data['manifold_pts']
    nper = horseshoe_data['nper']

    sections = []
    for k in range(nper + 1):
        if unwrap:
            th, w = wrap_for_plot(pts[k])
        else:
            th, w = pts[k, :, 0], pts[k, :, 1]
        sections.append(np.column_stack([th, w]))

    return sections


def compute_horseshoe_area(sections):
    """
    Estima a área coberta pela variedade em cada seção de Poincaré.
    Usa convex hull como proxy.
    """
    from scipy.spatial import ConvexHull

    areas = []
    for s in sections:
        try:
            hull = ConvexHull(s)
            areas.append(hull.volume)  # 2D volume = area
        except:
            areas.append(np.nan)

    return np.array(areas)


def melnikov_prediction(t0, q, f, Om, h=1e-6):
    """
    Predição analítica do tamanho do splitting de Melnikov.
    M(t0) = -8q + 2π f/cosh(πΩ/2) cos(Ωt0)

    Retorna (M_min, M_max) = (valor mínimo e máximo de |M| sobre t0 ∈ [0, 2π/Ω])
    """
    import scipy.optimize as opt

    def M_func(t0_val):
        return abs(-8*q + 2*math.pi*f/math.cosh(math.pi*Om/2)*math.cos(Om*t0_val))

    res_min = opt.minimize_scalar(M_func, bounds=(0, 2*math.pi/Om), method='bounded')
    res_max = opt.minimize_scalar(lambda t: -M_func(t), bounds=(0, 2*math.pi/Om), method='bounded')

    return res_min.fun, -res_max.fun


if __name__ == '__main__':
    # Teste rápido
    import matplotlib.pyplot as plt

    q, f, Om, spp = 0.5, 1.2, 2/3, 100
    t0 = 0.0
    nper = 4

    print("Computando ferradura de Smale...")
    hs = horseshoe_manifold(t0, q, f, Om, spp, nper=nper, N_per_ramo=300)

    print(f"  Sela em θ={hs['saddle'][0]:.4f}, ω={hs['saddle'][1]:.4f}")
    print(f"  μ_u={hs['mu_u']:.4f}, μ_s={hs['mu_s']:.4f}")

    sections = extract_poincare_sections(hs)
    areas = compute_horseshoe_area(sections)

    print(f"  Áreas das seções: {areas}")

    # Plotagem
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for k in range(min(6, len(sections))):
        ax = axes.flat[k]
        s = sections[k]
        # Remove NaNs para plotagem
        valid = ~(np.isnan(s[:, 0]) | np.isnan(s[:, 1]))
        ax.plot(s[valid, 0], s[valid, 1], 'b-', alpha=0.7, linewidth=0.5)
        ax.plot(hs['saddle'][0], hs['saddle'][1], 'r*', markersize=12)
        ax.set_xlabel('θ')
        ax.set_ylabel('ω')
        ax.set_title(f'Poincaré section k={k}')
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('/home/claude/melnikov/horseshoe_test.png', dpi=150)
    print("  Salvo em horseshoe_test.png")
