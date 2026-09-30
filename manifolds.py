import math
import numpy as np
from numba import njit, prange


@njit(cache=True)
def rhs(t, th, w, q, f, Om):
    return w, -q*w - math.sin(th) + f*math.cos(Om*t)


@njit(cache=True)
def rk4(t, th, w, dt, q, f, Om):
    k1x, k1y = rhs(t, th, w, q, f, Om)
    k2x, k2y = rhs(t + dt/2, th + dt*k1x/2, w + dt*k1y/2, q, f, Om)
    k3x, k3y = rhs(t + dt/2, th + dt*k2x/2, w + dt*k2y/2, q, f, Om)
    k4x, k4y = rhs(t + dt, th + dt*k3x, w + dt*k3y, q, f, Om)
    return (th + dt/6*(k1x + 2*k2x + 2*k3x + k4x),
            w + dt/6*(k1y + 2*k2y + 2*k3y + k4y))


@njit(cache=True)
def flow(th, w, t0, nper, spp, q, f, Om, sgn):
    """Integra nper períodos a partir do instante t0; sgn=+1 para frente, -1 para trás."""
    T = 2*math.pi/Om
    dt = sgn*T/spp
    t = t0
    for _ in range(nper*spp):
        th, w = rk4(t, th, w, dt, q, f, Om)
        t += dt
    return th, w


@njit(cache=True)
def rhs_var(t, th, w, a, b, c, d, q, f, Om):
    # estado + matriz tangente [[a, c], [b, d]] (colunas = vetores tangentes)
    ct = math.cos(th)
    return (w, -q*w - math.sin(th) + f*math.cos(Om*t),
            b, -ct*a - q*b,
            d, -ct*c - q*d)


@njit(cache=True)
def map_jac(th, w, t0, spp, q, f, Om):
    """Mapa estroboscópico P_{t0} e sua jacobiana (um período, para frente)."""
    T = 2*math.pi/Om
    dt = T/spp
    t = t0
    a, b, c, d = 1.0, 0.0, 0.0, 1.0
    for _ in range(spp):
        k1 = rhs_var(t, th, w, a, b, c, d, q, f, Om)
        k2 = rhs_var(t + dt/2, th + dt*k1[0]/2, w + dt*k1[1]/2, a + dt*k1[2]/2, b + dt*k1[3]/2,
                     c + dt*k1[4]/2, d + dt*k1[5]/2, q, f, Om)
        k3 = rhs_var(t + dt/2, th + dt*k2[0]/2, w + dt*k2[1]/2, a + dt*k2[2]/2, b + dt*k2[3]/2,
                     c + dt*k2[4]/2, d + dt*k2[5]/2, q, f, Om)
        k4 = rhs_var(t + dt, th + dt*k3[0], w + dt*k3[1], a + dt*k3[2], b + dt*k3[3],
                     c + dt*k3[4], d + dt*k3[5], q, f, Om)
        th += dt/6*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        w += dt/6*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        a += dt/6*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
        b += dt/6*(k1[3] + 2*k2[3] + 2*k3[3] + k4[3])
        c += dt/6*(k1[4] + 2*k2[4] + 2*k3[4] + k4[4])
        d += dt/6*(k1[5] + 2*k2[5] + 2*k3[5] + k4[5])
        t += dt
    return th, w, np.array([[a, c], [b, d]])


@njit(cache=True)
def seg_jac(th, w, t, nsteps, dt, q, f, Om):
    """Fluxo por nsteps passos a partir do instante t, com a jacobiana do trecho."""
    a, b, c, d = 1.0, 0.0, 0.0, 1.0
    for _ in range(nsteps):
        k1 = rhs_var(t, th, w, a, b, c, d, q, f, Om)
        k2 = rhs_var(t + dt/2, th + dt*k1[0]/2, w + dt*k1[1]/2, a + dt*k1[2]/2, b + dt*k1[3]/2,
                     c + dt*k1[4]/2, d + dt*k1[5]/2, q, f, Om)
        k3 = rhs_var(t + dt/2, th + dt*k2[0]/2, w + dt*k2[1]/2, a + dt*k2[2]/2, b + dt*k2[3]/2,
                     c + dt*k2[4]/2, d + dt*k2[5]/2, q, f, Om)
        k4 = rhs_var(t + dt, th + dt*k3[0], w + dt*k3[1], a + dt*k3[2], b + dt*k3[3],
                     c + dt*k3[4], d + dt*k3[5], q, f, Om)
        th += dt/6*(k1[0] + 2*k2[0] + 2*k3[0] + k4[0])
        w += dt/6*(k1[1] + 2*k2[1] + 2*k3[1] + k4[1])
        a += dt/6*(k1[2] + 2*k2[2] + 2*k3[2] + k4[2])
        b += dt/6*(k1[3] + 2*k2[3] + 2*k3[3] + k4[3])
        c += dt/6*(k1[4] + 2*k2[4] + 2*k3[4] + k4[4])
        d += dt/6*(k1[5] + 2*k2[5] + 2*k3[5] + k4[5])
        t += dt
    return th, w, np.array([[a, c], [b, d]])


def saddle_multishoot(t0, q, f, Om, spp, m=10, X=None, maxit=50, tol=1e-12):
    """Órbita de sela por tiro múltiplo: m nós igualmente espaçados em um período.
    Devolve os nós (m x 2); o nó 0 está no instante t0."""
    T = 2*math.pi/Om
    assert spp % m == 0
    ns = spp//m
    dt = T/spp
    tj = t0 + np.arange(m)*T/m
    if X is None:  # chute: resposta linear em torno de theta = pi
        z = f/complex(-Om**2 - 1.0, q*Om)
        X = np.array([[math.pi + (z*np.exp(1j*Om*t)).real, (1j*Om*z*np.exp(1j*Om*t)).real] for t in tj])
    for _ in range(maxit):
        A = np.zeros((2*m, 2*m))
        G = np.zeros(2*m)
        for j in range(m):
            th1, w1, J = seg_jac(X[j, 0], X[j, 1], tj[j], ns, dt, q, f, Om)
            jn = (j + 1) % m
            G[2*j:2*j + 2] = [th1 - X[jn, 0], w1 - X[jn, 1]]
            A[2*j:2*j + 2, 2*j:2*j + 2] = J
            A[2*j:2*j + 2, 2*jn:2*jn + 2] -= np.eye(2)
        dX = np.linalg.solve(A, -G).reshape(m, 2)
        X = X + dX
        if np.max(np.abs(dX)) < tol:
            return X, True
    return X, np.max(np.abs(G)) < 1e-9


def newton_fixed(x, t0, q, f, Om, spp, maxit=60, tol=1e-13):
    """Newton para P_{t0}(x) = x, com passo limitado."""
    for _ in range(maxit):
        th1, w1, J = map_jac(x[0], x[1], t0, spp, q, f, Om)
        G = np.array([th1 - x[0], w1 - x[1]])
        dx = np.linalg.solve(J - np.eye(2), -G)
        nrm = np.max(np.abs(dx))
        if nrm > 0.2:
            dx *= 0.2/nrm
        x = x + dx
        if nrm < tol:
            return x, True
    return x, False


def saddle_orbit(t0, q, f, Om, spp, nsteps=None):
    """Ponto da órbita periódica de sela (pêndulo invertido) na seção de fase t0.
    Continuação em f a partir de f = 0, onde a sela é exatamente (pi, 0)."""
    if nsteps is None:
        nsteps = max(1, int(math.ceil(abs(f)/0.1)))
    X = None
    for fk in np.linspace(0.0, f, nsteps + 1)[1:]:
        X, ok = saddle_multishoot(t0, q, fk, Om, spp, X=X)
        if not ok:
            raise RuntimeError(f"tiro múltiplo não convergiu em f={fk}")
    x = X[0].copy()
    _, _, J = map_jac(x[0], x[1], t0, spp, q, f, Om)
    mu, V = np.linalg.eig(J)
    iu, is_ = (0, 1) if abs(mu[0]) > abs(mu[1]) else (1, 0)
    return x, mu[iu].real, V[:, iu].real, mu[is_].real, V[:, is_].real


@njit(parallel=True, cache=True)
def iterate_points(P0, t0, nper, spp, q, f, Om, sgn):
    """Aplica o mapa (sgn=+1) ou o inverso (sgn=-1) nper vezes a cada ponto; devolve todas as iteradas."""
    N = P0.shape[0]
    out = np.empty((nper + 1, N, 2))
    T = 2*math.pi/Om
    for i in prange(N):
        th, w = P0[i, 0], P0[i, 1]
        out[0, i, 0] = th
        out[0, i, 1] = w
        t = t0
        for k in range(nper):
            th, w = flow(th, w, t, 1, spp, q, f, Om, sgn)
            t += sgn*T
            out[k + 1, i, 0] = th
            out[k + 1, i, 1] = w
    return out


def branch(t0, q, f, Om, spp, kind, h=1e-7, N=6000, nper=4):
    """Ramo superior de W^u (sela em -pi, saindo para a direita) ou de W^s (sela em +pi, chegando pela esquerda).
    Devolve s (parâmetro no domínio fundamental), as iteradas e dados da sela."""
    x, mu_u, vu, mu_s, vs = saddle_orbit(t0, q, f, Om, spp)
    if kind == "u":
        p = x - np.array([2*math.pi, 0.0])        # sela em -pi
        v = vu if vu[0] > 0 else -vu             # sai para theta crescente
        lam, sgn = mu_u, 1
    else:
        p = x.copy()                              # sela em +pi
        v = vs if vs[0] < 0 else -vs             # chega por theta menor
        lam, sgn = 1.0/mu_s, -1
    s = h*np.exp(np.linspace(0.0, math.log(abs(lam)), N))
    P0 = p[None, :] + s[:, None]*v[None, :]
    it = iterate_points(P0, t0, nper, spp, q, f, Om, sgn)
    return s, it, p, v, sgn, (mu_u, mu_s)


def first_crossing(t0, q, f, Om, spp, kind, h=1e-7, N=6000, nper=4):
    """omega no primeiro cruzamento do ramo com theta = 0 (refinado por bissecção)."""
    s, it, p, v, sgn, mus = branch(t0, q, f, Om, spp, kind, h, N, nper)
    T = 2*math.pi/Om
    for k in range(nper + 1):
        th = it[k, :, 0]
        if kind == "u":
            idx = np.where((th[:-1] < 0) & (th[1:] >= 0))[0]
        else:
            idx = np.where((th[:-1] > 0) & (th[1:] <= 0))[0]
        if len(idx):
            i = idx[0]
            a, b = s[i], s[i + 1]
            fa = th[i]
            for _ in range(60):
                m = 0.5*(a + b)
                xm = p + m*v
                thm, wm = flow(xm[0], xm[1], t0, k, spp, q, f, Om, sgn) if k > 0 else (xm[0], xm[1])
                if (thm < 0) == (fa < 0):
                    a, fa = m, thm
                else:
                    b = m
            xm = p + 0.5*(a + b)*v
            thm, wm = flow(xm[0], xm[1], t0, k, spp, q, f, Om, sgn) if k > 0 else (xm[0], xm[1])
            return wm, k
    return np.nan, -1


def splitting(t0, q, f, Om, spp=300, **kw):
    wu, ku = first_crossing(t0, q, f, Om, spp, "u", **kw)
    ws, ks = first_crossing(t0, q, f, Om, spp, "s", **kw)
    return wu - ws, wu, ws, ku, ks


def melnikov(t0, q, f, Om):
    return -8*q + 2*math.pi*f/math.cosh(math.pi*Om/2)*np.cos(Om*t0)


def manifold_curve(t0, q, f, Om, spp, kind, branch_sign=+1, h=1e-7, N=4000, nper=4,
                   gap=0.02, max_pts=300000, saddle=None):
    """Curva ordenada de um ramo de W^u (kind='u') ou W^s (kind='s') da sela em theta~pi.
    branch_sign escolhe o ramo (+1: direção +v com v orientado para theta crescente)."""
    if saddle is None:
        saddle = saddle_orbit(t0, q, f, Om, spp)
    x, mu_u, vu, mu_s, vs = saddle
    if kind == "u":
        v = vu if vu[0] > 0 else -vu
        lam, sgn = abs(mu_u), 1
    else:
        v = vs if vs[0] > 0 else -vs
        lam, sgn = 1.0/abs(mu_s), -1
    v = branch_sign*v
    pieces = []
    logs = np.linspace(0.0, math.log(lam), N)
    for k in range(1, nper + 1):
        L = logs.copy()
        for _ in range(12):
            P0 = x[None, :] + (h*np.exp(L))[:, None]*v[None, :]
            it = iterate_points(P0, t0, k, spp, q, f, Om, sgn)[-1]
            dist = np.hypot(np.diff(it[:, 0]), np.diff(it[:, 1]))
            bad = np.where(dist > gap)[0]
            if len(bad) == 0 or len(L) + len(bad) > max_pts:
                break
            L = np.sort(np.concatenate([L, 0.5*(L[bad] + L[bad + 1])]))
        pieces.append(it)
    return np.vstack(pieces), x


def wrap_for_plot(P, maxstep=0.25):
    """Reduz theta a (-pi, pi] e insere NaN nos saltos (troca de cópia ou trecho não resolvido)."""
    th = np.arctan2(np.sin(P[:, 0]), np.cos(P[:, 0]))
    w = P[:, 1].copy()
    step = np.hypot(np.diff(th), np.diff(w))
    jump = np.where((np.abs(np.diff(th)) > math.pi) | (step > maxstep))[0]
    th = np.insert(th, jump + 1, np.nan)
    w = np.insert(w, jump + 1, np.nan)
    return th, w
