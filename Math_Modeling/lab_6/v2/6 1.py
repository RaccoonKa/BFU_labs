import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider
from scipy.integrate import solve_ivp
from scipy.linalg import solve_banded

def get_D(x):
    return (x + 1.0) * (11.0 - x)

def get_f(x):
    return x

def get_initial_condition(x):
    return x * (10.0 - x)**2

def get_t_max():
    d_0 = get_D(0.0)
    d_10 = get_D(10.0)
    return 100.0 / np.sqrt(d_0 * d_10)

def build_reference_solution(cfg, t_max):
    x_dense = np.linspace(cfg["x_min"], cfg["x_max"], cfg["dense_points"])
    h = float(x_dense[1] - x_dense[0])
    D = get_D(x_dense)
    f = get_f(x_dense)

    def odefun(t, u):
        du = np.zeros_like(u)
        du[1:-1] = D[1:-1] * (u[2:] - 2.0 * u[1:-1] + u[:-2]) / (h**2) - f[1:-1] * u[1:-1] + 5.0
        return du

    u0 = get_initial_condition(x_dense)
    sol = solve_ivp(odefun, [0, t_max], u0, method='BDF', dense_output=True, rtol=1e-8, atol=1e-10)

    def ref_eval(t, x_target):
        u_dense = sol.sol(t)
        return np.interp(x_target, x_dense, u_dense)

    return ref_eval

def build_grid(nx, nt, x_min, x_max, t_min, t_max):
    x = np.linspace(x_min, x_max, nx + 1)
    t = np.linspace(t_min, t_max, nt + 1)
    h = float(x[1] - x[0])
    tau = float(t[1] - t[0])
    return x, t, h, tau

def solve_explicit_upwind(x, t, h, tau, cfg):
    nx = len(x) - 1
    nt = len(t) - 1
    u = np.zeros((nt + 1, nx + 1))
    u[0, :] = get_initial_condition(x)
    
    D = get_D(x)
    f = get_f(x)
    gamma = tau * (2.0 * D / (h**2) + f)
    c_max = float(np.max(gamma))
    
    for n in range(nt):
        u[n, 0] = 0.0
        u[n, -1] = 0.0
        u_next = u[n, 1:-1] + tau * (
            D[1:-1] * (u[n, 2:] - 2.0 * u[n, 1:-1] + u[n, :-2]) / (h**2)
            - f[1:-1] * u[n, 1:-1] + 5.0
        )
        u[n + 1, 1:-1] = np.clip(u_next, -cfg["max_abs_clip"], cfg["max_abs_clip"])
        
    u[:, 0] = 0.0
    u[:, -1] = 0.0
    return u, c_max

def solve_implicit_upwind(x, t, h, tau, cfg):
    nx = len(x) - 1
    nt = len(t) - 1
    u = np.zeros((nt + 1, nx + 1))
    u[0, :] = get_initial_condition(x)
    
    D = get_D(x)
    f = get_f(x)
    gamma = tau * (2.0 * D / (h**2) + f)
    c_max = float(np.max(gamma))
    
    A = -tau * D[1:-1] / (h**2)
    C = -tau * D[1:-1] / (h**2)
    B = 1.0 + 2.0 * tau * D[1:-1] / (h**2) + tau * f[1:-1]
    
    ab = np.zeros((3, nx - 1))
    ab[0, 1:] = C[:-1]
    ab[1, :] = B
    ab[2, :-1] = A[1:]
    
    for n in range(nt):
        u[n+1, 0] = 0.0
        u[n+1, -1] = 0.0
        F = u[n, 1:-1] + 5.0 * tau
        u[n+1, 1:-1] = solve_banded((1, 1), ab, F)
        
    return u, c_max

def calculate_rmse(u_num, u_exact, axis=None):
    if u_exact is None:
        return np.zeros(u_num.shape[0]) if axis == 1 else 0.0
    return np.sqrt(np.mean((u_num - u_exact) ** 2, axis=axis))

def run_all_variants(grid_variants, t_max, cfg):
    results = []
    
    if cfg["need_analytical"]:
        ref_func = build_reference_solution(cfg, t_max)
    else:
        ref_func = None

    for var in grid_variants:
        nx, nt = var["nx"], var["nt"]
        x, t, h, tau = build_grid(nx, nt, cfg["x_min"], cfg["x_max"], cfg["t_min"], t_max)

        u_exp, c_max = solve_explicit_upwind(x, t, h, tau, cfg)
        u_imp, _ = solve_implicit_upwind(x, t, h, tau, cfg)

        if ref_func is not None:
            u_exact = np.zeros_like(u_exp)
            for n_idx, t_val in enumerate(t):
                u_exact[n_idx, :] = ref_func(t_val, x)
            rmse_exp_by_t = calculate_rmse(u_exp, u_exact, axis=1)
            rmse_imp_by_t = calculate_rmse(u_imp, u_exact, axis=1)
            rmse_exp_2d = float(calculate_rmse(u_exp, u_exact))
            rmse_imp_2d = float(calculate_rmse(u_imp, u_exact))
        else:
            u_exact = np.zeros_like(u_exp)
            rmse_exp_by_t = np.zeros(len(t))
            rmse_imp_by_t = np.zeros(len(t))
            rmse_exp_2d = 0.0
            rmse_imp_2d = 0.0

        is_stable = c_max <= cfg["courant_limit"]

        results.append({
            "name": var["name"],
            "nx": nx,
            "nt": nt,
            "x": x,
            "t": t,
            "h": h,
            "tau": tau,
            "c_max": c_max,
            "is_stable": is_stable,
            "u_exp": u_exp,
            "u_imp": u_imp,
            "u_exact": u_exact,
            "rmse_exp_by_t": rmse_exp_by_t,
            "rmse_imp_by_t": rmse_imp_by_t,
            "rmse_exp_2d": rmse_exp_2d,
            "rmse_imp_2d": rmse_imp_2d,
        })
        
    return results, ref_func

def get_full_zone_ylim(results, y_padding):
    u_min = min(float(np.min(r["u_exp"])) for r in results if r["is_stable"])
    u_max = max(float(np.max(r["u_exp"])) for r in results if r["is_stable"])
    return (u_min - y_padding, u_max + y_padding)

def print_summary_table(results, t_max, cfg):
    t_init = cfg["t_initial_slider"]
    
    print(f"Время полного прохождения области: t_max = {t_max:.6f} с")
    print(f"Условие устойчивости явной схемы: C_max <= {cfg['courant_limit']}")
    
    if cfg["need_analytical"]:
        h_dense = (cfg["x_max"] - cfg["x_min"]) / (cfg["dense_points"] - 1)
        ref_error = h_dense ** 2
        print(f"Оценочная погрешность эталонного решения (Method of Lines): ~{ref_error:.2e}\n")
    else:
        print("Аналитическое/эталонное решение ОТКЛЮЧЕНО. Оценка порядка методом Рунге:")
        for i in range(len(results) - 2):
            r1, r2, r3 = results[i], results[i+1], results[i+2]
            if r2["nx"] == 2 * r1["nx"] and r3["nx"] == 2 * r2["nx"] and r1["is_stable"] and r2["is_stable"] and r3["is_stable"]:
                idx1 = int(np.argmin(np.abs(r1["t"] - t_init)))
                idx2 = int(np.argmin(np.abs(r2["t"] - t_init)))
                idx3 = int(np.argmin(np.abs(r3["t"] - t_init)))
                
                u1_e, u2_e, u3_e = r1["u_exp"][idx1, :], r2["u_exp"][idx2, ::2], r3["u_exp"][idx3, ::4]
                d1_e = np.sqrt(np.mean((u1_e - u2_e)**2))
                d2_e = np.sqrt(np.mean((u2_e - u3_e)**2))
                p_e = np.log2(d1_e / d2_e) if d2_e > 1e-16 else 0.0
                
                u1_i, u2_i, u3_i = r1["u_imp"][idx1, :], r2["u_imp"][idx2, ::2], r3["u_imp"][idx3, ::4]
                d1_i = np.sqrt(np.mean((u1_i - u2_i)**2))
                d2_i = np.sqrt(np.mean((u2_i - u3_i)**2))
                p_i = np.log2(d1_i / d2_i) if d2_i > 1e-16 else 0.0
                
                print(f"  [{r1['name']} -> {r2['name']} -> {r3['name']}] (t={t_init}):")
                print(f"    - Явная схема: p = {p_e:.3f}")
                print(f"    - Неявная схема: p = {p_i:.3f}\n")

    header = (
        f"{'Вариант':<16} | {'h':<6} | {'tau':<8} | {'C_max':<7} | {'Устойчивость':<14} | "
        f"{f'СКО Явн (t={t_init})':<17} | {f'СКО Неявн (t={t_init})':<19} | "
        f"{'СКО Явная (2D)':<14} | {'СКО Неявная (2D)':<16}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        idx = int(np.argmin(np.abs(r["t"] - t_init)))
        status = "Устойчива" if r["is_stable"] else "НЕУСТОЙЧИВА"
        print(
            f"{r['name']:<16} | {r['h']:<6.3f} | {r['tau']:<8.5f} | {r['c_max']:<7.3f} | "
            f"{status:<14} | {r['rmse_exp_by_t'][idx]:<17.5e} | {r['rmse_imp_by_t'][idx]:<19.5e} | "
            f"{r['rmse_exp_2d']:<14.5e} | {r['rmse_imp_2d']:<16.5e}"
        )

def plot_variants_grid_interactive(results, t_max, full_ylim, cfg, ref_func):
    plt.rcParams.update({"font.size": 9.5})
    fig, axes = plt.subplots(2, 3, figsize=cfg["fig1_size"])
    plt.subplots_adjust(left=0.06, right=0.98, top=0.89, bottom=0.18, hspace=0.45, wspace=0.24)

    t_init = cfg["t_initial_slider"]
    fig.suptitle("Сравнение 5 вариантов между схемами", fontsize=13, fontweight="bold")

    x_dense = np.linspace(cfg["x_min"], cfg["x_max"], cfg["dense_points"])
    
    if cfg["need_analytical"]:
        u_exact_dense = ref_func(t_init, x_dense)
        u_err_up = u_exact_dense + cfg["corridor_width"]
        u_err_dn = u_exact_dense - cfg["corridor_width"]
    else:
        u_exact_dense = np.zeros_like(x_dense)
        u_err_up = np.zeros_like(x_dense)
        u_err_dn = np.zeros_like(x_dense)

    flat_axes = axes.flatten()

    exact_lines = []
    err_up_lines = []
    err_dn_lines = []
    exp_lines = []
    imp_lines = []
    legends = []

    for i, r in enumerate(results):
        ax = flat_axes[i]
        idx = int(np.argmin(np.abs(r["t"] - t_init)))

        l_ex, = ax.plot(x_dense, u_exact_dense, "k-", lw=2.0, label="Эталонное решение")
        if cfg["need_analytical"]:
            l_e_u, = ax.plot(x_dense, u_err_up, color="gray", ls=":", alpha=0.6)
            l_e_d, = ax.plot(x_dense, u_err_dn, color="gray", ls=":", alpha=0.6)
        else:
            l_e_u, = ax.plot([], [])
            l_e_d, = ax.plot([], [])
            
        l_exp, = ax.plot(
            r["x"], r["u_exp"][idx, :], "r--", lw=1.5,
            label=f"Явный (СКО={r['rmse_exp_by_t'][idx]:.3f})"
        )
        l_imp, = ax.plot(
            r["x"], r["u_imp"][idx, :], "b-.", lw=1.6,
            label=f"Неявный (СКО={r['rmse_imp_by_t'][idx]:.3f})"
        )

        status_tag = "Устойч." if r["is_stable"] else "НЕУСТОЙЧ."
        ax.set_title(
            f"{r['name']}\nh={r['h']:.3f}, τ={r['tau']:.4e} | C_max={r['c_max']:.2f} ({status_tag})",
            fontsize=9.5,
        )
        ax.set_xlabel("Координата x")
        ax.set_ylabel("U(t, x)")
        ax.set_xlim(cfg["x_min"], cfg["x_max"])
        ax.set_ylim(full_ylim)
        ax.grid(True, alpha=0.4)
        leg = ax.legend(loc="upper right", fontsize=8)

        exact_lines.append(l_ex)
        err_up_lines.append(l_e_u)
        err_dn_lines.append(l_e_d)
        exp_lines.append(l_exp)
        imp_lines.append(l_imp)
        legends.append(leg)

    ax_err = flat_axes[5]
    indices = np.arange(len(results))
    bar_w = cfg["bar_width"]
    
    labels = [f"{r['name']}\nh={r['h']:.2g}\nτ={r['tau']:.1e}" for r in results]

    init_rmse_exp = [
        max(r["rmse_exp_by_t"][int(np.argmin(np.abs(r["t"] - t_init)))], cfg["rmse_floor"])
        for r in results
    ]
    init_rmse_imp = [
        max(r["rmse_imp_by_t"][int(np.argmin(np.abs(r["t"] - t_init)))], cfg["rmse_floor"])
        for r in results
    ]

    bars_exp = ax_err.bar(indices - bar_w / 2, init_rmse_exp, width=bar_w, color="#d62728", alpha=0.85, label="СКО Явная")
    bars_imp = ax_err.bar(indices + bar_w / 2, init_rmse_imp, width=bar_w, color="#1f77b4", alpha=0.85, label="СКО Неявная")

    texts_exp = []
    texts_imp = []
    for i, (b_e, b_i, err_e, err_i) in enumerate(zip(bars_exp, bars_imp, init_rmse_exp, init_rmse_imp)):
        te = ax_err.text(b_e.get_x() + b_e.get_width() / 2, err_e * 1.25,
                         f"{err_e:.1e}", ha='center', va='bottom', fontsize=7.5, rotation=90, color="#d62728", fontweight="bold")
        ti = ax_err.text(b_i.get_x() + b_i.get_width() / 2, err_i * 1.25,
                         f"{err_i:.1e}", ha='center', va='bottom', fontsize=7.5, rotation=90, color="#1f77b4", fontweight="bold")
        texts_exp.append(te)
        texts_imp.append(ti)

    ax_err.set_yscale("log")
    ax_err.set_ylim(cfg["rmse_ylim"])
    ax_err.set_xticks(indices)
    ax_err.set_xticklabels(labels, fontsize=8)
    ax_err.set_title("6. Сравнение СКО (RMSE)", fontsize=10)
    ax_err.set_ylabel("СКО (RMSE, лог. шкала)")
    ax_err.grid(True, which="both", axis="y", alpha=0.4)
    ax_err.legend(loc="upper left", fontsize=8.5)

    ax_slider = fig.add_axes(cfg["slider_rect"])
    time_slider = Slider(
        ax=ax_slider,
        label="Время t (с): ",
        valmin=cfg["t_min"],
        valmax=t_max,
        valinit=t_init,
        valstep=cfg["slider_step"],
        color="#2ca02c",
    )

    def update(t_val):
        if cfg["need_analytical"]:
            u_ex_now = ref_func(t_val, x_dense)
            u_eu_now = u_ex_now + cfg["corridor_width"]
            u_ed_now = u_ex_now - cfg["corridor_width"]
        else:
            u_ex_now = np.zeros_like(x_dense)
            u_eu_now = np.zeros_like(x_dense)
            u_ed_now = np.zeros_like(x_dense)

        for i, r in enumerate(results):
            idx = int(np.argmin(np.abs(r["t"] - t_val)))
            exact_lines[i].set_ydata(u_ex_now)
            
            if cfg["need_analytical"]:
                err_up_lines[i].set_ydata(u_eu_now)
                err_dn_lines[i].set_ydata(u_ed_now)
                
            exp_lines[i].set_ydata(r["u_exp"][idx, :])
            imp_lines[i].set_ydata(r["u_imp"][idx, :])

            err_e = r["rmse_exp_by_t"][idx]
            err_i = r["rmse_imp_by_t"][idx]

            texts = legends[i].get_texts()
            texts[1].set_text(f"Явный (СКО={err_e:.3f})")
            texts[2].set_text(f"Неявный (СКО={err_i:.3f})")

            h_e = max(err_e, cfg["rmse_floor"])
            h_i = max(err_i, cfg["rmse_floor"])
            
            bars_exp[i].set_height(h_e)
            bars_imp[i].set_height(h_i)

            texts_exp[i].set_position((bars_exp[i].get_x() + bars_exp[i].get_width() / 2, h_e * 1.25))
            texts_exp[i].set_text(f"{err_e:.1e}")
            
            texts_imp[i].set_position((bars_imp[i].get_x() + bars_imp[i].get_width() / 2, h_i * 1.25))
            texts_imp[i].set_text(f"{err_i:.1e}")

        fig.canvas.draw_idle()

    time_slider.on_changed(update)
    return fig, time_slider

def plot_schemes_overlay_interactive(results, t_max, full_ylim, cfg, ref_func):
    fig, (ax_exp, ax_imp) = plt.subplots(1, 2, figsize=cfg["fig2_size"])
    plt.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.18, wspace=0.20)

    t_init = cfg["t_initial_slider"]
    fig.suptitle("Сравнение 5 вариантов сеток внутри схем", fontsize=13, fontweight="bold")

    x_dense = np.linspace(cfg["x_min"], cfg["x_max"], cfg["dense_points"])
    
    if cfg["need_analytical"]:
        u_exact_dense = ref_func(t_init, x_dense)
    else:
        u_exact_dense = np.zeros_like(x_dense)

    l_ex_left, = ax_exp.plot(x_dense, u_exact_dense, "k-", lw=2.5, label="Эталонное решение")
    exp_overlay_lines = []
    for r in results:
        idx = int(np.argmin(np.abs(r["t"] - t_init)))
        lbl = f"{r['name']}: h={r['h']:.3f}, τ={r['tau']:.4f} (C={r['c_max']:.2f})"
        line, = ax_exp.plot(r["x"], r["u_exp"][idx, :], ls="--", lw=1.5, label=lbl)
        exp_overlay_lines.append(line)

    ax_exp.set_title("Явная схема")
    ax_exp.set_xlabel("Координата x")
    ax_exp.set_ylabel("U(t, x)")
    ax_exp.set_xlim(cfg["x_min"], cfg["x_max"])
    ax_exp.set_ylim(full_ylim)
    ax_exp.grid(True, alpha=0.4)
    ax_exp.legend(loc="upper right", fontsize=8.5)

    l_ex_right, = ax_imp.plot(x_dense, u_exact_dense, "k-", lw=2.5, label="Эталонное решение")
    imp_overlay_lines = []
    for r in results:
        idx = int(np.argmin(np.abs(r["t"] - t_init)))
        lbl = f"{r['name']}: h={r['h']:.3f}, τ={r['tau']:.4f} (C={r['c_max']:.2f})"
        line, = ax_imp.plot(r["x"], r["u_imp"][idx, :], ls="-.", lw=1.6, label=lbl)
        imp_overlay_lines.append(line)

    ax_imp.set_title("Неявная схема")
    ax_imp.set_xlabel("Координата x")
    ax_imp.set_ylabel("U(t, x)")
    ax_imp.set_xlim(cfg["x_min"], cfg["x_max"])
    ax_imp.set_ylim(full_ylim)
    ax_imp.grid(True, alpha=0.4)
    ax_imp.legend(loc="upper right", fontsize=8.5)

    ax_slider = fig.add_axes(cfg["slider_rect"])
    time_slider = Slider(
        ax=ax_slider,
        label="Время t (с): ",
        valmin=cfg["t_min"],
        valmax=t_max,
        valinit=t_init,
        valstep=cfg["slider_step"],
        color="#ff7f0e",
    )

    def update(t_val):
        if cfg["need_analytical"]:
            u_ex_now = ref_func(t_val, x_dense)
        else:
            u_ex_now = np.zeros_like(x_dense)
            
        l_ex_left.set_ydata(u_ex_now)
        l_ex_right.set_ydata(u_ex_now)

        for i, r in enumerate(results):
            idx = int(np.argmin(np.abs(r["t"] - t_val)))
            exp_overlay_lines[i].set_ydata(r["u_exp"][idx, :])
            imp_overlay_lines[i].set_ydata(r["u_imp"][idx, :])

        fig.canvas.draw_idle()

    time_slider.on_changed(update)
    return fig, time_slider

def main():
    BOOL_NEED_ANALYTICAL = True

    config = {
        "need_analytical": BOOL_NEED_ANALYTICAL,
        "x_min": 0.0,
        "x_max": 10.0,
        "t_min": 0.0,
        "courant_limit": 1.0,
        "max_abs_clip": 5000.0,
        "t_initial_slider": 0.5,
        "slider_step": 0.05,
        "slider_rect": [0.18, 0.03, 0.65, 0.035],
        "y_padding": 5.0,
        "rmse_floor": 1e-4,
        "rmse_ylim": (1e-4, 1e4),
        "dense_points": 1000,
        "bar_width": 0.35,
        "corridor_width": 2.5,
        "fig1_size": (16, 9.5),
        "fig2_size": (15, 7.0)
    }

    grid_variants = [
        {"name": "Вариант 1", "nx": 10,  "nt": 800},
        {"name": "Вариант 2", "nx": 20,  "nt": 3200},
        {"name": "Вариант 3", "nx": 40,  "nt": 12800},
        {"name": "Вариант 4", "nx": 80,  "nt": 51200},
        {"name": "Вариант 5", "nx": 40,  "nt": 8000},
    ]

    t_max = get_t_max()

    results, ref_func = run_all_variants(grid_variants, t_max, config)
    full_ylim = get_full_zone_ylim(results, config["y_padding"])

    print_summary_table(results, t_max, config)

    fig1, slider1 = plot_variants_grid_interactive(results, t_max, full_ylim, config, ref_func)
    fig2, slider2 = plot_schemes_overlay_interactive(results, t_max, full_ylim, config, ref_func)

    plt.show()

if __name__ == "__main__":
    main()