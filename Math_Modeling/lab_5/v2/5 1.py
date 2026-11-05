import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider


def get_velocity(x, v_slope, v_offset):
    return v_slope * x + v_offset


def get_initial_condition(x, u_cap, root_left, root_right):
    return np.minimum(u_cap, (x - root_left) * (x - root_right))


def get_boundary_condition(t, t_factor, denom):
    return t * (t_factor - t) / denom


def calculate_t_max(x_min, x_max, v_slope, v_offset):
    v_start = get_velocity(x_min, v_slope, v_offset)
    v_end = get_velocity(x_max, v_slope, v_offset)
    return float(np.log(v_end / v_start) / v_slope)


def get_exact_solution(t, x, cfg):
    a = cfg["v_slope"]
    b = cfg["v_offset"]
    x_min = cfg["x_min"]
    f_val = cfg["f_source"]

    v_x = get_velocity(x, a, b)
    v_min = get_velocity(x_min, a, b)

    t_char = np.log(v_x / v_min) / a

    x0 = (v_x * np.exp(-a * t) - b) / a
    u_init = get_initial_condition(x0, cfg["u_init_cap"], cfg["x_root1"], cfg["x_root2"])
    u_zone1 = u_init + f_val * t

    t0 = t - t_char
    u_bound = get_boundary_condition(t0, cfg["t_bound_factor"], cfg["bound_denom"])
    u_zone2 = u_bound + f_val * t_char

    return np.where(t <= t_char, u_zone1, u_zone2)


def build_grid(nx, nt, x_min, x_max, t_min, t_max):
    x = np.linspace(x_min, x_max, nx + 1)
    t = np.linspace(t_min, t_max, nt + 1)
    h = float(x[1] - x[0])
    tau = float(t[1] - t[0])
    return x, t, h, tau


def solve_explicit_upwind(x, t, h, tau, cfg):
    nx = len(x) - 1
    nt = len(t) - 1
    f_val = cfg["f_source"]
    clip_lim = cfg["max_abs_clip"]

    u = np.zeros((nt + 1, nx + 1))
    u[0, :] = get_initial_condition(x, cfg["u_init_cap"], cfg["x_root1"], cfg["x_root2"])
    u[:, 0] = get_boundary_condition(t, cfg["t_bound_factor"], cfg["bound_denom"])

    v = get_velocity(x, cfg["v_slope"], cfg["v_offset"])
    gamma = tau * v / h

    for n in range(nt):
        u_next = (1.0 - gamma[1:]) * u[n, 1:] + gamma[1:] * u[n, :-1] + tau * f_val
        u[n + 1, 1:] = np.clip(u_next, -clip_lim, clip_lim)

    c_max = float(np.max(gamma))
    return u, c_max


def solve_implicit_upwind(x, t, h, tau, cfg):
    nx = len(x) - 1
    nt = len(t) - 1
    f_val = cfg["f_source"]

    u = np.zeros((nt + 1, nx + 1))
    u[0, :] = get_initial_condition(x, cfg["u_init_cap"], cfg["x_root1"], cfg["x_root2"])
    u[:, 0] = get_boundary_condition(t, cfg["t_bound_factor"], cfg["bound_denom"])

    v = get_velocity(x, cfg["v_slope"], cfg["v_offset"])
    gamma = tau * v / h

    for n in range(nt):
        for i in range(1, nx + 1):
            u[n + 1, i] = (u[n, i] + gamma[i] * u[n + 1, i - 1] + tau * f_val) / (1.0 + gamma[i])

    c_max = float(np.max(gamma))
    return u, c_max


def calculate_rmse(u_num, u_exact, axis=None):
    return np.sqrt(np.mean((u_num - u_exact) ** 2, axis=axis))


def run_all_variants(grid_variants, t_max, cfg):
    results = []

    for var in grid_variants:
        nx, nt = var["nx"], var["nt"]
        x, t, h, tau = build_grid(nx, nt, cfg["x_min"], cfg["x_max"], cfg["t_min"], t_max)

        u_exp, c_max = solve_explicit_upwind(x, t, h, tau, cfg)
        u_imp, _ = solve_implicit_upwind(x, t, h, tau, cfg)

        t_grid, x_grid = np.meshgrid(t, x, indexing="ij")
        u_exact = get_exact_solution(t_grid, x_grid, cfg)

        rmse_exp_by_t = calculate_rmse(u_exp, u_exact, axis=1)
        rmse_imp_by_t = calculate_rmse(u_imp, u_exact, axis=1)

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
            "rmse_exp_2d": float(calculate_rmse(u_exp, u_exact)),
            "rmse_imp_2d": float(calculate_rmse(u_imp, u_exact)),
        })

    return results


def get_full_zone_ylim(results, y_padding):
    u_min = min(float(np.min(r["u_exact"])) for r in results)
    u_max = max(float(np.max(r["u_exact"])) for r in results)
    return (u_min - y_padding, u_max + y_padding)


def print_summary_table(results, t_max, cfg):
    t_init = cfg["t_initial_slider"]
    print(f"Время полного прохождения области: t_max = {t_max:.6f} с")
    print(f"Условие устойчивости явной схемы: C_max <= {cfg['courant_limit']}\n")

    header = (
        f"{'Вариант':<12} | {'h':<6} | {'tau':<8} | {'C_max':<6} | {'Устойчивость':<14} | "
        f"{f'СКО Явн (t={t_init})':<17} | {f'СКО Неявн (t={t_init})':<19} | "
        f"{'СКО Явная (2D)':<14} | {'СКО Неявная (2D)':<16}"
    )
    print(header)
    print("-" * len(header))

    for r in results:
        idx = int(np.argmin(np.abs(r["t"] - t_init)))
        status = "Устойчива" if r["is_stable"] else "НЕУСТОЙЧИВА"
        print(
            f"{r['name']:<12} | {r['h']:<6.3f} | {r['tau']:<8.5f} | {r['c_max']:<6.3f} | "
            f"{status:<14} | {r['rmse_exp_by_t'][idx]:<17.5e} | {r['rmse_imp_by_t'][idx]:<19.5e} | "
            f"{r['rmse_exp_2d']:<14.5e} | {r['rmse_imp_2d']:<16.5e}"
        )


def plot_variants_grid_interactive(results, t_max, full_ylim, cfg):
    plt.rcParams.update({"font.size": 9.5})
    fig, axes = plt.subplots(2, 3, figsize=cfg["fig1_size"])
    plt.subplots_adjust(left=0.06, right=0.98, top=0.89, bottom=0.14, hspace=0.38, wspace=0.24)

    t_init = cfg["t_initial_slider"]
    fig.suptitle(
        f"Сравнение 5 вариантов между схемами",
        fontsize=13,
        fontweight="bold",
    )

    x_dense = np.linspace(cfg["x_min"], cfg["x_max"], cfg["dense_points"])
    u_exact_dense = get_exact_solution(t_init, x_dense, cfg)
    flat_axes = axes.flatten()

    exact_lines = []
    exp_lines = []
    imp_lines = []
    legends = []

    for i, r in enumerate(results):
        ax = flat_axes[i]
        idx = int(np.argmin(np.abs(r["t"] - t_init)))

        l_ex, = ax.plot(x_dense, u_exact_dense, "k-", lw=2.0, label="Аналитическое")
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
            f"{r['name']}\nh={r['h']:.3f}, τ={r['tau']:.4f} | C_max={r['c_max']:.2f} ({status_tag})",
            fontsize=9.5,
        )
        ax.set_xlabel("Координата x")
        ax.set_ylabel("U(t, x)")
        ax.set_xlim(cfg["x_min"], cfg["x_max"])
        ax.set_ylim(full_ylim)
        ax.grid(True, alpha=0.4)
        leg = ax.legend(loc="upper right", fontsize=8)

        exact_lines.append(l_ex)
        exp_lines.append(l_exp)
        imp_lines.append(l_imp)
        legends.append(leg)

    ax_err = flat_axes[5]
    indices = np.arange(len(results))
    bar_w = cfg["bar_width"]
    labels = [f"{r['name']}\n(h={r['h']:.2g})" for r in results]

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

    ax_err.set_yscale("log")
    ax_err.set_ylim(cfg["rmse_ylim"])
    ax_err.set_xticks(indices)
    ax_err.set_xticklabels(labels)
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
        u_ex_now = get_exact_solution(t_val, x_dense, cfg)

        for i, r in enumerate(results):
            idx = int(np.argmin(np.abs(r["t"] - t_val)))
            exact_lines[i].set_ydata(u_ex_now)
            exp_lines[i].set_ydata(r["u_exp"][idx, :])
            imp_lines[i].set_ydata(r["u_imp"][idx, :])

            err_e = r["rmse_exp_by_t"][idx]
            err_i = r["rmse_imp_by_t"][idx]

            texts = legends[i].get_texts()
            texts[1].set_text(f"Явный (СКО={err_e:.3f})")
            texts[2].set_text(f"Неявный (СКО={err_i:.3f})")

            bars_exp[i].set_height(max(err_e, cfg["rmse_floor"]))
            bars_imp[i].set_height(max(err_i, cfg["rmse_floor"]))

        fig.canvas.draw_idle()

    time_slider.on_changed(update)
    return fig, time_slider


def plot_schemes_overlay_interactive(results, t_max, full_ylim, cfg):
    fig, (ax_exp, ax_imp) = plt.subplots(1, 2, figsize=cfg["fig2_size"])
    plt.subplots_adjust(left=0.07, right=0.98, top=0.86, bottom=0.18, wspace=0.20)

    t_init = cfg["t_initial_slider"]
    fig.suptitle(
        "Сравнение 5 вариантов сеток внутри схем",
        fontsize=13,
        fontweight="bold",
    )

    x_dense = np.linspace(cfg["x_min"], cfg["x_max"], cfg["dense_points"])
    u_exact_dense = get_exact_solution(t_init, x_dense, cfg)

    l_ex_left, = ax_exp.plot(x_dense, u_exact_dense, "k-", lw=2.5, label="Аналитическое решение")
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

    l_ex_right, = ax_imp.plot(x_dense, u_exact_dense, "k-", lw=2.5, label="Аналитическое решение")
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
        u_ex_now = get_exact_solution(t_val, x_dense, cfg)
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
    config = {
        "x_min": 0.0,
        "x_max": 10.0,
        "t_min": 0.0,
        "v_slope": 1.0,
        "v_offset": 1.0,
        "f_source": -19.1,
        "u_init_cap": 0.0,
        "x_root1": 3.0,
        "x_root2": 8.0,
        "t_bound_factor": 100.0,
        "bound_denom": 2500.0,
        "courant_limit": 1.0,
        "max_abs_clip": 1000.0,
        "t_initial_slider": 0.45,
        "slider_step": 0.01,
        "slider_rect": [0.18, 0.03, 0.65, 0.035],
        "y_padding": 3.0,
        "rmse_floor": 1e-5,
        "rmse_ylim": (1e-4, 1e3),
        "dense_points": 1000,
        "bar_width": 0.35,
        "fig1_size": (16, 9.5),
        "fig2_size": (15, 7.0),
        "plot_title": "",
    }

    grid_variants = [
        {"name": "Вариант 1", "nx": 25,  "nt": 80},
        {"name": "Вариант 2", "nx": 50,  "nt": 160},
        {"name": "Вариант 3", "nx": 100, "nt": 320},
        {"name": "Вариант 4", "nx": 400, "nt": 1280},
        {"name": "Вариант 5", "nx": 150, "nt": 240},
    ]

    t_max = calculate_t_max(
        config["x_min"], config["x_max"], config["v_slope"], config["v_offset"]
    )

    results = run_all_variants(grid_variants, t_max, config)
    full_ylim = get_full_zone_ylim(results, config["y_padding"])

    print_summary_table(results, t_max, config)

    fig1, slider1 = plot_variants_grid_interactive(results, t_max, full_ylim, config)
    fig2, slider2 = plot_schemes_overlay_interactive(results, t_max, full_ylim, config)

    plt.show()


if __name__ == "__main__":
    main()