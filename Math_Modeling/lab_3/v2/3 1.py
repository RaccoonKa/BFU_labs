import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Dict
from matplotlib.widgets import Slider

def target_function(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    return x**2 + y**3

def is_inside_spirals(x: np.ndarray, y: np.ndarray, num_turns: int = 3) -> np.ndarray:
    r = np.sqrt(x**2 + y**2)
    theta = np.arctan2(y, x)
    theta = np.where(theta < 0, theta + 2 * np.pi, theta)
    
    mask = np.zeros_like(r, dtype=bool)
    for k in range(num_turns):
        phi = theta + 2 * np.pi * k
        mask |= (r >= phi) & (r <= phi + np.pi)
    return mask

def run_monte_carlo(N: int, R_max: float, exact_area: float, exact_integral: float) -> Dict[str, float]:
    S_box = (2 * R_max)**2
    
    x = np.random.uniform(-R_max, R_max, N)
    y = np.random.uniform(-R_max, R_max, N)
    
    mask = is_inside_spirals(x, y)
    area_mc = S_box * np.mean(mask)
    
    area_err_rel = (np.abs(area_mc - exact_area) / exact_area) * 100
    area_stat_err = S_box * (np.std(mask) / np.sqrt(N)) 
    
    f_vals = target_function(x, y) * mask
    integral_mc = S_box * np.mean(f_vals)
    
    integral_err_rel = (np.abs(integral_mc - exact_integral) / np.abs(exact_integral)) * 100
    integral_stat_err = S_box * (np.std(f_vals) / np.sqrt(N))
    
    return {
        'N': N,
        'Area': area_mc,
        'Area_Err_%': area_err_rel,
        'Area_eps': area_stat_err,
        'Integral': integral_mc,
        'Int_Exact': exact_integral,
        'Int_Err_%': integral_err_rel,
        'Int_eps': integral_stat_err
    }

def plot_results(df: pd.DataFrame, R_max: float, exact_area: float, exact_integral: float):
    plt.style.use('seaborn-v0_8-whitegrid')

    fig, axs = plt.subplots(2, 2, figsize=(14, 10))
    fig.subplots_adjust(bottom=0.15, hspace=0.3, wspace=0.25) 

    ax_geom = axs[0, 0]
    N_max_slider = 10000
    x_all = np.random.uniform(-R_max, R_max, N_max_slider)
    y_all = np.random.uniform(-R_max, R_max, N_max_slider)
    mask_all = is_inside_spirals(x_all, y_all)

    scat_out = ax_geom.scatter(x_all[:100][~mask_all[:100]], y_all[:100][~mask_all[:100]], 
                              s=1, alpha=0.3, color='royalblue', label='Мимо')
    scat_in = ax_geom.scatter(x_all[:100][mask_all[:100]], y_all[:100][mask_all[:100]], 
                             s=1, color='tab:orange', label='Внутри')
    
    phi_ideal = np.linspace(0, 6*np.pi, 1000)
    ax_geom.plot(phi_ideal * np.cos(phi_ideal), phi_ideal * np.sin(phi_ideal), 'k-', linewidth=1, label=r'$r=\varphi$')
    ax_geom.plot((phi_ideal+np.pi) * np.cos(phi_ideal), (phi_ideal+np.pi) * np.sin(phi_ideal), 'k--', linewidth=1, label=r'$r=\varphi+\pi$')
    
    ax_geom.set_title('Область интеграции (N=100)')
    ax_geom.set_aspect('equal', 'box')
    ax_geom.legend(loc='upper right')

    ax_slider = fig.add_axes([0.15, 0.05, 0.3, 0.03])
    allowed_Ns = np.concatenate([
        np.arange(10, 100, 10),
        np.arange(100, 1000, 100),
        np.arange(1000, 10001, 1000)
    ])
    
    slider = Slider(ax=ax_slider, label='Точек (N)', valmin=0, valmax=len(allowed_Ns) - 1, valinit=9, valstep=1)
    slider.valtext.set_text('100')

    def update(val):
        idx = int(slider.val)
        n = allowed_Ns[idx]
        slider.valtext.set_text(str(n)) 
        
        out_mask = ~mask_all[:n]
        in_mask = mask_all[:n]
        scat_out.set_offsets(np.c_[x_all[:n][out_mask], y_all[:n][out_mask]])
        scat_in.set_offsets(np.c_[x_all[:n][in_mask], y_all[:n][in_mask]])
        
        ax_geom.set_title(f'Область интеграции (N={n})')
        fig.canvas.draw_idle()
        
    slider.on_changed(update)

    ax_area = axs[0, 1]
    ax_area.errorbar(df['N'], df['Area'], yerr=df['Area_eps'], fmt='-o', capsize=4, color='forestgreen', label=r'Площадь $\pm \varepsilon$')
    ax_area.axhline(exact_area, color='r', linestyle='--', label='Ожидаемая площадь')
    ax_area.set_xscale('log')
    ax_area.set_title('Сходимость расчета площади')
    ax_area.set_xlabel('Количество точек N')
    ax_area.set_ylabel('Площадь')
    ax_area.legend()

    ax_int = axs[1, 0]
    ax_int.errorbar(df['N'], df['Integral'], yerr=df['Int_eps'], fmt='-o', capsize=4, color='purple', label=r'Интеграл $\pm \varepsilon$')

    ax_int.axhline(exact_integral, color='r', linestyle='--', label='Точный интеграл')
    ax_int.set_xscale('log')
    ax_int.set_title('Сходимость расчета интеграла')
    ax_int.set_xlabel('Количество точек N')
    ax_int.set_ylabel('Значение интеграла')
    ax_int.legend()

    ax_err = axs[1, 1]
    ax_err_int = ax_err.twinx()
    
    ax_err.loglog(df['N'], df['Area_eps'], marker='o', color='forestgreen', label=r'$\varepsilon$ площади')
    ax_err_int.loglog(df['N'], df['Int_eps'], marker='s', color='purple', label=r'$\varepsilon$ интеграла')
    
    #ax_err.plot(df['N'], df['Area_eps'], marker='o', color='forestgreen', label=r'$\varepsilon$ площади')
    #ax_err_int.plot(df['N'], df['Int_eps'], marker='s', color='purple', label=r'$\varepsilon$ интеграла')
    
    ax_err.set_title('Убывание статистической погрешности')
    ax_err.set_xlabel('Количество точек N')
    ax_err.set_ylabel(r'$\varepsilon$ площади', color='forestgreen')
    ax_err_int.set_ylabel(r'$\varepsilon$ интеграла', color='purple')
    
    lines_1, labels_1 = ax_err.get_legend_handles_labels()
    lines_2, labels_2 = ax_err_int.get_legend_handles_labels()
    ax_err.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right')

    plt.show()
    return slider

def main():
    N_values = [10**3, 5*10**3, 10**4, 5*10**4, 10**5, 5*10**5, 10**6]
    R_max = 7 * np.pi
    
    exact_area = 21 * np.pi**3
    exact_integral = (4669 * np.pi**3 / 12) - (3913 * np.pi**5 / 4)
    
    np.random.seed(42)
    print(f"Запуск симуляции Монте-Карло")
    print(f"Точная площадь: {exact_area:.4f}")
    print(f"Точный интеграл: {exact_integral:.4f}\n")
    
    results: List[Dict] = []
    for N in N_values:
        experiment_result = run_monte_carlo(N, R_max, exact_area, exact_integral)
        results.append(experiment_result)
        
    df = pd.DataFrame(results)
    pd.options.display.float_format = '{:.4f}'.format 
    
    print("Таблица результатов:")
    print(df.to_string(index=False))
    
    _keep_slider_alive = plot_results(df, R_max, exact_area, exact_integral)

if __name__ == "__main__":
    main()