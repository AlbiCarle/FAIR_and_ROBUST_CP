import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter

from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

def plot_cp_size(results, alpha_fill=0.1, epsilon_star=0.1, name = "adult"):

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 17,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14
    })

    eps = np.array([r["eps"] for r in results])

    # ===== METHODS =====
    methods = {
        "V": "natural_vanilla",
        "EQ": "natural_EQ",
        "EQ_k_a": "natural_EQ_ka",
        "EQ_max": "natural_EQ_max"
    }

    percp_methods = {
        "V": "percp_vanilla",
        "EQ": "percp_EQ",
        "EQ_k_a": "percp_EQ_ka",
        "EQ_max": "percp_EQ_max"
    }

    colors = {
    "V": "#2f2f2f",   
    "EQ": "#d62728",
    "EQ_k_a": "#1f77b4",
    "EQ_max": "#2ca02c"
    }

    colors_cf = {
        "V": "#8c8c8c",   
        "EQ": "#ff9896",
        "EQ_k_a": "#aec7e8",
        "EQ_max": "#98df8a"
    }

    if name == "bank":
        
        
        groups = {
            "Global": 0,
            "Married": 1,
            "Single": 2
        }
    
    elif name == "compas":
        
        groups = {
            "Global": 0,
            "White": 1,
            "Non-white": 2
        }
        
    else:
           
        groups = {
            "Global": 0,
            "Female": 1,
            "Male": 2
        }

    linewidths = {k: 2.8 for k in methods.keys()}

    def get(key, field, idx):
        return np.array([r[key][field][idx] for r in results])

    def get_ci(key, field, idx):
        return np.array([r[key]["ci"][field][idx] for r in results])

    def add_markers(ax, x, y):

        x = np.array(x)
        y = np.array(y)

        idx0 = np.argmin(np.abs(x - 0))
        idx_star = np.argmin(np.abs(x - epsilon_star))

        # ● epsilon = 0
        ax.scatter(
            x[idx0], y[idx0],
            color="black",
            edgecolors="white",   # stacca dal fill
            linewidths=1.2,
            marker='o',
            s=70,
            zorder=10
        )

        # ★ epsilon = epsilon_star
        ax.scatter(
            x[idx_star], y[idx_star],
            color="black",
            edgecolors="white",
            linewidths=1.2,
            marker='*',
            s=180,
            zorder=11
        )

    def plot_block(method_dict, title):

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for ax, (gname, idx) in zip(axes, groups.items()):

            for m, key in method_dict.items():

                lw = linewidths[m]

                # ===== FACTUAL =====
                y = get(key, "size", idx)
                ci = get_ci(key, "size", idx)

                ax.plot(
                    eps, y,
                    color=colors[m],
                    linestyle="-",
                    linewidth=lw
                )

                ax.fill_between(
                    eps,
                    y - ci * 2.1,
                    y + ci * 2.1,
                    color=colors[m],
                    alpha=alpha_fill
                )

                #add_markers(ax, eps, y, colors[m])
                add_markers(ax, eps, y)
                # ===== COUNTERFACTUAL =====
                idx_cf = (-idx) % 3

                y_cf = get(key, "size_cf", idx_cf)
                ci_cf = get_ci(key, "size_cf", idx_cf)

                ax.plot(
                    eps, y_cf,
                    color=colors_cf[m],
                    linestyle="--",
                    linewidth=lw
                )

                ax.fill_between(
                    eps,
                    y_cf - ci_cf * 2.1,
                    y_cf + ci_cf * 2.1,
                    color=colors_cf[m],
                    alpha=alpha_fill
                )

                #add_markers(ax, eps, y_cf, colors_cf[m])
                add_markers(ax, eps, y_cf)
                
            ax.set_title(gname)
            ax.set_xlabel("ε")
            ax.set_ylabel("Average Set Size")

            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            ax.tick_params(axis='both', labelsize=14)
            ax.grid(alpha=0.25)

        fig.suptitle(title, fontsize=20)

        plt.tight_layout()
        plt.show()

    # ===== CALLS =====
    plot_block(methods, "Adversarial Setting (No Defense)")
    plot_block(percp_methods, "Adversarial Setting (With Defense)")

def plot_cp_coverage(results, alpha_fill=0.1, epsilon_star=0.1, name = "adult"):

    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 16,
        "axes.titlesize": 18,
        "axes.labelsize": 17,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14
    })

    eps = np.array([r["eps"] for r in results])

    # ===== METHODS =====
    methods = {
        "V": "natural_vanilla",
        "EQ": "natural_EQ",
        "EQ_k_a": "natural_EQ_ka",
        "EQ_max": "natural_EQ_max"
    }

    percp_methods = {
        "V": "percp_vanilla",
        "EQ": "percp_EQ",
        "EQ_k_a": "percp_EQ_ka",
        "EQ_max": "percp_EQ_max"
    }

    colors = {
    "V": "#2f2f2f",   
    "EQ": "#d62728",
    "EQ_k_a": "#1f77b4",
    "EQ_max": "#2ca02c"
    }

    colors_cf = {
        "V": "#8c8c8c",   
        "EQ": "#ff9896",
        "EQ_k_a": "#aec7e8",
        "EQ_max": "#98df8a"
    }
    
    if name == "bank":
        
        
        groups = {
            "Global": 0,
            "Married": 1,
            "Single": 2
        }
    
    elif name == "compas":
        
        groups = {
            "Global": 0,
            "White": 1,
            "Non-white": 2
        }
        
    else:
           
        groups = {
            "Global": 0,
            "Female": 1,
            "Male": 2
        }


    linewidths = {k: 2.8 for k in methods.keys()}

    def get(key, field, idx):
        return np.array([r[key][field][idx] for r in results])

    def get_ci(key, field, idx):
        return np.array([r[key]["ci"][field][idx] for r in results])

    def add_markers(ax, x, y):

        x = np.array(x)
        y = np.array(y)

        idx0 = np.argmin(np.abs(x - 0))
        idx_star = np.argmin(np.abs(x - epsilon_star))

        # ● epsilon = 0
        ax.scatter(
            x[idx0], y[idx0],
            color="black",
            edgecolors="white",   # stacca dal fill
            linewidths=1.2,
            marker='o',
            s=70,
            zorder=10
        )

        # ★ epsilon = epsilon_star
        ax.scatter(
            x[idx_star], y[idx_star],
            color="black",
            edgecolors="white",
            linewidths=1.2,
            marker='*',
            s=180,
            zorder=11
        )

    def plot_block(method_dict, title):

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        for ax, (gname, idx) in zip(axes, groups.items()):

            for m, key in method_dict.items():

                lw = linewidths[m]

                # ===== FACTUAL =====
                y = get(key, "cov", idx)
                ci = get_ci(key, "cov", idx)

                ax.plot(
                    eps, y,
                    color=colors[m],
                    linestyle="-",
                    linewidth=lw
                )

                ax.fill_between(
                    eps,
                    y - ci * 2.1,
                    y + ci * 2.1,
                    color=colors[m],
                    alpha=alpha_fill
                )

                #add_markers(ax, eps, y, colors[m])
                add_markers(ax, eps, y)
                # ===== COUNTERFACTUAL =====
                idx_cf = (-idx) % 3

                y_cf = get(key, "cov_cf", idx_cf)
                ci_cf = get_ci(key, "cov_cf", idx_cf)

                ax.plot(
                    eps, y_cf,
                    color=colors_cf[m],
                    linestyle="--",
                    linewidth=lw
                )

                ax.fill_between(
                    eps,
                    y_cf - ci_cf * 2.1,
                    y_cf + ci_cf * 2.1,
                    color=colors_cf[m],
                    alpha=alpha_fill
                )

                #add_markers(ax, eps, y_cf, colors_cf[m])
                add_markers(ax, eps, y_cf)
                
            ax.set_title(gname)
            ax.set_xlabel("ε")
            ax.set_ylabel("Coverage")

            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))

            ax.tick_params(axis='both', labelsize=14)
            ax.grid(alpha=0.25)

        fig.suptitle(title, fontsize=20)

        plt.tight_layout()
        plt.show()

    # ===== CALLS =====
    plot_block(methods, "Adversarial Setting (No Defense)")
    plot_block(percp_methods, "Adversarial Setting (With Defense)")

def add_global_legend(epsilon_star=0.1):

    # ===== COLORS =====
    colors = {
        "V": "#2f2f2f",   # grigio più scuro
        "EQ": "#d62728",
        "EQ_k_a": "#1f77b4",
        "EQ_max": "#2ca02c"
    }

    # ===== METHODS =====
    method_handles = [
        Line2D([0], [0], color=colors["V"], lw=2.0, label=r"$\mathcal{C}$"),
        Line2D([0], [0], color=colors["EQ"], lw=2.0, label=r"$\mathcal{C}^{EQ}$"),
        Line2D([0], [0], color=colors["EQ_k_a"], lw=2.0, label=r"$\mathcal{C}^{EQ}_{k_a}$"),
        Line2D([0], [0], color=colors["EQ_max"], lw=2.0, label=r"$\mathcal{C}^{EQ}_{max}$"),
    ]

    # ===== STYLE =====
    style_handles = [
        Line2D([0], [0], color="black", lw=2.0, linestyle="-", label="Factual"),
        Line2D([0], [0], color="black", lw=2.0, linestyle="--", label="Counterfactual"),
    ]

    # ===== MARKERS =====
    marker_handles = [
        Line2D(
            [0], [0],
            marker='o',
            color='black',
            markerfacecolor='black',
            markeredgecolor='white',
            markeredgewidth=1.2,
            markersize=7,
            linestyle='None',
            label=r"$\epsilon = 0$"
        ),
        Line2D(
            [0], [0],
            marker='*',
            color='black',
            markerfacecolor='black',
            markeredgecolor='white',
            markeredgewidth=1.2,
            markersize=11,
            linestyle='None',
            label=rf"$\epsilon = {epsilon_star}$"
        ),
    ]

    # ===== FIGURE =====
    fig = plt.figure(figsize=(12, 1.4))

    fig.legend(
        handles=method_handles + style_handles + marker_handles,
        loc="center",
        ncol=8,                 # tutto su una riga
        frameon=False,
        fontsize=13,
        handlelength=1.6,       # accorcia le linee
        columnspacing=1.2       # compatta orizzontalmente
    )

    plt.axis("off")
    plt.tight_layout()
    plt.show()
