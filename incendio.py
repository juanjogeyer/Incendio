import numpy as np
import matplotlib.pyplot as plt
from numba import njit
import time
from collections import deque

# ============================================================================
# SIMULACIÓN DE PROPAGACIÓN DE INCENDIO FORESTAL
# Basado en percolación - Relacionado con fractales y DLA
# Incluye análisis de dimensión fractal del área quemada
# ============================================================================

@njit(cache=True)
def _burn_cluster_bfs(grid, start_row, start_col, L):
    """
    Propaga el fuego usando BFS (Breadth-First Search) desde un punto inicial.
    Cuenta cuántos árboles se queman en el cluster conectado.
    
    Returns:
        burned_count: número de árboles quemados
        visited: matriz booleana con las celdas quemadas
    """
    visited = np.zeros((L, L), dtype=np.bool_)
    
    # Si el punto inicial no tiene árbol, no se quema nada
    if grid[start_row, start_col] == 0:
        return 0, visited
    
    # Cola para BFS (usamos un array pre-allocado para eficiencia)
    max_queue_size = L * L
    queue = np.zeros((max_queue_size, 2), dtype=np.int32)
    queue_start = 0
    queue_end = 0
    
    # Agregar punto inicial
    queue[queue_end, 0] = start_row
    queue[queue_end, 1] = start_col
    queue_end += 1
    visited[start_row, start_col] = True
    burned_count = 1
    
    # Direcciones: arriba, abajo, izquierda, derecha
    directions = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]], dtype=np.int32)
    
    while queue_start < queue_end:
        # Sacar elemento de la cola
        current_row = queue[queue_start, 0]
        current_col = queue[queue_start, 1]
        queue_start += 1
        
        # Explorar vecinos
        for d in range(4):
            new_row = current_row + directions[d, 0]
            new_col = current_col + directions[d, 1]
            
            # Verificar límites
            if new_row < 0 or new_row >= L or new_col < 0 or new_col >= L:
                continue
            
            # Si hay árbol y no ha sido visitado, quemarlo
            if grid[new_row, new_col] == 1 and not visited[new_row, new_col]:
                visited[new_row, new_col] = True
                queue[queue_end, 0] = new_row
                queue[queue_end, 1] = new_col
                queue_end += 1
                burned_count += 1
    
    return burned_count, visited


# ============================================================================
# CÁLCULO DE DIMENSIÓN FRACTAL (Inspirado en DLA)
# ============================================================================

def calculate_fractal_dimension(burned_grid, method='mass-radius', recentralize=True):
    """
    Calcula la dimensión fractal del área quemada usando Mass-Radius Relation.
    Similar al método sandbox usado en DLA.
    
    Args:
        burned_grid: matriz booleana con las celdas quemadas
        method: método a usar ('mass-radius')
        recentralize: si True, usa centro de masa; si False, usa centro geométrico
        
    Returns:
        dict con dimensión fractal, R², datos del ajuste, y puntos para graficar
    """
    # Obtener coordenadas de píxeles quemados
    rows, cols = np.where(burned_grid)
    
    if len(rows) < 10:
        # Muy pocos puntos para análisis fractal
        return None
    
    # Calcular centro (de masa o geométrico)
    if recentralize:
        cx = np.mean(rows)
        cy = np.mean(cols)
    else:
        cx = burned_grid.shape[0] / 2.0
        cy = burned_grid.shape[1] / 2.0
    
    # Calcular distancias desde el centro
    dr = rows - cx
    dc = cols - cy
    distances = np.sqrt(dr**2 + dc**2)
    
    if len(distances) == 0:
        return None
    
    # Definir radios para análisis (log-uniformes)
    R_min = max(1.0, np.min(distances))
    R_max = np.max(distances)
    
    if R_max < 2 * R_min:
        return None
    
    # Crear radios logarítmicamente espaciados
    n_radii = min(100, int(len(distances) / 3))
    radii = np.geomspace(R_min, R_max, num=n_radii)
    radii = np.unique(radii)
    
    # Contar masa (píxeles) dentro de cada radio
    distances_sorted = np.sort(distances)
    masses = np.searchsorted(distances_sorted, radii, side='right')
    
    # Filtrar radios válidos (con suficiente masa)
    valid_mask = (masses > 5) & (masses < len(distances) * 0.95)
    
    if np.sum(valid_mask) < 10:
        # Muy pocos puntos para ajuste confiable
        return None
    
    radii_valid = radii[valid_mask]
    masses_valid = masses[valid_mask]
    
    # Logaritmos para ajuste lineal
    log_R = np.log(radii_valid)
    log_M = np.log(masses_valid)
    
    # Ajuste lineal: log(M) = D_f * log(R) + const
    coeffs = np.polyfit(log_R, log_M, 1)
    D_f = coeffs[0]  # Pendiente = dimensión fractal
    
    # Calcular R² (bondad de ajuste)
    log_M_pred = np.polyval(coeffs, log_R)
    ss_res = np.sum((log_M - log_M_pred)**2)
    ss_tot = np.sum((log_M - np.mean(log_M))**2)
    R_squared = 1.0 - (ss_res / (ss_tot + 1e-12))
    
    return {
        'D_f': D_f,
        'R2': R_squared,
        'log_R': log_R,
        'log_M': log_M,
        'coeffs': coeffs,
        'center': (cx, cy),
        'n_points': len(log_R),
        'R_range': (float(np.min(radii_valid)), float(np.max(radii_valid))),
        'total_mass': len(distances)
    }


def plot_fractal_analysis(burned_grid, fractal_data, result=None, save_path=None):
    """
    Visualiza el análisis fractal del área quemada.
    """
    if fractal_data is None:
        print("No hay suficientes datos para análisis fractal")
        return
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Panel 1: Área quemada con centro de masa
    ax = axes[0]
    ax.imshow(burned_grid, cmap='Reds', interpolation='nearest')
    cx, cy = fractal_data['center']
    ax.plot(cy, cx, 'b*', markersize=20, markeredgecolor='yellow', 
            markeredgewidth=2, label='Centro de masa')
    
    # Dibujar algunos círculos de referencia
    theta = np.linspace(0, 2*np.pi, 100)
    R_min, R_max = fractal_data['R_range']
    for R in [R_min, (R_min + R_max)/2, R_max]:
        circle_x = cy + R * np.cos(theta)
        circle_y = cx + R * np.sin(theta)
        ax.plot(circle_x, circle_y, 'cyan', alpha=0.5, linewidth=1.5)
    
    title = f'Área Quemada con Centro de Masa\n(Total: {fractal_data["total_mass"]} píxeles)'
    if result is not None:
        title += f'\np={result["p"]:.2f}'
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.legend(loc='upper right')
    ax.axis('equal')
    ax.axis('off')
    
    # Panel 2: Ajuste fractal log-log
    ax = axes[1]
    log_R = fractal_data['log_R']
    log_M = fractal_data['log_M']
    coeffs = fractal_data['coeffs']
    
    ax.plot(log_R, log_M, 'o', color='darkred', markersize=6, alpha=0.6, label='Datos')
    ax.plot(log_R, np.polyval(coeffs, log_R), 'b--', linewidth=3, 
            label=f'Ajuste: $D_f$ = {fractal_data["D_f"]:.3f}')
    
    ax.set_xlabel('log(R) - Radio desde centro de masa', fontsize=12)
    ax.set_ylabel('log(M) - Masa acumulada', fontsize=12)
    ax.set_title(f'Dimensión Fractal del Incendio\n$R^2$ = {fractal_data["R2"]:.4f} | {fractal_data["n_points"]} puntos', 
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.4, linestyle='--')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figura guardada: {save_path}")
    plt.show()


def simulate_single_fire(L, p, seed=None):
    """
    Simula un único incendio forestal.
    
    Args:
        L: tamaño de la grilla (LxL)
        p: probabilidad de que una celda tenga árbol
        seed: semilla para reproducibilidad
        
    Returns:
        dict con resultados de la simulación
    """
    if seed is not None:
        np.random.seed(seed)
    
    # Crear bosque con densidad p
    grid = (np.random.rand(L, L) < p).astype(np.int32)
    total_trees = np.sum(grid)
    
    # Elegir punto aleatorio para iniciar fuego
    start_row = np.random.randint(0, L)
    start_col = np.random.randint(0, L)
    
    # Simular propagación
    burned_count, burned_grid = _burn_cluster_bfs(grid, start_row, start_col, L)
    
    # Calcular métricas
    fraction_of_trees = burned_count / total_trees if total_trees > 0 else 0
    fraction_of_surface = burned_count / (L * L)
    
    # Calcular dimensión fractal si hay suficiente área quemada
    fractal_data = None
    if burned_count > 20:
        fractal_data = calculate_fractal_dimension(burned_grid, recentralize=True)
    
    return {
        'grid': grid,
        'burned_grid': burned_grid,
        'start_pos': (start_row, start_col),
        'total_trees': total_trees,
        'burned_count': burned_count,
        'fraction_of_trees': fraction_of_trees,
        'fraction_of_surface': fraction_of_surface,
        'fractal_data': fractal_data,
        'p': p,
        'L': L
    }


def simulate_multiple_fires(L, p, n_simulations=1000, verbose=True):
    """
    Realiza múltiples simulaciones de incendio para obtener estadísticas.
    
    Args:
        L: tamaño de la grilla
        p: densidad de árboles
        n_simulations: número de simulaciones a realizar
        verbose: mostrar progreso
        
    Returns:
        dict con arrays de fracciones quemadas y dimensiones fractales
    """
    fractions_trees = []
    fractions_surface = []
    fractal_dimensions = []
    
    if verbose:
        print(f"\nSimulando {n_simulations} incendios con p={p:.2f}, L={L}")
        t0 = time.time()
    
    for i in range(n_simulations):
        result = simulate_single_fire(L, p, seed=None)
        fractions_trees.append(result['fraction_of_trees'])
        fractions_surface.append(result['fraction_of_surface'])
        
        # Guardar dimensión fractal si existe
        if result['fractal_data'] is not None and result['fractal_data']['R2'] > 0.8:
            fractal_dimensions.append(result['fractal_data']['D_f'])
        
        if verbose and (i + 1) % 200 == 0:
            print(f"  {i+1}/{n_simulations} simulaciones completadas...")
    
    if verbose:
        elapsed = time.time() - t0
        mean_D_f = np.mean(fractal_dimensions) if len(fractal_dimensions) > 0 else 0
        print(f"  ✓ Completado en {elapsed:.2f}s | D_f promedio: {mean_D_f:.3f} ({len(fractal_dimensions)} muestras)")
    
    return {
        'fractions_trees': np.array(fractions_trees),
        'fractions_surface': np.array(fractions_surface),
        'fractal_dimensions': np.array(fractal_dimensions),
        'p': p,
        'L': L,
        'n_simulations': n_simulations
    }


def plot_fire_example(result, save_path=None, show_fractal=True):
    """
    Visualiza un ejemplo de incendio forestal.
    """
    n_cols = 3 if not show_fractal or result['fractal_data'] is None else 4
    fig, axes = plt.subplots(1, n_cols, figsize=(5*n_cols, 5))
    
    # Bosque original
    ax = axes[0]
    ax.imshow(result['grid'], cmap='Greens', interpolation='nearest')
    ax.plot(result['start_pos'][1], result['start_pos'][0], 'r*', markersize=15, 
            markeredgecolor='white', markeredgewidth=1.5, label='Inicio fuego')
    ax.set_title(f'Bosque Original\n(p={result["p"]:.2f}, {result["total_trees"]} árboles)', fontsize=12)
    ax.legend(loc='upper right')
    ax.axis('off')
    
    # Área quemada
    ax = axes[1]
    ax.imshow(result['burned_grid'], cmap='Reds', interpolation='nearest')
    ax.plot(result['start_pos'][1], result['start_pos'][0], 'y*', markersize=15,
            markeredgecolor='black', markeredgewidth=1.5)
    ax.set_title(f'Área Quemada\n({result["burned_count"]} árboles)', fontsize=12)
    ax.axis('off')
    
    # Superposición
    ax = axes[2]
    # Crear imagen RGB
    display = np.zeros((result['L'], result['L'], 3))
    display[:, :, 1] = result['grid'] * 0.5  # Verde para árboles no quemados
    display[:, :, 0] = result['burned_grid'] * 1.0  # Rojo para quemados
    ax.imshow(display, interpolation='nearest')
    ax.plot(result['start_pos'][1], result['start_pos'][0], 'y*', markersize=15,
            markeredgecolor='black', markeredgewidth=1.5)
    ax.set_title(f'Composición\n(Fracción quemada: {result["fraction_of_trees"]:.1%})', fontsize=12)
    ax.axis('off')
    
    # Análisis fractal (si existe)
    if show_fractal and result['fractal_data'] is not None and n_cols == 4:
        ax = axes[3]
        fractal_data = result['fractal_data']
        log_R = fractal_data['log_R']
        log_M = fractal_data['log_M']
        coeffs = fractal_data['coeffs']
        
        ax.plot(log_R, log_M, 'o', color='darkred', markersize=5, alpha=0.6, label='Datos')
        ax.plot(log_R, np.polyval(coeffs, log_R), 'b--', linewidth=2.5, 
                label=f'$D_f$ = {fractal_data["D_f"]:.3f}')
        ax.set_xlabel('log(R)', fontsize=11)
        ax.set_ylabel('log(M)', fontsize=11)
        ax.set_title(f'Dimensión Fractal\n$R^2$ = {fractal_data["R2"]:.3f}', fontsize=12)
        ax.legend(fontsize=10)
        ax.grid(alpha=0.4)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figura guardada: {save_path}")
    plt.show()


def plot_histograms_comparison(results_list, save_path=None):
    """
    Compara histogramas para diferentes valores de p.
    Incluye dimensión fractal.
    
    Args:
        results_list: lista de diccionarios de resultados para diferentes p
    """
    n_plots = len(results_list)
    fig, axes = plt.subplots(3, n_plots, figsize=(5*n_plots, 12))
    
    if n_plots == 1:
        axes = axes.reshape(3, 1)
    
    colors = ['steelblue', 'forestgreen', 'coral', 'purple', 'orange']
    
    for idx, result in enumerate(results_list):
        p = result['p']
        color = colors[idx % len(colors)]
        
        # Histograma 1: Fracción de árboles quemados
        ax = axes[0, idx]
        ax.hist(result['fractions_trees'], bins=50, alpha=0.7, color=color, edgecolor='black')
        mean_val = np.mean(result['fractions_trees'])
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Media: {mean_val:.3f}')
        ax.set_xlabel('Fracción de árboles quemados', fontsize=11)
        ax.set_ylabel('Frecuencia', fontsize=11)
        ax.set_title(f'p = {p:.2f}\n(Fracción de árboles)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Histograma 2: Fracción de superficie quemada
        ax = axes[1, idx]
        ax.hist(result['fractions_surface'], bins=50, alpha=0.7, color=color, edgecolor='black')
        mean_val = np.mean(result['fractions_surface'])
        ax.axvline(mean_val, color='red', linestyle='--', linewidth=2, label=f'Media: {mean_val:.3f}')
        ax.set_xlabel('Fracción de superficie quemada', fontsize=11)
        ax.set_ylabel('Frecuencia', fontsize=11)
        ax.set_title(f'p = {p:.2f}\n(Fracción de superficie)', fontsize=12, fontweight='bold')
        ax.legend()
        ax.grid(alpha=0.3)
        
        # Histograma 3: Dimensión fractal
        ax = axes[2, idx]
        if len(result['fractal_dimensions']) > 0:
            ax.hist(result['fractal_dimensions'], bins=30, alpha=0.7, color='darkviolet', edgecolor='black')
            mean_D_f = np.mean(result['fractal_dimensions'])
            std_D_f = np.std(result['fractal_dimensions'])
            ax.axvline(mean_D_f, color='red', linestyle='--', linewidth=2, 
                      label=f'$D_f$ = {mean_D_f:.3f} ± {std_D_f:.3f}')
            ax.set_xlabel('Dimensión Fractal $D_f$', fontsize=11)
            ax.set_ylabel('Frecuencia', fontsize=11)
            ax.set_title(f'p = {p:.2f}\n(Dimensión Fractal)', fontsize=12, fontweight='bold')
            ax.legend()
            ax.grid(alpha=0.3)
            # Línea de referencia para d=2 (superficie completa)
            ax.axvline(2.0, color='blue', linestyle=':', linewidth=1.5, alpha=0.5)
        else:
            ax.text(0.5, 0.5, 'Insuficientes datos\nfractales', 
                   ha='center', va='center', fontsize=11, transform=ax.transAxes)
            ax.set_title(f'p = {p:.2f}\n(Sin datos fractales)', fontsize=12)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figura guardada: {save_path}")
    plt.show()


def analyze_percolation_threshold(L=100, p_values=None, n_simulations=500):
    """
    Analiza el umbral de percolación variando p.
    """
    if p_values is None:
        p_values = np.linspace(0.1, 0.9, 17)
    
    mean_fractions = []
    std_fractions = []
    
    print("\n" + "="*60)
    print("ANÁLISIS DE UMBRAL DE PERCOLACIÓN")
    print("="*60)
    
    for p in p_values:
        result = simulate_multiple_fires(L, p, n_simulations, verbose=False)
        mean_fractions.append(np.mean(result['fractions_trees']))
        std_fractions.append(np.std(result['fractions_trees']))
        print(f"p={p:.2f} → Fracción media quemada: {mean_fractions[-1]:.3f} ± {std_fractions[-1]:.3f}")
    
    # Graficar transición de fase
    plt.figure(figsize=(10, 6))
    plt.errorbar(p_values, mean_fractions, yerr=std_fractions, 
                 marker='o', capsize=5, capthick=2, linewidth=2, markersize=8,
                 color='darkred', ecolor='gray', label='Media ± Desv. Est.')
    plt.axhline(0.5, color='blue', linestyle=':', linewidth=2, alpha=0.5, label='Umbral teórico (≈0.5)')
    plt.xlabel('Densidad de árboles (p)', fontsize=13)
    plt.ylabel('Fracción media de árboles quemados', fontsize=13)
    plt.title('Transición de Fase en Percolación de Incendios\n(Umbral crítico cerca de p ≈ 0.59)', 
              fontsize=14, fontweight='bold')
    plt.grid(alpha=0.3)
    plt.legend(fontsize=11)
    plt.tight_layout()
    plt.show()
    
    return p_values, mean_fractions, std_fractions


def analyze_fractal_vs_density(results_list, save_path=None):
    """
    Analiza cómo varía la dimensión fractal con la densidad p.
    Conexión directa con la teoría de DLA y percolación.
    """
    p_vals = []
    mean_D_f = []
    std_D_f = []
    
    print("\n" + "="*60)
    print("ANÁLISIS: DIMENSIÓN FRACTAL vs DENSIDAD")
    print("="*60)
    
    for result in results_list:
        if len(result['fractal_dimensions']) > 5:
            p_vals.append(result['p'])
            mean_D_f.append(np.mean(result['fractal_dimensions']))
            std_D_f.append(np.std(result['fractal_dimensions']))
            print(f"p={result['p']:.2f} → D_f = {mean_D_f[-1]:.3f} ± {std_D_f[-1]:.3f} ({len(result['fractal_dimensions'])} muestras)")
    
    if len(p_vals) == 0:
        print("No hay suficientes datos fractales para analizar")
        return
    
    # Graficar
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.errorbar(p_vals, mean_D_f, yerr=std_D_f, 
                marker='o', capsize=8, capthick=2.5, linewidth=2.5, markersize=10,
                color='darkgreen', ecolor='gray', label='$D_f$ promedio')
    
    # Líneas de referencia
    ax.axhline(2.0, color='blue', linestyle='--', linewidth=2, alpha=0.6, 
              label='$D_f$ = 2 (superficie euclidiana)')
    ax.axhline(1.71, color='red', linestyle=':', linewidth=2, alpha=0.6, 
              label='$D_f$ ≈ 1.71 (DLA 2D teórico)')
    
    ax.set_xlabel('Densidad de árboles (p)', fontsize=13)
    ax.set_ylabel('Dimensión Fractal ($D_f$)', fontsize=13)
    ax.set_title('Evolución de la Dimensión Fractal con la Densidad\n(Conexión Percolación-DLA)', 
                fontsize=14, fontweight='bold')
    ax.set_ylim([1.3, 2.1])
    ax.grid(alpha=0.4, linestyle='--')
    ax.legend(fontsize=11, loc='best')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Figura guardada: {save_path}")
    plt.show()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == '__main__':
    print("\n" + "="*60)
    print("SIMULACIÓN DE PROPAGACIÓN DE INCENDIO FORESTAL")
    print("Proyecto basado en percolación y sistemas fractales")
    print("Incluye: Análisis de Dimensión Fractal (DLA)")
    print("="*60)
    
    # Parámetros
    L = 2000  # Tamaño de grilla
    n_simulations = 1000  # Número de simulaciones por cada p
    p_values = [0.2, 0.5, 0.6, 0.8]  # Densidades a analizar
    
    # 1. Mostrar ejemplo visual de un incendio con análisis fractal
    print("\n[1] Generando ejemplo visual con análisis fractal...")
    example = simulate_single_fire(L=L, p=0.60)
    plot_fire_example(example, show_fractal=True)
    
    # Si el ejemplo tiene datos fractales, mostrar análisis detallado
    if example['fractal_data'] is not None:
        print("\n[1.5] Análisis fractal detallado del ejemplo...")
        plot_fractal_analysis(example['burned_grid'], example['fractal_data'], result=example)
    
    # 2. Realizar simulaciones múltiples para cada p
    print("\n[2] Realizando simulaciones múltiples...")
    results_list = []
    for p in p_values:
        result = simulate_multiple_fires(L, p, n_simulations, verbose=True)
        results_list.append(result)
    
    # 3. Mostrar histogramas comparativos (ahora con dimensión fractal)
    print("\n[3] Generando histogramas comparativos...")
    plot_histograms_comparison(results_list)
    
    # 4. Análisis de dimensión fractal vs densidad
    print("\n[4] Analizando dimensión fractal vs densidad...")
    analyze_fractal_vs_density(results_list)
    
    # 5. Análisis de umbral de percolación
    print("\n[5] Analizando umbral de percolación...")
    analyze_percolation_threshold(L=100, n_simulations=500)
    
    # Estadísticas finales
    print("\n" + "="*60)
    print("RESUMEN DE RESULTADOS")
    print("="*60)
    for result in results_list:
        p = result['p']
        mean_trees = np.mean(result['fractions_trees'])
        std_trees = np.std(result['fractions_trees'])
        mean_surf = np.mean(result['fractions_surface'])
        
        print(f"\np = {p:.2f}:")
        print(f"  • Fracción de árboles quemados: {mean_trees:.3f} ± {std_trees:.3f}")
        print(f"  • Fracción de superficie quemada: {mean_surf:.3f}")
        print(f"  • Probabilidad de incendio masivo (>50% árboles): {np.mean(result['fractions_trees'] > 0.5):.1%}")
        
        if len(result['fractal_dimensions']) > 0:
            mean_D_f = np.mean(result['fractal_dimensions'])
            std_D_f = np.std(result['fractal_dimensions'])
            print(f"  • Dimensión fractal: D_f = {mean_D_f:.3f} ± {std_D_f:.3f} ({len(result['fractal_dimensions'])} muestras válidas)")
    
    print("\n" + "="*60)
    print("INTERPRETACIÓN FÍSICA")
    print("="*60)
    print("• D_f ≈ 2.0 → Incendio compacto (superficie euclidiana)")
    print("• D_f < 2.0 → Incendio ramificado/fractal (similar a DLA)")
    print("• D_f teórico DLA 2D ≈ 1.71")
    print("• A menor p, más fractal (estructura ramificada)")
    print("• A mayor p, más compacto (se acerca a D_f = 2)")
    print("="*60)
    
    print("\n✓ Simulación completada exitosamente!")
