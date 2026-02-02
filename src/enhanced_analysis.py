"""
Enhanced Analysis for Windfoil Optimization
============================================

This script provides additional analyses:
1. Smooth pitch angle distribution using polynomial fitting
2. Power vs RPM analysis at mean wind speed  
3. Torque brake specifications

Run this after completing the main optimization.
Can specify output directory with --dir argument.
"""

import numpy as np
import math
from scipy import interpolate
from scipy.integrate import trapezoid
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for batch processing
import matplotlib.pyplot as plt
from pathlib import Path
BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "outputs" / "viterna_corrigan"

import argparse
parser = argparse.ArgumentParser(description='Enhanced analysis for windfoil optimization')
parser.add_argument('--dir', type=str, default=str(RESULTS_DIR),
                    help='Directory containing optimization results')
args = parser.parse_args()
OUTPUT_DIR = Path(args.dir)
# Load optimization results
print("Loading optimization results...")
df_results = pd.read_csv(str(OUTPUT_DIR / 'optimal_blade_parameters.csv'))
r_list = df_results['radial_position_mm'].values / 1000  # Convert to meters
pitch_angles = df_results['pitch_angle_deg'].values
chord_list = df_results['chord_mm'].values / 1000  # Convert to meters

# Constants (must match windfoil_optimization.py)
RHO = 1.225
FAILURE_STRESS = 55e6
SAFETY_FACTOR = 1.5
MAX_STRESS = FAILURE_STRESS / SAFETY_FACTOR
HUB_RADIUS = 0.0254
MAX_RADIUS = 0.0254 + 0.1524
MAX_RPM = 2000
WEIBULL_K = 5.0
WEIBULL_C = 5.0
N_RADIAL_POINTS = 10
N_SEGMENTS = N_RADIAL_POINTS - 1
B = 1

# Load polar data
print("Loading NACA 4412 polar data...")
df_polar = pd.read_csv(str(BASE_DIR / 'data' / 'xf-naca4412-il-50000.csv'), skiprows=10)
alpha_list = df_polar['Alpha'].values
cl_list = df_polar['Cl'].values
cd_list = df_polar['Cd'].values
cl_interp = interpolate.interp1d(alpha_list, cl_list, kind='linear', fill_value='extrapolate')
cd_interp = interpolate.interp1d(alpha_list, cd_list, kind='linear', fill_value='extrapolate')

# Calculate I
def calculate_naca4412_I():
    m, p, t = 0.04, 0.4, 0.12
    n_points = 1000
    x = np.linspace(0, 1, n_points)
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 0.2843 * x**3 - 0.1015 * x**4)
    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)
    mask1 = x < p
    mask2 = x >= p
    yc[mask1] = m / p**2 * (2 * p * x[mask1] - x[mask1]**2)
    dyc_dx[mask1] = m / p**2 * (2 * p - 2 * x[mask1])
    yc[mask2] = m / (1 - p)**2 * (1 - 2 * p + 2 * p * x[mask2] - x[mask2]**2)
    dyc_dx[mask2] = m / (1 - p)**2 * (2 * p - 2 * x[mask2])
    theta = np.arctan(dyc_dx)
    yu = yc + yt * np.cos(theta)
    yl = yc - yt * np.cos(theta)
    A = trapezoid(yu - yl, x)
    int_ybar = 0.5 * trapezoid(yu**2 - yl**2, x)
    y_bar = int_ybar / A
    int1 = (1/3) * trapezoid(yu**3 - yl**3, x)
    int2 = (1/2) * trapezoid(yu**2 - yl**2, x)
    int3 = trapezoid(yu - yl, x)
    return int1 - y_bar * int2 + y_bar**2 * int3

I_non_dim = calculate_naca4412_I()

# Helper functions
def calculate_forces_and_torque(theta_list, rpm, U):
    omega = rpm * 2 * np.pi / 60
    dFn_seg = np.zeros(N_SEGMENTS)
    r_mid = np.zeros(N_SEGMENTS)
    Q = 0.0
    for seg in range(N_SEGMENTS):
        r1, r2 = r_list[seg], r_list[seg + 1]
        dr = r2 - r1
        r_mid[seg] = (r1 + r2) / 2
        theta_mid = (theta_list[seg] + theta_list[seg + 1]) / 2
        c_mid = (chord_list[seg] + chord_list[seg + 1]) / 2
        wr = omega * r_mid[seg]
        W = np.sqrt(U**2 + wr**2)
        beta_deg = np.degrees(np.arctan(wr / U))
        phi_deg = 90 - beta_deg
        alpha_deg = phi_deg - theta_mid
        cl = cl_interp(alpha_deg)
        cd = cd_interp(alpha_deg)
        beta_rad = np.deg2rad(beta_deg)
        phi_rad = np.deg2rad(phi_deg)
        dFn = 0.5 * RHO * W**2 * c_mid * dr * (cl * np.sin(beta_rad) + cd * np.sin(phi_rad))
        dFt = 0.5 * RHO * W**2 * c_mid * dr * (cl * np.cos(beta_rad) - cd * np.cos(phi_rad))
        dFn_seg[seg] = dFn
        Q += r_mid[seg] * dFt
    Q *= B
    return Q, dFn_seg, r_mid

def calculate_max_stress(dFn_seg, r_mid):
    M = np.zeros(N_RADIAL_POINTS)
    for k in range(N_RADIAL_POINTS):
        for seg in range(k, N_SEGMENTS):
            if r_mid[seg] > r_list[k]:
                M[k] += dFn_seg[seg] * (r_mid[seg] - r_list[k])
    sigma = np.zeros(N_RADIAL_POINTS)
    for k in range(N_RADIAL_POINTS):
        chord_k = chord_list[k]
        h_k = 0.12 * chord_k
        I_k = I_non_dim * chord_k**4
        c_k = h_k / 2
        sigma[k] = abs(M[k] * c_k / I_k)
    return np.max(sigma)

# Read optimal RPM from CSV metadata or use default
try:
    with open(str(OUTPUT_DIR / 'optimal_blade_parameters.csv'), 'r') as f:
        first_line = f.readline()
        if 'RPM' in first_line:
            rpm_optimal = float(first_line.split(':')[1].strip())
        else:
            rpm_optimal = 1500  # Default if not found
except:
    rpm_optimal = 1500

print(f"\nUsing optimal RPM: {rpm_optimal:.1f}")

# ============================================================================
# 1. SMOOTH PITCH ANGLE DISTRIBUTION
# ============================================================================
print("\n" + "="*70)
print("SMOOTH PITCH ANGLE DISTRIBUTION")
print("="*70)

poly_degree = 2
coeffs = np.polyfit(r_list, pitch_angles, poly_degree)
poly_func = np.poly1d(coeffs)
pitch_smooth = poly_func(r_list)

# Ensure monotonically decreasing
for i in range(1, len(pitch_smooth)):
    if pitch_smooth[i] > pitch_smooth[i-1]:
        pitch_smooth[i] = pitch_smooth[i-1]

print("\nSmoothed Pitch Angle Function theta(r):")
print("theta(r) = ", end="")
for i, coeff in enumerate(coeffs):
    power = poly_degree - i
    if i > 0:
        print(" + " if coeff >= 0 else " - ", end="")
        print(f"{abs(coeff):.6f}·r^{power}", end="")
    else:
        print(f"{coeff:.6f}·r^{power}", end="")
print(" (degrees, r in meters)")

# Visualize
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
ax1.plot(r_list * 1000, pitch_angles, 'bo-', label='Optimized (Raw)', markersize=6)
ax1.plot(r_list * 1000, pitch_smooth, 'r-', label=f'Smoothed (Poly deg {poly_degree})', linewidth=2)
ax1.set_xlabel('Radial Position (mm)')
ax1.set_ylabel('Pitch Angle (degrees)')
ax1.set_title('Pitch Angle Distribution: Raw vs Smoothed')
ax1.legend()
ax1.grid(True, alpha=0.3)

diff = pitch_angles - pitch_smooth
ax2.plot(r_list * 1000, diff, 'g-o', markersize=6)
ax2.axhline(0, color='k', linestyle='--', linewidth=0.5)
ax2.set_xlabel('Radial Position (mm)')
ax2.set_ylabel('Difference (degrees)')
ax2.set_title('Raw - Smoothed Pitch Angle')
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / 'smooth_pitch_analysis.png'), dpi=300)
print(f"\nPlot saved to '{str(OUTPUT_DIR / 'smooth_pitch_analysis.png')}'")

# Save smoothed pitch to CSV
df_smooth = pd.DataFrame({
    'radial_position_mm': r_list * 1000,
    'pitch_angle_raw_deg': pitch_angles,
    'pitch_angle_smooth_deg': pitch_smooth,
    'chord_mm': chord_list * 1000
})
df_smooth.to_csv(str(OUTPUT_DIR / 'smoothed_blade_parameters.csv'), index=False)
print(f"Smoothed parameters saved to '{str(OUTPUT_DIR / 'smoothed_blade_parameters.csv')}'")

# ============================================================================
# 2. POWER VS RPM AT MEAN WIND SPEED
# ============================================================================
print("\n" + "="*70)
print("POWER VS RPM ANALYSIS")
print("="*70)

mean_wind = WEIBULL_C * math.gamma(1 + 1/WEIBULL_K)
print(f"\nMean wind speed: {mean_wind:.2f} m/s")

rpm_range = np.linspace(100, MAX_RPM, 100)
power_range = []
torque_range = []
stress_range = []

for rpm in rpm_range:
    Q, dFn_seg, r_mid = calculate_forces_and_torque(pitch_angles, rpm, mean_wind)
    max_sigma = calculate_max_stress(dFn_seg, r_mid)
    omega = rpm * 2 * np.pi / 60
    P = Q * omega if max_sigma <= MAX_STRESS else 0.0
    power_range.append(P)
    torque_range.append(Q)
    stress_range.append(max_sigma / 1e6)

# Find peak power
max_power_idx = np.argmax(power_range)
peak_rpm = rpm_range[max_power_idx]
peak_power = power_range[max_power_idx]
peak_torque = torque_range[max_power_idx]

print(f"\nAt mean wind speed ({mean_wind:.2f} m/s):")
print(f"  Peak power: {peak_power:.3f} W at {peak_rpm:.1f} RPM")
print(f"  Torque at peak: {peak_torque:.6f} N·m")
print(f"  Optimal operating RPM: {rpm_optimal:.1f}")

# Plot Power vs RPM
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(rpm_range, power_range, 'b-', linewidth=2)
ax1.axvline(rpm_optimal, color='r', linestyle='--', label=f'Optimal RPM ({rpm_optimal:.0f})')
ax1.axvline(peak_rpm, color='g', linestyle='--', label=f'Peak Power RPM ({peak_rpm:.0f})')
ax1.set_xlabel('RPM')
ax1.set_ylabel('Power Output (W)')
ax1.set_title(f'Power vs RPM at Mean Wind Speed ({mean_wind:.2f} m/s)')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(rpm_range, torque_range, 'm-', linewidth=2)
ax2.axvline(rpm_optimal, color='r', linestyle='--', label=f'Optimal RPM')
ax2.set_xlabel('RPM')
ax2.set_ylabel('Torque (N·m)')
ax2.set_title('Torque vs RPM')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / 'power_vs_rpm_analysis.png'), dpi=300)
print(f"\nPlot saved to '{str(OUTPUT_DIR / 'power_vs_rpm_analysis.png')}'")

# ============================================================================
# 3. SMOOTHED BLADE POWER/TORQUE vs RPM ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("SMOOTHED BLADE POWER/TORQUE vs RPM ANALYSIS")
print("="*70)

print("Using smoothed pitch angles for analysis...")

rpm_range_smooth = np.linspace(100, MAX_RPM, 100)
power_range_smooth = []
torque_range_smooth = []
stress_range_smooth = []

for rpm in rpm_range_smooth:
    Q, dFn_seg, r_mid = calculate_forces_and_torque(pitch_smooth, rpm, mean_wind)
    max_sigma = calculate_max_stress(dFn_seg, r_mid)
    omega = rpm * 2 * np.pi / 60
    P = Q * omega if max_sigma <= MAX_STRESS else 0.0
    power_range_smooth.append(P)
    torque_range_smooth.append(Q)
    stress_range_smooth.append(max_sigma / 1e6)

# Find peak power for smoothed blade
max_power_idx_smooth = np.argmax(power_range_smooth)
peak_rpm_smooth = rpm_range_smooth[max_power_idx_smooth]
peak_power_smooth = power_range_smooth[max_power_idx_smooth]
peak_torque_smooth = torque_range_smooth[max_power_idx_smooth]

print(f"\nSmoothed blade at mean wind speed ({mean_wind:.2f} m/s):")
print(f"  Peak power: {peak_power_smooth:.3f} W at {peak_rpm_smooth:.1f} RPM")
print(f"  Torque at peak: {peak_torque_smooth:.6f} N·m")
print(f"  Optimal operating RPM: {rpm_optimal:.1f}")

# Plot Power vs RPM for smoothed blade
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(rpm_range_smooth, power_range_smooth, 'b-', linewidth=2)
ax1.axvline(rpm_optimal, color='r', linestyle='--', label=f'Optimal RPM ({rpm_optimal:.0f})')
ax1.axvline(peak_rpm_smooth, color='g', linestyle='--', label=f'Peak Power RPM ({peak_rpm_smooth:.0f})')
ax1.set_xlabel('RPM')
ax1.set_ylabel('Power Output (W)')
ax1.set_title(f'Smoothed Blade: Power vs RPM ({mean_wind:.2f} m/s)')
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(rpm_range_smooth, torque_range_smooth, 'm-', linewidth=2)
ax2.axvline(rpm_optimal, color='r', linestyle='--', label=f'Optimal RPM')
ax2.set_xlabel('RPM')
ax2.set_ylabel('Torque (N·m)')
ax2.set_title('Smoothed Blade: Torque vs RPM')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(str(OUTPUT_DIR / 'smoothed_power_vs_rpm_analysis.png'), dpi=300)
print(f"\nPlot saved to '{str(OUTPUT_DIR / 'smoothed_power_vs_rpm_analysis.png')}'")

# ============================================================================
# 4. TORQUE BRAKE SPECIFICATIONS
# ============================================================================
print("\n" + "="*70)
print("TORQUE BRAKE SPECIFICATIONS")
print("="*70)

# Calculate torque at various wind speeds
wind_speeds = np.linspace(3, 15, 13)
print(f"\n{'Wind Speed (m/s)':<20} {'Torque (N·m)':<20} {'Power (W)':<20}")
print("-" * 60)

for U in wind_speeds:
    Q, dFn_seg, r_mid = calculate_forces_and_torque(pitch_angles, rpm_optimal, U)
    max_sigma = calculate_max_stress(dFn_seg, r_mid)
    omega = rpm_optimal * 2 * np.pi / 60
    P = Q * omega if max_sigma <= MAX_STRESS else 0.0
    print(f"{U:<20.1f} {Q:<20.6f} {P:<20.3f}")

Q_mean, _, _ = calculate_forces_and_torque(pitch_angles, rpm_optimal, mean_wind)
omega_mean = rpm_optimal * 2 * np.pi / 60
P_mean = Q_mean * omega_mean

print(f"\n{'='*70}")
print("RECOMMENDED TORQUE BRAKE SETTING")
print(f"{'='*70}")
print(f"Operating RPM: {rpm_optimal:.1f}")
print(f"Torque brake: {Q_mean:.6f} N·m")
print(f"At mean wind speed ({mean_wind:.2f} m/s):")
print(f"  Expected power: {P_mean:.3f} W")

print("\n" + "="*70)
print("ENHANCED ANALYSIS COMPLETE")
print("="*70)
print(f"\nGenerated files in {OUTPUT_DIR}/:")
print("  - smooth_pitch_analysis.png")
print("  - smoothed_blade_parameters.csv")
print("  - power_vs_rpm_analysis.png")  # Raw optimized blade
print("  - smoothed_power_vs_rpm_analysis.png")  # Smoothed blade
