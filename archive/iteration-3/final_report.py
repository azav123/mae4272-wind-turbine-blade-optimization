"""
Final Report Generator for Windfoil Optimization
==============================================

Generates comprehensive final analysis for the optimized smoothed blade:
1. Torque vs RPM analysis (0-2000 RPM and 0-3000 RPM ranges)
2. Power vs RPM analysis (0-2000 RPM and 0-3000 RPM ranges)
3. Bending stress vs RPM analysis (0-2000 RPM with failure line)
4. Text summary with optimal values

Reads smoothed_blade_parameters.csv from viterna_corrigan output and
saves results to output2/ folder.

Run after completing enhanced_analysis.py.
"""

import numpy as np
import math
from scipy import interpolate
from scipy.integrate import trapezoid
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for batch processing
import matplotlib.pyplot as plt
import os

# Output directory
OUTPUT2_DIR = 'output2'

# Create output directory
os.makedirs(OUTPUT2_DIR, exist_ok=True)

# Load smoothed blade parameters
print("Loading smoothed blade parameters...")
df_smooth = pd.read_csv('output/viterna_corrigan/smoothed_blade_parameters.csv')
r_list = df_smooth['radial_position_mm'].values / 1000  # Convert to meters
pitch_angles = df_smooth['pitch_angle_smooth_deg'].values  # Use smoothed values
chord_list = df_smooth['chord_mm'].values / 1000  # Convert to meters

print(f"Blade parameters loaded: {len(r_list)} radial points")
print(f"Radial range: {r_list[0]*1000:.1f} to {r_list[-1]*1000:.1f} mm")
print(f"Chord range: {chord_list[0]*1000:.1f} to {chord_list[-1]*1000:.1f} mm")

# Constants (match windfoil_optimization.py)
RHO = 1.225  # kg/m^3, air density
FAILURE_STRESS = 44e6  # Pa, ultimate stress
SAFETY_FACTOR = 1.5
MAX_STRESS = FAILURE_STRESS / SAFETY_FACTOR  # 29.33 MPa
WEIBULL_K = 5.0
WEIBULL_C = 5.0
N_RADIAL_POINTS = 10
N_SEGMENTS = N_RADIAL_POINTS - 1
B = 1  # Blades per analysis

# Calculate mean wind speed
mean_wind = WEIBULL_C * math.gamma(1 + 1/WEIBULL_K)
print(f"Analysis at mean wind speed: {mean_wind:.2f} m/s")

# Load polar data
print("Loading NACA 4412 polar data...")
df_polar = pd.read_csv('airfoil data/xf-naca4412-il-50000.csv', skiprows=10)
alpha_list = df_polar['Alpha'].values
cl_list = df_polar['Cl'].values
cd_list = df_polar['Cd'].values
cl_interp = interpolate.interp1d(alpha_list, cl_list, kind='linear', fill_value='extrapolate')
cd_interp = interpolate.interp1d(alpha_list, cd_list, kind='linear', fill_value='extrapolate')

# Calculate moments of inertia
print("Calculating airfoil moments of inertia...")
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

# Physics functions
def calculate_forces_and_torque(rpm, U):
    omega = rpm * 2 * np.pi / 60
    dFn_seg = np.zeros(N_SEGMENTS)
    r_mid = np.zeros(N_SEGMENTS)
    Q = 0.0

    for seg in range(N_SEGMENTS):
        r1, r2 = r_list[seg], r_list[seg + 1]
        dr = r2 - r1
        r_mid[seg] = (r1 + r2) / 2
        theta_mid = (pitch_angles[seg] + pitch_angles[seg + 1]) / 2
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
        if I_k > 0:  # Avoid division by zero
            sigma[k] = abs(M[k] * c_k / I_k)
        else:
            sigma[k] = 0.0

    return np.max(sigma)

print("\n" + "="*80)
print("FINAL PERFORMANCE ANALYSIS - SMOOTHED BLADE")
print("="*80)

# Analysis ranges
RPM_COUNT = 300
rpm_range_2000 = np.linspace(0, 2000, RPM_COUNT)
rpm_range_3000 = np.linspace(0, 3000, RPM_COUNT)

print(f"Analyzing {RPM_COUNT} RPM points from 0 to 2000 RPM and 0 to 3000 RPM...")

# Calculate performance arrays
torque_2000 = []
power_2000 = []
stress_2000 = []

torque_3000 = []
power_3000 = []
stress_3000 = []

for rpm in rpm_range_2000:
    Q, dFn_seg, r_mid = calculate_forces_and_torque(rpm, mean_wind)
    omega = rpm * 2 * np.pi / 60
    P = Q * omega
    max_sigma = calculate_max_stress(dFn_seg, r_mid)

    torque_2000.append(Q)
    power_2000.append(P)
    stress_2000.append(max_sigma / 1e6)  # Convert to MPa

for rpm in rpm_range_3000:
    Q, dFn_seg, r_mid = calculate_forces_and_torque(rpm, mean_wind)
    omega = rpm * 2 * np.pi / 60
    P = Q * omega
    max_sigma = calculate_max_stress(dFn_seg, r_mid)

    torque_3000.append(Q)
    power_3000.append(P)
    stress_3000.append(max_sigma / 1e6)  # Convert to MPa

# Find optimal values (maximum power point)
max_power_idx = np.argmax(power_2000)
optimal_rpm = rpm_range_2000[max_power_idx]
optimal_torque = torque_2000[max_power_idx]
optimal_power = power_2000[max_power_idx]
optimal_max_stress = stress_2000[max_power_idx]

print(f"\nOptimal Operating Point:")
print(f"  RPM: {optimal_rpm:.1f}")
print(f"  Torque: {optimal_torque:.6f} N·m")
print(f"  Power: {optimal_power:.3f} W")
print(f"  Max Bending Stress: {optimal_max_stress:.3f} MPa")

# ============================================================================
# 1. TORQUE vs RPM (0-2000 RPM)
# ============================================================================

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(rpm_range_2000, torque_2000, 'b-', linewidth=2)
ax.axvline(optimal_rpm, color='r', linestyle='--', linewidth=2,
           label=f'Optimal RPM ({optimal_rpm:.0f})')
ax.axhline(optimal_torque, color='g', linestyle='--', linewidth=2,
           label=f'Optimal Torque ({optimal_torque:.4f} N·m)')
ax.set_xlabel('RPM')
ax.set_ylabel('Torque (N·m)')
ax.set_title('Torque vs RPM (0-2000 RPM Range)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT2_DIR, 'torque_vs_rpm_0_2000.png'), dpi=300)
print(f"\nSaved: {os.path.join(OUTPUT2_DIR, 'torque_vs_rpm_0_2000.png')}")

# ============================================================================
# 2. TORQUE vs RPM (0-3000 RPM)
# ============================================================================

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(rpm_range_3000, torque_3000, 'b-', linewidth=2)
ax.axvline(optimal_rpm, color='r', linestyle='--', linewidth=2,
           label=f'Optimal RPM ({optimal_rpm:.0f})')
ax.axhline(optimal_torque, color='g', linestyle='--', linewidth=2,
           label=f'Optimal Torque ({optimal_torque:.4f} N·m)')
ax.set_xlabel('RPM')
ax.set_ylabel('Torque (N·m)')
ax.set_title('Torque vs RPM (0-3000 RPM Range)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT2_DIR, 'torque_vs_rpm_0_3000.png'), dpi=300)
print(f"Saved: {os.path.join(OUTPUT2_DIR, 'torque_vs_rpm_0_3000.png')}")

# ============================================================================
# 3. POWER vs RPM (0-2000 RPM)
# ============================================================================

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(rpm_range_2000, power_2000, 'g-', linewidth=2)
ax.axvline(optimal_rpm, color='r', linestyle='--', linewidth=2,
           label=f'Optimal RPM ({optimal_rpm:.0f})')
ax.axhline(optimal_power, color='purple', linestyle='--', linewidth=2,
           label=f'Max Power ({optimal_power:.3f} W)')
ax.set_xlabel('RPM')
ax.set_ylabel('Power Output (W)')
ax.set_title('Power vs RPM (0-2000 RPM Range)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT2_DIR, 'power_vs_rpm_0_2000.png'), dpi=300)
print(f"Saved: {os.path.join(OUTPUT2_DIR, 'power_vs_rpm_0_2000.png')}")

# ============================================================================
# 4. POWER vs RPM (0-3000 RPM)
# ============================================================================

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(rpm_range_3000, power_3000, 'g-', linewidth=2)
ax.axvline(optimal_rpm, color='r', linestyle='--', linewidth=2,
           label=f'Optimal RPM ({optimal_rpm:.0f})')
ax.axhline(optimal_power, color='purple', linestyle='--', linewidth=2,
           label=f'Max Power ({optimal_power:.3f} W)')
ax.set_xlabel('RPM')
ax.set_ylabel('Power Output (W)')
ax.set_title('Power vs RPM (0-3000 RPM Range)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT2_DIR, 'power_vs_rpm_0_3000.png'), dpi=300)
print(f"Saved: {os.path.join(OUTPUT2_DIR, 'power_vs_rpm_0_3000.png')}")

# ============================================================================
# 5. BENDING STRESS vs RPM (0-2000 RPM)
# ============================================================================

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(rpm_range_2000, stress_2000, 'm-', linewidth=2)
ax.axvline(optimal_rpm, color='r', linestyle='--', linewidth=2,
           label=f'Optimal RPM ({optimal_rpm:.0f})')
ax.axhline(MAX_STRESS / 1e6, color='r', linestyle='-', linewidth=2,
           label=f'Failure Stress ({MAX_STRESS/1e6:.1f} MPa)')
ax.set_xlabel('RPM')
ax.set_ylabel('Maximum Bending Stress (MPa)')
ax.set_title('Bending Stress vs RPM (0-2000 RPM Range)')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(OUTPUT2_DIR, 'bending_stress_vs_rpm_0_2000.png'), dpi=300)
print(f"Saved: {os.path.join(OUTPUT2_DIR, 'bending_stress_vs_rpm_0_2000.png')}")

# ============================================================================
# 6. SAVE SMOOTHED BLADE PARAMETERS TO output2/
# ============================================================================

df_smooth.to_csv(os.path.join(OUTPUT2_DIR, 'smoothed_blade_parameters_final.csv'), index=False)
print(f"Saved smoothed blade parameters: {os.path.join(OUTPUT2_DIR, 'smoothed_blade_parameters_final.csv')}")

# ============================================================================
# 7. SAVE FINAL REPORT TXT SUMMARY
# ============================================================================

print(f"\nGenerating final report summary...")
with open(os.path.join(OUTPUT2_DIR, 'final_report_summary.txt'), 'w') as f:
    f.write("="*80 + "\n")
    f.write("WINDFOIL OPTIMIZATION FINAL REPORT\n")
    f.write("="*80 + "\n\n")

    f.write("ANALYSIS CONDITIONS:\n")
    f.write("-" * 30 + "\n")
    f.write(f"Mean wind speed: {mean_wind:.2f} m/s\n")
    f.write(f"Airfoil: NACA 4412\n")
    f.write(f"Number of radial points: {N_RADIAL_POINTS}\n")
    f.write(f"Analysis RPM range: 0-2000 RPM\n")
    f.write(f"Material failure stress: {FAILURE_STRESS/1e6:.1f} MPa (ultimate)\n")
    f.write(f"Safety factor: {SAFETY_FACTOR})\n")
    f.write(f"Allowable stress: {MAX_STRESS/1e6:.1f} MPa (failure stress/safety factor)\n\n")

    f.write("OPTIMAL OPERATING POINT:\n")
    f.write("-" * 30 + "\n")
    f.write(f"Optimal RPM: {optimal_rpm:.1f} RPM\n")
    f.write(f"Ideal torque: {optimal_torque:.6f} N·m\n")
    f.write(f"Ideal power: {optimal_power:.3f} W\n")
    f.write(f"Max bending stress: {optimal_max_stress:.3f} MPa\n\n")

    f.write("BLADE GEOMETRY (smoothed parameters):\n")
    f.write("-" * 30 + "\n")
    f.write(f"Radial positions: {r_list[0]*1000:.1f} to {r_list[-1]*1000:.1f} mm\n")
    f.write(f"Chord lengths: {chord_list[0]*1000:.1f} to {chord_list[-1]*1000:.1f} mm\n")
    f.write(f"Pitch angles: {pitch_angles[0]:.1f}° to {pitch_angles[-1]:.1f}°\n\n")

    f.write("FILES GENERATED:\n")
    f.write("-" * 30 + "\n")
    f.write("torque_vs_rpm_0_2000.png\n")
    f.write("torque_vs_rpm_0_3000.png\n")
    f.write("power_vs_rpm_0_2000.png\n")
    f.write("power_vs_rpm_0_3000.png\n")
    f.write("bending_stress_vs_rpm_0_2000.png\n")
    f.write("smoothed_blade_parameters_final.csv\n")
    f.write("final_report_summary.txt\n")

print(f"Saved final report summary: {os.path.join(OUTPUT2_DIR, 'final_report_summary.txt')}")

print(f"\n{'='*80}")
print("FINAL REPORT GENERATION COMPLETE!")
print(f"{'='*80}")
print(f"\nAll files saved to: {OUTPUT2_DIR}/")
print("Generated:")
print("  ✓ torque_vs_rpm_0_2000.png")
print("  ✓ torque_vs_rpm_0_3000.png")
print("  ✓ power_vs_rpm_0_2000.png")
print("  ✓ power_vs_rpm_0_3000.png")
print("  ✓ bending_stress_vs_rpm_0_2000.png")
print("  ✓ smoothed_blade_parameters_final.csv")
print("  ✓ final_report_summary.txt")

print(f"  Optimal RPM: {optimal_rpm:.1f}")
print(f"  Ideal torque: {optimal_torque:.6f} N·m")
print(f"  Ideal power: {optimal_power:.3f} W")
print(f"  Max bending stress: {optimal_max_stress:.3f} MPa")
