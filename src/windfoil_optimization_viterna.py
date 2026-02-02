"""
Windfoil Blade Optimization Script
MAE 4272 Design Project

Optimizes blade pitch/twist distribution and operating RPM to maximize
expected power output over Weibull wind distribution.

Constraints:
- Max blade length: 6 inches (0.1524 m)
- Hub radius: 1 inch (0.0254 m)
- Max RPM: 2000
- Max stress: 36.67 MPa (55 MPa / 1.5 safety factor)
"""

import numpy as np
from scipy import interpolate
from scipy.integrate import trapezoid, quad
from scipy.optimize import differential_evolution
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for batch processing
import matplotlib.pyplot as plt
from typing import Tuple, Dict
import time
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
RESULTS_DIR = BASE_DIR / "outputs" / "viterna_corrigan"


# ============================================================================
# CONSTANTS AND PARAMETERS
# ============================================================================

# Physical constants
RHO = 1.225  # kg/m^3, air density
FAILURE_STRESS = 55e6  # Pa
SAFETY_FACTOR = 1.5
MAX_STRESS = FAILURE_STRESS / SAFETY_FACTOR  # 29.33 MPa

# Blade geometry constraints
HUB_RADIUS = 0.0254  # m (1 inch)
MAX_BLADE_LENGTH = 0.1524  # m (6 inches)
MAX_RPM = 2000
MAX_TORQUE_PER_BLADE = 0.025 / 3  # N·m (2.5 N·cm total for 3 blades)

# Weibull distribution parameters
WEIBULL_K = 5.0
WEIBULL_C = 5.0  # m/s

# Discretization
N_RADIAL_POINTS = 10
N_SEGMENTS = N_RADIAL_POINTS - 1

# Number of blades (per blade analysis)
B = 1


# ============================================================================
# NACA 4412 MOMENT OF INERTIA CALCULATION
# ============================================================================

def calculate_naca4412_I():
    """
    Calculate non-dimensional moments of inertia and extreme fiber distances for NACA 4412.
    
    Returns:
    --------
    I_y_non_dim : float
        Non-dimensional second moment about y-axis (strong axis, perpendicular to chord)
    I_z_non_dim : float
        Non-dimensional second moment about z-axis (weak axis, parallel to chord)
    c_max_non_dim : float
        Non-dimensional maximum extreme fiber distance from centroid
    """
    m = 0.04
    p = 0.4
    t = 0.12
    n_points = 1000
    
    x = np.linspace(0, 1, n_points)
    yt = 5 * t * (0.2969 * np.sqrt(x) - 0.1260 * x - 0.3516 * x**2 + 
                   0.2843 * x**3 - 0.1015 * x**4)
    
    yc = np.zeros_like(x)
    dyc_dx = np.zeros_like(x)
    
    mask1 = x < p
    mask2 = x >= p
    
    yc[mask1] = m / p**2 * (2 * p * x[mask1] - x[mask1]**2)
    dyc_dx[mask1] = m / p**2 * (2 * p - 2 * x[mask1])
    
    yc[mask2] = m / (1 - p)**2 * (1 - 2 * p + 2 * p * x[mask2] - x[mask2]**2)
    dyc_dx[mask2] = m / (1 - p)**2 * (2 * p - 2 * x[mask2])
    
    theta = np.arctan(dyc_dx)
    xu = x - yt * np.sin(theta)
    yu = yc + yt * np.cos(theta)
    xl = x + yt * np.sin(theta)
    yl = yc - yt * np.cos(theta)
    
    # Calculate area and centroid
    A = trapezoid(yu - yl, x)
    int_ybar = 0.5 * trapezoid(yu**2 - yl**2, x)
    y_bar = int_ybar / A
    
    # I_y: Second moment about y-axis (perpendicular to chord) - strong axis
    int1 = (1/3) * trapezoid(yu**3 - yl**3, x)
    int2 = (1/2) * trapezoid(yu**2 - yl**2, x)
    int3 = trapezoid(yu - yl, x)
    I_y_non_dim = int1 - y_bar * int2 + y_bar**2 * int3
    
    # I_z: Second moment about z-axis (parallel to chord) - weak axis
    # I_z = ∫ x² dA about centroid at x = 0.5 (assuming chord from 0 to 1)
    x_bar = 0.5  # Centroid x-location for symmetric airfoil
    int_x2_1 = trapezoid(x**2 * (yu - yl), x)
    int_x_1 = trapezoid(x * (yu - yl), x)
    I_z_non_dim = int_x2_1 - 2 * x_bar * int_x_1 + x_bar**2 * A
    
    # Calculate extreme fiber distances from centroid
    y_max = np.max(yu)
    y_min = np.min(yl)
    c_top = y_max - y_bar
    c_bottom = y_bar - y_min
    c_max_non_dim = max(c_top, c_bottom)
    
    return I_y_non_dim, I_z_non_dim, c_max_non_dim


# ============================================================================
# LOAD POLAR DATA
# ============================================================================

def load_polar_data(filepath='airfoil data/xf-naca4412-il-50000.csv'):
    """
    Load NACA 4412 polar data with Viterna-Corrigan extrapolation.
    
    Uses measured data for -9.25° ≤ α ≤ 14°, and Viterna-Corrigan method
    for extrapolation beyond this range. This provides physically accurate
    behavior at high angles of attack where linear extrapolation fails.
    
    Viterna-Corrigan method:
    - For high α: Cl → 0 as α → 90° (flat plate behavior)
    - For low α: Simple flat plate theory
    """
    df_polar = pd.read_csv(filepath, skiprows=10)
    alpha_data = df_polar['Alpha'].values
    cl_data = df_polar['Cl'].values
    cd_data = df_polar['Cd'].values
    
    # Define extrapolation boundaries
    alpha_min = alpha_data.min()  # -9.25°
    alpha_max = alpha_data.max()  # 14.0°
    cl_stall = cl_data[-1]  # Cl at stall (14°)
    cd_stall = cd_data[-1]  # Cd at stall (14°)
    alpha_stall = alpha_max
    
    # Calculate Viterna-Corrigan coefficients
    # Based on matching flat plate theory at stall point
    AR = 1.0  # Aspect ratio for 2D airfoil
    cd_max = 1.11 + 0.018 * AR  # Empirical flat plate max Cd
    
    alpha_stall_rad = np.deg2rad(alpha_stall)
    sin_alpha = np.sin(alpha_stall_rad)
    cos_alpha = np.cos(alpha_stall_rad)
    sin_2alpha = np.sin(2 * alpha_stall_rad)
    
    # Coefficients for Cl extrapolation: Cl = A1*sin(2α) + A2*cos²(α)/sin(α)
    A1 = cd_max / 2
    A2 = (cl_stall - A1 * sin_2alpha) * sin_alpha / (cos_alpha**2)
    
    # Coefficients for Cd extrapolation: Cd = B1*sin²(α) + B2*cos(α)
    B1 = cd_max
    B2 = (cd_stall - B1 * sin_alpha**2) / cos_alpha
    
    # Viterna-Corrigan extrapolation functions
    def cl_viterna(alpha_deg):
        """Viterna-Corrigan Cl for high angles of attack."""
        alpha_rad = np.deg2rad(alpha_deg)
        sin_a = np.sin(alpha_rad)
        cos_a = np.cos(alpha_rad)
        # Avoid division by zero at α = 0°, 180°
        with np.errstate(divide='ignore', invalid='ignore'):
            result = A1 * np.sin(2 * alpha_rad) + A2 * (cos_a**2) / sin_a
            result = np.where(np.abs(sin_a) < 1e-10, 0.0, result)
        return result
    
    def cd_viterna(alpha_deg):
        """Viterna-Corrigan Cd for high angles of attack."""
        alpha_rad = np.deg2rad(alpha_deg)
        return B1 * np.sin(alpha_rad)**2 + B2 * np.cos(alpha_rad)
    
    # Create piecewise interpolation functions
    def cl_interp(alpha):
        """Piecewise Cl: measured data + Viterna-Corrigan extrapolation."""
        alpha = np.atleast_1d(alpha)
        result = np.zeros_like(alpha, dtype=float)
        
        # Use measured data within range
        mask_data = (alpha >= alpha_min) & (alpha <= alpha_max)
        if np.any(mask_data):
            interp_func = interpolate.interp1d(alpha_data, cl_data, 
                                               kind='linear', fill_value='extrapolate')
            result[mask_data] = interp_func(alpha[mask_data])
        
        # Viterna-Corrigan for high alpha (post-stall)
        mask_high = alpha > alpha_max
        if np.any(mask_high):
            result[mask_high] = cl_viterna(alpha[mask_high])
        
        # Flat plate theory for low alpha
        mask_low = alpha < alpha_min
        if np.any(mask_low):
            alpha_rad = np.deg2rad(alpha[mask_low])
            result[mask_low] = 2 * np.sin(alpha_rad) * np.cos(alpha_rad)  # sin(2α)
        
        return result.item() if result.size == 1 else result
    
    def cd_interp(alpha):
        """Piecewise Cd: measured data + Viterna-Corrigan extrapolation."""
        alpha = np.atleast_1d(alpha)
        result = np.zeros_like(alpha, dtype=float)
        
        # Use measured data within range
        mask_data = (alpha >= alpha_min) & (alpha <= alpha_max)
        if np.any(mask_data):
            interp_func = interpolate.interp1d(alpha_data, cd_data, 
                                               kind='linear', fill_value='extrapolate')
            result[mask_data] = interp_func(alpha[mask_data])
        
        # Viterna-Corrigan for high alpha
        mask_high = alpha > alpha_max
        if np.any(mask_high):
            result[mask_high] = cd_viterna(alpha[mask_high])
        
        # Flat plate drag for low alpha (increases with |α|)
        mask_low = alpha < alpha_min
        if np.any(mask_low):
            result[mask_low] = cd_data[0] + 0.01 * np.abs(alpha[mask_low] - alpha_min)
        
        return result.item() if result.size == 1 else result
    
    return cl_interp, cd_interp


# ============================================================================
# BLADE GEOMETRY PARAMETERIZATION
# ============================================================================

def create_blade_geometry(pitch_angles, blade_length=MAX_BLADE_LENGTH, c_root=0.0508, chord_distribution='polynomial'):
    """
    Create blade geometry arrays with optimizable parameters.
    
    Parameters:
    -----------
    pitch_angles : array-like, shape (N_RADIAL_POINTS,)
        Pitch angles in degrees at each radial station
    blade_length : float
        Total blade length (m), from hub to tip
    c_root : float
        Chord at hub (m)
    chord_distribution : str
        Method for chord distribution ('polynomial', 'linear', 'constant', etc.)
    
    Returns:
    --------
    r_list : ndarray
        Radial positions (m)
    theta_list : ndarray
        Pitch angles (degrees)
    chord_list : ndarray
        Chord lengths (m)
    """
    # Radial positions from hub to tip
    MAX_RADIUS_LOCAL = HUB_RADIUS + blade_length
    r_list = np.linspace(HUB_RADIUS, MAX_RADIUS_LOCAL, N_RADIAL_POINTS)
    
    # Pitch angles
    theta_list = np.array(pitch_angles)
    
    # Chord distribution
    if chord_distribution == 'polynomial':
        # Polynomial (quadratic, n=2) taper from hub to tip
        # c(r) = c_tip + (c_root - c_tip) * ((R - r)/(R - r_hub))^n
        # Design constraint: c_tip = 0.5 * c_root
        c_tip = 0.5 * c_root
        n = 2  # quadratic exponent
        
        # Normalized distance from tip (1 at hub, 0 at tip)
        r_normalized = (MAX_RADIUS_LOCAL - r_list) / (MAX_RADIUS_LOCAL - HUB_RADIUS)
        chord_list = c_tip + (c_root - c_tip) * r_normalized**n
    elif chord_distribution == 'linear':
        # Linear taper from hub to tip
        chord_max = 0.04  # m at hub
        chord_min = 0.02  # m at tip
        chord_list = np.linspace(chord_max, chord_min, N_RADIAL_POINTS)
    elif chord_distribution == 'constant':
        chord_list = np.ones(N_RADIAL_POINTS) * 0.03  # m
    else:
        raise ValueError(f"Unknown chord distribution: {chord_distribution}")
    
    return r_list, theta_list, chord_list


# ============================================================================
# AERODYNAMIC CALCULATIONS
# ============================================================================

def calculate_forces_and_torque(r_list, theta_list, chord_list, rpm, U,
                                cl_interp, cd_interp):
    """
    Calculate aerodynamic forces and torque for given blade geometry and conditions.
    
    Returns:
    --------
    Q : float
        Total torque (N·m)
    dFn_seg : ndarray
        Normal force per segment (N)
    dFt_seg : ndarray
        Tangential force per segment (N)
    """
    omega = rpm * 2 * np.pi / 60
    
    dFn_seg = np.zeros(N_SEGMENTS)
    dFt_seg = np.zeros(N_SEGMENTS)
    r_mid = np.zeros(N_SEGMENTS)
    
    for seg in range(N_SEGMENTS):
        r1, r2 = r_list[seg], r_list[seg + 1]
        dr = r2 - r1
        r_mid[seg] = (r1 + r2) / 2
        theta_mid = (theta_list[seg] + theta_list[seg + 1]) / 2
        c_mid = (chord_list[seg] + chord_list[seg + 1]) / 2
        
        # Velocity components
        wr = omega * r_mid[seg]
        W = np.sqrt(U**2 + wr**2)
        
        # Angles
        beta_deg = np.degrees(np.arctan(wr / U))
        beta_rad = np.deg2rad(beta_deg)
        phi_deg = 90 - beta_deg
        phi_rad = np.deg2rad(phi_deg)
        alpha_deg = phi_deg - theta_mid
        
        # Lift and drag coefficients
        cl = cl_interp(alpha_deg)
        cd = cd_interp(alpha_deg)
        
        # Forces
        dFn = 0.5 * RHO * W**2 * c_mid * dr * (cl * np.sin(beta_rad) + 
                                                cd * np.sin(phi_rad))
        dFt = 0.5 * RHO * W**2 * c_mid * dr * (cl * np.cos(beta_rad) - 
                                                cd * np.cos(phi_rad))
        
        dFn_seg[seg] = dFn
        dFt_seg[seg] = dFt
    
    # Calculate torque
    Q = np.sum(r_mid * dFt_seg) * B
    
    return Q, dFn_seg, dFt_seg, r_mid


def calculate_max_stress(r_list, theta_list, chord_list, dFn_seg, r_mid, 
                        I_y_non_dim, I_z_non_dim, c_max_non_dim):
    """
    Calculate maximum bending stress in the blade accounting for pitch angle effects.
    
    The pitch angle rotates the bending plane, requiring calculation of the moment
    of inertia in the rotated frame: I_θ = I_y*cos²(θ) + I_z*sin²(θ)
    
    Parameters:
    -----------
    r_list : ndarray
        Radial positions
    theta_list : ndarray
        Pitch angles at each station (degrees)
    chord_list : ndarray
        Chord lengths
    dFn_seg : ndarray
        Normal forces per segment
    r_mid : ndarray
        Midpoint radial positions of segments
    I_y_non_dim : float
        Non-dimensional second moment about y-axis (strong axis)
    I_z_non_dim : float
        Non-dimensional second moment about z-axis (weak axis)
    c_max_non_dim : float
        Non-dimensional maximum extreme fiber distance
    
    Returns:
    --------
    max_sigma : float
        Maximum stress (Pa)
    """
    # Calculate bending moment at each radial station
    M = np.zeros(N_RADIAL_POINTS)
    for k in range(N_RADIAL_POINTS):
        for seg in range(k, N_SEGMENTS):
            if r_mid[seg] > r_list[k]:
                M[k] += dFn_seg[seg] * (r_mid[seg] - r_list[k])
    
    # Calculate stress at each station with rotated moment of inertia
    sigma = np.zeros(N_RADIAL_POINTS)
    for k in range(N_RADIAL_POINTS):
        chord_k = chord_list[k]
        theta_k_rad = np.deg2rad(theta_list[k])
        
        # Calculate rotated moment of inertia based on pitch angle
        # I_θ = I_y*cos²(θ) + I_z*sin²(θ)
        cos_theta_sq = np.cos(theta_k_rad)**2
        sin_theta_sq = np.sin(theta_k_rad)**2
        I_theta_non_dim = I_y_non_dim * cos_theta_sq + I_z_non_dim * sin_theta_sq
        
        # Scale by chord
        I_k = I_theta_non_dim * chord_k**4
        c_k = c_max_non_dim * chord_k
        
        # Calculate stress
        sigma[k] = abs(M[k] * c_k / I_k)
    
    return np.max(sigma)


# ============================================================================
# WEIBULL DISTRIBUTION
# ============================================================================

def weibull_pdf(U, k=WEIBULL_K, c=WEIBULL_C):
    """Weibull probability density function."""
    return (k / c) * (U / c)**(k - 1) * np.exp(-(U / c)**k)


def find_equilibrium_rpm(U, pitch_angles, cl_interp, cd_interp, 
                         rpm_search_range=(0, MAX_RPM)):
    """
    Find equilibrium RPM where torque = 0 for a given wind speed.
    
    Returns:
    --------
    rpm_equil : float
        Equilibrium RPM, or MAX_RPM if no equilibrium found
    """
    r_list, theta_list, chord_list = create_blade_geometry(pitch_angles)
    
    # Search for equilibrium
    rpm_test = np.linspace(rpm_search_range[0], rpm_search_range[1], 100)
    Q_test = np.zeros_like(rpm_test)
    
    for i, rpm in enumerate(rpm_test):
        Q, _, _, _ = calculate_forces_and_torque(r_list, theta_list, chord_list,
                                                  rpm, U, cl_interp, cd_interp)
        Q_test[i] = Q
    
    # Find where Q crosses zero (positive to negative)
    sign_changes = np.where(np.diff(np.sign(Q_test)) < 0)[0]
    
    if len(sign_changes) > 0:
        # Linear interpolation to find exact crossing
        idx = sign_changes[-1]  # Take highest RPM crossing
        rpm_low = rpm_test[idx]
        rpm_high = rpm_test[idx + 1]
        Q_low = Q_test[idx]
        Q_high = Q_test[idx + 1]
        rpm_equil = rpm_low - Q_low * (rpm_high - rpm_low) / (Q_high - Q_low)
        return min(rpm_equil, MAX_RPM)
    else:
        # No equilibrium found, use max RPM
        return MAX_RPM


# ============================================================================
# OPTIMIZATION OBJECTIVE AND CONSTRAINTS
# ============================================================================

def calculate_power_at_wind_speed(U, pitch_angles, blade_length, c_root, rpm, cl_interp, cd_interp,
                                   I_y_non_dim, I_z_non_dim, c_max_non_dim):
    """
    Calculate power output and check constraints for a given wind speed and RPM.
    
    Returns:
    --------
    P : float
        Power output (W), or 0 if constraints violated
    constraint_violation : bool
        True if stress, torque, or RPM constraints violated
    """
    r_list, theta_list, chord_list = create_blade_geometry(pitch_angles, blade_length, c_root)
    
    # Calculate forces and torque
    Q, dFn_seg, _, r_mid = calculate_forces_and_torque(
        r_list, theta_list, chord_list, rpm, U, cl_interp, cd_interp)
    
    # Check torque constraint
    if abs(Q) > MAX_TORQUE_PER_BLADE:
        return 0.0, True
    
    # Check stress constraint
    max_sigma = calculate_max_stress(r_list, theta_list, chord_list, dFn_seg, r_mid, 
                                      I_y_non_dim, I_z_non_dim, c_max_non_dim)
    
    if max_sigma > MAX_STRESS:
        return 0.0, True
    
    # Calculate power
    omega = rpm * 2 * np.pi / 60
    P = Q * omega
    
    return max(P, 0.0), False


def objective_function(x, cl_interp, cd_interp, I_y_non_dim, I_z_non_dim, c_max_non_dim, verbose=False):
    """
    Objective function for optimization: negative expected power + operational constraint penalty.

    Operational Constraint: Maximum power must occur between 1300-1950 RPM.
    Heavy penalty applied if peak power occurs outside this safe operating range.

    Parameters:
    -----------
    x : array-like
        Optimization variables: [pitch_0, ..., pitch_9, rpm, c_root]
        First N_RADIAL_POINTS elements are pitch angles (degrees)
        Next element is target RPM
        Last element is chord root (m)

    Returns:
    --------
    neg_expected_power : float
        Negative of expected power (for minimization) + operational penalties
    """
    # Extract variables
    pitch_angles = x[:N_RADIAL_POINTS]
    rpm_target = x[N_RADIAL_POINTS]
    c_root = x[N_RADIAL_POINTS + 1]
    blade_length = MAX_BLADE_LENGTH  # Fixed to 6 inches

    # Define wind speed range for integration
    U_min, U_max = 1.0, 20.0
    U_samples = np.linspace(U_min, U_max, 20)

    # Calculate expected power with probabilistic constraint penalties
    expected_power = 0.0
    total_penalty = 0.0
    du = U_samples[1] - U_samples[0]  # Integration step size

    for U in U_samples:
        # Use target RPM (controlled by brake)
        P, violation = calculate_power_at_wind_speed(
            U, pitch_angles, blade_length, c_root, rpm_target, cl_interp, cd_interp,
            I_y_non_dim, I_z_non_dim, c_max_non_dim)

        # Weight by Weibull probability
        pdf_val = weibull_pdf(U)
        expected_power += P * pdf_val * du

        # Add probabilistic penalty if constraints violated (weighted by wind probability)
        if violation:
            total_penalty += 1e6 * pdf_val * du  # Heavy penalty weighted by likelihood

    # OPERATIONAL CONSTRAINT: Check if maximum power occurs in safe RPM range (1300-1950 RPM)
    # Calculate power over RPM range to find maximum power point
    operational_penalty = 0.0
    rpm_test_range = np.linspace(100, MAX_RPM, 100)
    power_vs_rpm = []

    r_list_geom, _, chord_list_geom = create_blade_geometry(pitch_angles, blade_length, c_root)
    for rpm in rpm_test_range:
        Q, dFn_seg, _, r_mid = calculate_forces_and_torque(
            r_list_geom, pitch_angles, chord_list_geom, rpm, WEIBULL_C, cl_interp, cd_interp)
        max_sigma = calculate_max_stress(
            r_list_geom, pitch_angles, chord_list_geom, dFn_seg, r_mid, I_y_non_dim, I_z_non_dim, c_max_non_dim)
        P = Q * rpm * 2 * np.pi / 60 if max_sigma <= MAX_STRESS else 0.0
        power_vs_rpm.append(P)

    # Find RPM where maximum power occurs
    max_power_idx = np.argmax(power_vs_rpm)
    peak_power_rpm = rpm_test_range[max_power_idx]

    # Apply heavy penalty if peak power occurs outside safe operational range (1300-1950 RPM)
    SAFE_OPERATIONAL_RPM_MIN = 1300.0
    SAFE_OPERATIONAL_RPM_MAX = 1950.0

    if peak_power_rpm < SAFE_OPERATIONAL_RPM_MIN or peak_power_rpm > SAFE_OPERATIONAL_RPM_MAX:
        operational_penalty = 1e6  # Heavy penalty equivalent to other major constraint violations
        if verbose:
            print(f"OPERATIONAL CONSTRAINT VIOLATION: Peak power at {peak_power_rpm:.0f} RPM (must be {SAFE_OPERATIONAL_RPM_MIN:.0f}-{SAFE_OPERATIONAL_RPM_MAX:.0f} RPM)")

    # Add operational penalty to total
    total_penalty += operational_penalty

    if verbose:
        print(f"Pitch: {pitch_angles[0]:.1f}° to {pitch_angles[-1]:.1f}°, "
              f"RPM: {rpm_target:.0f}, L: {blade_length*1000:.1f}mm, c_root: {c_root*1000:.1f}mm, "
              f"Power: {expected_power:.3f} W, Max Power RPM: {peak_power_rpm:.0f}, "
              f"Total Penalty: {total_penalty:.0e}")

    return -expected_power + total_penalty


# ============================================================================
# OPTIMIZATION ROUTINE
# ============================================================================

def optimize_blade_design(cl_interp, cd_interp, I_y_non_dim, I_z_non_dim, c_max_non_dim,
                          maxiter=100, popsize=15):
    """
    Run optimization to find optimal blade design.
    
    Returns:
    --------
    result : OptimizeResult
        Optimization result containing optimal design parameters
    """
    # Define bounds
    bounds = []
    
    # Pitch angle bounds (degrees) - physically reasonable range
    for i in range(N_RADIAL_POINTS):
        bounds.append((0.0, 45.0))
    
    # RPM bound
    bounds.append((100.0, MAX_RPM))
    
    # Chord root bound (m, 1-2 inches)
    bounds.append((0.0254, 0.0508))
    
    print("Starting optimization...")
    print(f"Number of variables: {len(bounds)} (10 pitch + RPM + c_root)")
    print(f"Blade length fixed to: {MAX_BLADE_LENGTH*1000:.1f} mm (6 inches)")
    print(f"Max iterations: {maxiter}")
    print(f"Population size: {popsize}")
    
    # Run optimization
    start_time = time.time()
    np.random.seed(42)  # For reproducibility

    result = differential_evolution(
        objective_function,
        bounds,
        args=(cl_interp, cd_interp, I_y_non_dim, I_z_non_dim, c_max_non_dim),
        maxiter=maxiter,
        popsize=popsize,
        seed=42,
        disp=True,
        workers=1,
        updating='deferred',
        polish=True
    )
    
    elapsed_time = time.time() - start_time
    
    print(f"\nOptimization complete in {elapsed_time:.1f} seconds")
    print(f"Success: {result.success}")
    print(f"Best expected power: {-result.fun:.3f} W")
    
    return result


# ============================================================================
# RESULTS ANALYSIS
# ============================================================================

def analyze_optimal_design(optimal_params, cl_interp, cd_interp, I_y_non_dim, I_z_non_dim, c_max_non_dim):
    """Analyze and visualize optimal blade design."""
    pitch_angles = optimal_params[:N_RADIAL_POINTS]
    rpm_optimal = optimal_params[N_RADIAL_POINTS]
    c_root = optimal_params[N_RADIAL_POINTS + 1]
    blade_length = MAX_BLADE_LENGTH  # Fixed to 6 inches
    
    r_list, theta_list, chord_list = create_blade_geometry(pitch_angles, blade_length, c_root)
    
    print("\n" + "="*70)
    print("OPTIMAL BLADE DESIGN")
    print("="*70)
    print(f"\nOptimal RPM: {rpm_optimal:.1f}")
    print(f"\nPitch angles (degrees):")
    print(f"  Root (r={r_list[0]*1000:.1f} mm): {pitch_angles[0]:.2f}°")
    print(f"  Mid  (r={r_list[N_RADIAL_POINTS//2]*1000:.1f} mm): "
          f"{pitch_angles[N_RADIAL_POINTS//2]:.2f}°")
    print(f"  Tip  (r={r_list[-1]*1000:.1f} mm): {pitch_angles[-1]:.2f}°")
    print(f"\nTwist (root to tip): {pitch_angles[0] - pitch_angles[-1]:.2f}°")
    
    # Test at various wind speeds
    print(f"\n{'Wind Speed':<15} {'Power (W)':<15} {'Max Stress (MPa)':<20}")
    print("-" * 50)
    
    U_test = np.linspace(3, 15, 7)
    for U in U_test:
        P, violation = calculate_power_at_wind_speed(
            U, pitch_angles, blade_length, c_root, rpm_optimal, cl_interp, cd_interp, 
            I_y_non_dim, I_z_non_dim, c_max_non_dim)
        
        # Get stress for display
        r_list, theta_list, chord_list = create_blade_geometry(pitch_angles, blade_length, c_root)
        Q, dFn_seg, _, r_mid = calculate_forces_and_torque(
            r_list, theta_list, chord_list, rpm_optimal, U, 
            cl_interp, cd_interp)
        max_sigma = calculate_max_stress(r_list, theta_list, chord_list, dFn_seg, 
                                          r_mid, I_y_non_dim, I_z_non_dim, c_max_non_dim)
        
        violation_flag = " (VIOLATED)" if violation else ""
        print(f"{U:<15.1f} {P:<15.3f} {max_sigma/1e6:<20.2f}{violation_flag}")
    
    # Plot results
    plot_optimal_design(optimal_params, cl_interp, cd_interp, I_y_non_dim, I_z_non_dim, c_max_non_dim)


def plot_optimal_design(optimal_params, cl_interp, cd_interp, I_y_non_dim, I_z_non_dim, c_max_non_dim):
    """Create visualization plots for optimal design."""
    pitch_angles = optimal_params[:N_RADIAL_POINTS]
    rpm_optimal = optimal_params[N_RADIAL_POINTS]
    c_root = optimal_params[N_RADIAL_POINTS + 1]
    blade_length = MAX_BLADE_LENGTH  # Fixed to 6 inches
    
    r_list, theta_list, chord_list = create_blade_geometry(pitch_angles, blade_length, c_root)
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Plot 1: Pitch and chord distribution
    ax1 = axes[0, 0]
    ax1_twin = ax1.twinx()
    ax1.plot(r_list * 1000, pitch_angles, 'b-o', label='Pitch Angle')
    ax1_twin.plot(r_list * 1000, chord_list * 1000, 'r-s', label='Chord')
    ax1.set_xlabel('Radial Position (mm)')
    ax1.set_ylabel('Pitch Angle (degrees)', color='b')
    ax1_twin.set_ylabel('Chord (mm)', color='r')
    ax1.set_title('Blade Geometry')
    ax1.grid(True, alpha=0.3)
    ax1.tick_params(axis='y', labelcolor='b')
    ax1_twin.tick_params(axis='y', labelcolor='r')
    
    # Plot 2: Power vs wind speed
    ax2 = axes[0, 1]
    U_range = np.linspace(1, 20, 50)
    P_range = []
    for U in U_range:
        P, _ = calculate_power_at_wind_speed(
            U, pitch_angles, blade_length, c_root, rpm_optimal, cl_interp, cd_interp, 
            I_y_non_dim, I_z_non_dim, c_max_non_dim)
        P_range.append(P)
    ax2.plot(U_range, P_range, 'g-', linewidth=2)
    ax2.set_xlabel('Wind Speed (m/s)')
    ax2.set_ylabel('Power Output (W)')
    ax2.set_title(f'Power Output at {rpm_optimal:.0f} RPM')
    ax2.grid(True, alpha=0.3)
    
    # Plot 3: Stress vs wind speed
    ax3 = axes[1, 0]
    stress_range = []
    for U in U_range:
        Q, dFn_seg, _, r_mid = calculate_forces_and_torque(
            r_list, theta_list, chord_list, rpm_optimal, U, 
            cl_interp, cd_interp)
        max_sigma = calculate_max_stress(r_list, theta_list, chord_list, dFn_seg, 
                                          r_mid, I_y_non_dim, I_z_non_dim, c_max_non_dim)
        stress_range.append(max_sigma / 1e6)
    ax3.plot(U_range, stress_range, 'm-', linewidth=2)
    ax3.axhline(MAX_STRESS / 1e6, color='r', linestyle='--', 
                label=f'Max Allowed ({MAX_STRESS/1e6:.1f} MPa)')
    ax3.set_xlabel('Wind Speed (m/s)')
    ax3.set_ylabel('Max Stress (MPa)')
    ax3.set_title('Structural Stress')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Plot 4: Weibull distribution and weighted power
    ax4 = axes[1, 1]
    pdf_vals = [weibull_pdf(U) for U in U_range]
    weighted_power = [P * pdf for P, pdf in zip(P_range, pdf_vals)]
    ax4_twin = ax4.twinx()
    ax4.plot(U_range, pdf_vals, 'b-', label='Weibull PDF', linewidth=2)
    ax4_twin.plot(U_range, weighted_power, 'g--', 
                  label='Weighted Power', linewidth=2)
    ax4.set_xlabel('Wind Speed (m/s)')
    ax4.set_ylabel('Probability Density', color='b')
    ax4_twin.set_ylabel('Weighted Power (W)', color='g')
    ax4.set_title('Wind Distribution & Expected Power')
    ax4.tick_params(axis='y', labelcolor='b')
    ax4_twin.tick_params(axis='y', labelcolor='g')
    ax4.grid(True, alpha=0.3)
    ax4.legend(loc='upper left')
    ax4_twin.legend(loc='upper right')
    
    plt.tight_layout()
    plt.savefig(str(RESULTS_DIR / 'optimal_blade_design.png'), dpi=300)
    print(f"\nPlots saved to '{str(RESULTS_DIR / 'optimal_blade_design.png')}'")


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    print("="*70)
    print("WINDFOIL BLADE OPTIMIZATION")
    print("="*70)
    
    # Load data
    print("\nLoading NACA 4412 polar data...")
    cl_interp, cd_interp = load_polar_data(str(BASE_DIR / 'data' / 'xf-naca4412-il-50000.csv'))
    
    print("Calculating moments of inertia...")
    I_y_non_dim, I_z_non_dim, c_max_non_dim = calculate_naca4412_I()
    print(f"Non-dimensional I_y (strong axis): {I_y_non_dim:.10f}")
    print(f"Non-dimensional I_z (weak axis): {I_z_non_dim:.10f}")
    print(f"Non-dimensional c_max: {c_max_non_dim:.10f}")
    
    # Run optimization (longer iterations for more complex aero model)
    result = optimize_blade_design(cl_interp, cd_interp, I_y_non_dim, I_z_non_dim, c_max_non_dim,
                                   maxiter=2000, popsize=60)
    
    # Analyze results - save even if not fully converged
    if result.success:
        CONVERGENCE_STATUS = f"OPTIMIZATION RESULT: SUCCESS (Converged in {result.nit} iterations!)"
    else:
        CONVERGENCE_STATUS = f"OPTIMIZATION RESULT: WARNING (Reached {result.nfev} function evaluations, {result.nit} iterations - best solution saved)"

    print("\n" + "="*80)
    print(CONVERGENCE_STATUS)
    print("="*80)
    
    analyze_optimal_design(result.x, cl_interp, cd_interp, I_y_non_dim, I_z_non_dim, c_max_non_dim)
    
    # Save optimal parameters
    pitch_angles = result.x[:N_RADIAL_POINTS]
    rpm_optimal = result.x[N_RADIAL_POINTS]
    c_root = result.x[N_RADIAL_POINTS + 1]
    blade_length = MAX_BLADE_LENGTH  # Fixed to 6 inches

    # Create blade geometry to get chord distribution
    r_list, theta_list, chord_list = create_blade_geometry(pitch_angles, blade_length, c_root)

    output_data = {
        'radial_position_mm': r_list * 1000,
        'pitch_angle_deg': pitch_angles,
        'chord_mm': chord_list * 1000,
        'optimal_rpm': rpm_optimal,
        'blade_length_mm': blade_length * 1000,
        'c_root_mm': c_root * 1000
    }

    df_output = pd.DataFrame({
        'radial_position_mm': output_data['radial_position_mm'],
        'pitch_angle_deg': output_data['pitch_angle_deg'],
        'chord_mm': output_data['chord_mm']
    })
    df_output.to_csv(str(RESULTS_DIR / 'optimal_blade_parameters.csv'), index=False)
    print(f"\nOptimal parameters saved to '{str(RESULTS_DIR / 'optimal_blade_parameters.csv')}'")
    print(f"Blade length: {MAX_BLADE_LENGTH*1000:.1f} mm (fixed), c_root: {c_root*1000:.1f} mm, c_tip: {c_root*500:.1f} mm")
    
    return result


if __name__ == "__main__":
    main()
