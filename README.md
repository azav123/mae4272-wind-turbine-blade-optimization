# Wind Turbine Blade Optimization

MAE 4272 Design Project - Wind Turbine Blade Optimization

## Project Overview

This project optimizes wind turbine blade design (pitch/twist distribution and operating RPM) to maximize expected power output over a Weibull wind distribution using the industry-standard **Viterna-Corrigan aerodynamic extrapolation method**.

**Design Constraints:**
- Max blade length: 6 inches (152.4 mm)
- Hub radius: 1 inch (25.4 mm)
- Max RPM: 2000
- Max stress: 29.33 MPa (with 1.5 safety factor)
- Wind: Weibull distribution (k=5, c=5 m/s)
- Airfoil: NACA 4412

## Quick Start

```bash
python src/run_full_analysis.py
```

This master script runs the complete optimization workflow. It will prompt for confirmation before proceeding.

## Project Structure

- **`src/`** - Core source code
  - `run_full_analysis.py` - Master script for complete workflow
  - `windfoil_optimization_viterna.py` - Main optimization script
  - `enhanced_analysis.py` - Post-optimization analysis
  - `final_report.py` - Report generation
- **`data/`** - Input data files (airfoil polars)
- **`outputs/`** - Results directory
  - `viterna_corrigan/` - Improved results (post-penalty fix)
  - `submission_results/` - Original archived results (pre-fix)
- **`docs/`** - Documentation and notes
- **`notebooks/`** - Jupyter notebooks for reference
- **`archive/`** - Previous iterations
- **`variants/s822/`** - Alternative airfoil variant

## Running the Optimization

### Full Optimization (Recommended)

For production-quality results:
```bash
python src/run_full_analysis.py
```

This runs differential evolution optimization with popsize=60, maxiter=2000 (estimated 5-10 hours runtime). It includes:
1. Viterna-Corrigan optimization with Weibull-weighted penalties
2. Enhanced analysis (smoothing, RPM analysis)

### Quick Test Run

For testing or development:
```bash
python src/windfoil_optimization_viterna.py --popsize 10 --maxiter 50
```

This provides a fast preview (5-10 minutes) with lower fidelity.

### Individual Scripts

Run components separately if needed:
```bash
# Optimization only
python src/windfoil_optimization_viterna.py

# Analysis only
python src/enhanced_analysis.py
```

## Key Features

### Viterna-Corrigan Aerodynamic Extrapolation

Industry-standard method for wind turbine aerodynamics:
- Physically correct post-stall behavior
- Smooth transitions at stall points
- Suitable for high pitch angles at blade root

### Probabilistic Penalty Weighting

Penalties are scaled by Weibull PDF values (k=5.0, c=5.0):
- Rare high wind speeds contribute minimally to total penalty
- Optimizes for design mean wind speed (4.5 m/s)
- Improves torque/power at typical operating conditions

### Enhanced Analysis Features

Post-optimization processing includes:
- Polynomial smoothing of pitch distribution
- Power vs RPM performance curves
- Torque brake specifications
- Stress constraint verification

## Results

Two sets of results are included:

- **outputs/viterna_corrigan/** → improved version (post-submission)
  - Penalty terms now weighted by Weibull probability density → realistic behavior at design wind speed (~4.5 m/s)
  - Higher torque/power expected at mean operating condition

- **outputs/submission_results/** → original results submitted for grading (October 2025)
  - See README.md in that folder for explanation of known limitation

**Improved Results** (outputs/viterna_corrigan/):
- `optimal_blade_design.png` - Design visualization
- `optimal_blade_parameters.csv` - Raw optimized parameters
- `smoothed_blade_parameters.csv` - Manufacturing-ready smoothed parameters
- `power_vs_rpm_analysis.png` - Performance analysis

**Archived Original Results** (outputs/submission_results/):
- Original submission results from October 2025
- May contain penalty weighting bugs
- See included README.md for details

## Post-Submission Updates

- **Path Improvements**: Replaced hardcoded paths with pathlib-based relative paths
- **Probabilistic Penalty Fix**: Implemented Weibull PDF weighting for penalties (Option 2)
- **Result Archiving**: Moved original submission results to dedicated archive folder
- **Optimization Fidelity**: Increased to popsize=60, maxiter=2000 for higher accuracy
- **Project Reorganization**: Cleaned structure with src/, data/, outputs/, docs/, etc.

## Potential Future Improvements

- Hyperparameter tuning for differential evolution algorithm
- Advanced DE variants (e.g., self-adaptive parameters)
- Cloud computing for parallel optimization runs
- Experimental validation of optimized designs
- 3D aerodynamic effects and tip loss corrections
- Multi-objective optimization (power vs. structural weight)

## Dependencies

```bash
pip install -r requirements.txt
```

Required packages: numpy, scipy, pandas, matplotlib

## Documentation

- `docs/OPTIMIZATION_DOCUMENTATION.txt` - Technical background
- `docs/notes.txt` - Implementation details and rationale

## Contact

For questions about this design project, refer to MAE 4272 course materials and documentation in `docs/provided/`.

## License

MIT License - See LICENSE file for details