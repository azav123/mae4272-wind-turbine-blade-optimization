"""
Master Script - Run Full Windfoil Optimization Analysis (Viterna-Corrigan Only)
================================================================================

This script orchestrates the complete analysis workflow using the Viterna-Corrigan
extrapolation method, which has been confirmed as superior to linear extrapolation.

Workflow:
1. Run Viterna-Corrigan optimization
2. Enhanced analysis on Viterna-Corrigan results

Usage: python run_full_analysis.py
"""

import subprocess
import sys
import time
import threading

def run_script(script_name, args=None):
    """Run a Python script with progress indicator."""
    cmd = [sys.executable, script_name]
    if args:
        cmd.extend(args)
    
    script_display = script_name
    if args:
        script_display += f" {' '.join(args)}"
    
    print(f"Starting {script_display}...")
    
    # Progress indicator in separate thread
    stop_progress = threading.Event()
    progress_thread = threading.Thread(target=_show_progress, args=(stop_progress,))
    progress_thread.daemon = True
    
    start_time = time.time()
    _show_progress.start_time = start_time  # Update for progress indicator
    
    try:
        # Start progress indicator
        progress_thread.start()
        
        # Run script silently (capture output)
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        
        # Stop progress indicator
        stop_progress.set()
        progress_thread.join(timeout=1)
        
        elapsed = time.time() - start_time
        print(f"\r✓ Completed in {elapsed:.1f} seconds" + " " * 20)
        return True
        
    except subprocess.CalledProcessError as e:
        # Stop progress indicator
        stop_progress.set()
        progress_thread.join(timeout=1)
        
        elapsed = time.time() - start_time
        print(f"\r✗ Failed after {elapsed:.1f} seconds" + " " * 20)
        print("\nError output:")
        print(e.stderr if e.stderr else e.stdout)
        return False
        
    except KeyboardInterrupt:
        stop_progress.set()
        print("\n\n✗ Interrupted by user")
        return False

def _show_progress(stop_event):
    """Show animated progress indicator."""
    dots = 0
    while not stop_event.is_set():
        elapsed = time.time() - _show_progress.start_time
        dot_str = "." * (dots % 4)
        print(f"\rRunning{dot_str:<4} ({elapsed:.0f}s elapsed)", end="", flush=True)
        dots += 1
        time.sleep(2)

# Initialize start time for progress indicator
_show_progress.start_time = time.time()

def main():
    """Run the complete analysis workflow using Viterna-Corrigan method."""
    print("="*70)
    print("WINDFOIL OPTIMIZATION - VITERNA-CORRIGAN ANALYSIS")
    print("="*70)
    print("\nThis will run the optimization using Viterna-Corrigan method:")
    print("  1. Viterna-Corrigan optimization")
    print("  2. Enhanced analysis (smoothing, RPM analysis, etc.)")
    print("\nNote: This may take 10-20 minutes depending on your system.")
    
    response = input("\nProceed? (y/n): ")
    if response.lower() != 'y':
        print("Aborted by user.")
        return
    
    overall_start = time.time()
    
    # Step 1: Viterna-Corrigan optimization
    print("\n" + "#"*70)
    print("# STEP 1/2: Viterna-Corrigan Optimization")
    print("#"*70)
    if not run_script('windfoil_optimization_viterna.py'):
        print("\n✗ Workflow stopped due to error in Step 1")
        return
    
    # Step 2: Viterna-Corrigan enhanced analysis
    print("\n" + "#"*70)
    print("# STEP 2/2: Enhanced Analysis - Viterna-Corrigan")
    print("#"*70)
    if not run_script('enhanced_analysis.py', ['--dir', 'output/viterna_corrigan']):
        print("\n✗ Workflow stopped due to error in Step 2")
        return
    
    # Complete!
    overall_elapsed = time.time() - overall_start
    
    print("\n" + "="*70)
    print("WORKFLOW COMPLETE!")
    print("="*70)
    print(f"\nTotal time: {overall_elapsed/60:.1f} minutes")
    print("\nGenerated files in output/viterna_corrigan/:")
    print("  - optimal_blade_design.png")
    print("  - optimal_blade_parameters.csv")
    print("  - smooth_pitch_analysis.png")
    print("  - smoothed_blade_parameters.csv")
    print("  - power_vs_rpm_analysis.png")
    print("\nOptimal blade parameters ready for manufacturing!")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n✗ Unexpected error: {e}")
        sys.exit(1)
