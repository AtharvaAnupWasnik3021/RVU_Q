#!/usr/bin/env python3
"""
Quick-Start Script for IoT Q-Learning Simulation
Handles dependency installation and simulation execution
"""

import subprocess
import sys
import os

def check_python_version():
    """Verify Python version >= 3.7"""
    if sys.version_info < (3, 7):
        print("ERROR: Python 3.7+ required")
        print(f"Current version: {sys.version}")
        sys.exit(1)
    print(f"✓ Python {sys.version.split()[0]} detected")

def check_ffmpeg():
    """Check if ffmpeg is installed"""
    try:
        subprocess.run(['ffmpeg', '-version'], 
                      capture_output=True, 
                      timeout=5,
                      check=True)
        print("✓ FFmpeg installed")
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("✗ FFmpeg not found")
        print("  Install with:")
        print("    Linux/Debian: sudo apt-get install ffmpeg")
        print("    macOS: brew install ffmpeg")
        print("    Windows: choco install ffmpeg")
        return False

def install_dependencies():
    """Install Python dependencies"""
    print("\n" + "="*80)
    print("Installing dependencies...")
    print("="*80)
    
    try:
        subprocess.run([sys.executable, '-m', 'pip', 'install', '--upgrade', 'pip'],
                      check=True, 
                      capture_output=True)
        print("✓ pip upgraded")
        
        # Install requirements
        subprocess.run([sys.executable, '-m', 'pip', 'install', '-r', 'requirements.txt'],
                      check=False)
        print("✓ Dependencies installed")
        return True
    except Exception as e:
        print(f"✗ Dependency installation failed: {e}")
        return False

def verify_imports():
    """Verify all required imports are available"""
    print("\n" + "="*80)
    print("Verifying imports...")
    print("="*80)
    
    required = ['numpy', 'pandas', 'matplotlib', 'networkx']
    failed = []
    
    for module in required:
        try:
            __import__(module)
            print(f"✓ {module}")
        except ImportError:
            print(f"✗ {module} - FAILED")
            failed.append(module)
    
    return len(failed) == 0

def run_simulation():
    """Execute the main simulation"""
    print("\n" + "="*80)
    print("STARTING SIMULATION")
    print("="*80)
    
    try:
        subprocess.run([sys.executable, 'iot_qlearning_simulation.py'],
                      check=True)
        return True
    except subprocess.CalledProcessError as e:
        print(f"✗ Simulation failed: {e}")
        return False

def verify_outputs():
    """Check that output files were created"""
    print("\n" + "="*80)
    print("Verifying outputs...")
    print("="*80)
    
    expected_files = [
        'network_animation.mp4',
        'simulation_metrics.csv'
    ]
    
    all_exist = True
    for filename in expected_files:
        if os.path.exists(filename):
            size = os.path.getsize(filename)
            print(f"✓ {filename} ({size:,} bytes)")
        else:
            print(f"✗ {filename} - NOT FOUND")
            all_exist = False
    
    return all_exist

def main():
    """Main execution flow"""
    print("\n")
    print("╔" + "="*78 + "╗")
    print("║" + " "*78 + "║")
    print("║" + "IoT Mesh Network Q-Learning Simulation - Quick Start".center(78) + "║")
    print("║" + "Security-Aware Routing Under Node Infection".center(78) + "║")
    print("║" + " "*78 + "║")
    print("╚" + "="*78 + "╝")
    
    # Step 1: Verify Python
    print("\n[1/5] Checking Python version...")
    check_python_version()
    
    # Step 2: Check FFmpeg
    print("\n[2/5] Checking FFmpeg installation...")
    ffmpeg_ok = check_ffmpeg()
    if not ffmpeg_ok:
        print("\nWARNING: FFmpeg not installed. Animation generation will fail.")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            sys.exit(1)
    
    # Step 3: Install dependencies
    print("\n[3/5] Installing Python dependencies...")
    if not install_dependencies():
        print("\nWARNING: Some dependencies may not have installed correctly.")
        response = input("Continue anyway? (y/n): ").strip().lower()
        if response != 'y':
            sys.exit(1)
    
    # Step 4: Verify imports
    print("\n[4/5] Verifying Python imports...")
    if not verify_imports():
        print("\nERROR: Required packages not available")
        sys.exit(1)
    
    # Step 5: Run simulation
    print("\n[5/5] Running simulation...")
    if not run_simulation():
        print("\nERROR: Simulation failed")
        sys.exit(1)
    
    # Verify outputs
    print("\nVerifying output files...")
    if verify_outputs():
        print("\n" + "="*80)
        print("SUCCESS! Simulation completed successfully")
        print("="*80)
        print("\nOutput files:")
        print("  • network_animation.mp4 - Full animation (1920×1080, 60 FPS)")
        print("  • simulation_metrics.csv - Performance metrics across infection levels")
        print("\nNext steps:")
        print("  1. Play animation: ffmpeg -i network_animation.mp4")
        print("  2. View metrics:  cat simulation_metrics.csv")
        print("  3. Customize:     Edit SIMULATION_PARAMS in iot_qlearning_simulation.py")
        print("\nFor detailed information, see SIMULATION_GUIDE.md")
    else:
        print("\nWARNING: Simulation ran but some output files are missing")
        sys.exit(1)

if __name__ == '__main__':
    main()