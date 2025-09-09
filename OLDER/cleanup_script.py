import os
import shutil
from pathlib import Path

def cleanup_project():
    """
    Clean up the project by organizing files into current and older folders
    """
    
    print("📁 PROJECT CLEANUP SCRIPT")
    print("="*50)
    
    # Create OLDER folder if it doesn't exist
    older_folder = Path("OLDER")
    older_folder.mkdir(exist_ok=True)
    print(f"✅ Created/confirmed OLDER folder")
    
    # Define which files to KEEP (these are the good ones)
    files_to_keep = {
        # Core scripts we want to keep
        'z08_price_path.py',  # Linear price path analysis (theoretical straddle decay)
        'z09_random_price_paths.py',  # Random walk price paths 
        'z17_5_year_simulation.py',  # 5-year simulation
        'z16.py',  # 5-asset portfolio simulation
        'z18_plots_for_z17.py',  # Visualization script for z17
        
        # Essential project files
        '.gitignore',
        'README.md',
        
        # Any CSV data files you want to keep
        'spot_prices.csv',  # Original data
    }
    
    # Files to move to OLDER (everything else)
    files_to_move = [
        'Z01.py', 'z02.py', 'z03.py', 'z04.py', 'z05.py', 'z06.py', 'z07.py',
        'z10_random_price_path_trades_simulator.py',
        'z11_big_simulator.py',
        'z12_better_simulation.py', 
        'z13.py',
        'z14_simulation.py',
        'z15_plots.py',
        'z15_plots_2.py',
    ]
    
    # Output files to move (CSV and PNG files from old analyses)
    output_patterns_to_move = [
        '*straddle_analysis.png',
        '*_straddle_values.csv',
        '*moneyness*.csv',
        '*analysis*.csv',
        '*removed*.csv',
        '*kept*.csv',
        '*backtest*.csv',
        '*simulation*.csv',
        '*trading*.csv',
        '*individual*.png',
        '*batch*.png',
        '*portfolio*.png',
        '*correlation*.png',
        '*risk*.png',
        '*continuous*.csv',
        '*realistic*.csv',
        '*timeline*.csv',
    ]
    
    moved_count = 0
    kept_count = 0
    
    print(f"\n📋 FILES TO KEEP:")
    for file in files_to_keep:
        if Path(file).exists():
            print(f"  ✅ {file}")
            kept_count += 1
        else:
            print(f"  ❓ {file} (not found)")
    
    print(f"\n📦 MOVING FILES TO OLDER FOLDER:")
    
    # Move specific script files
    for file in files_to_move:
        file_path = Path(file)
        if file_path.exists():
            try:
                shutil.move(str(file_path), str(older_folder / file_path.name))
                print(f"  📁 Moved: {file}")
                moved_count += 1
            except Exception as e:
                print(f"  ❌ Error moving {file}: {e}")
        else:
            print(f"  ❓ Not found: {file}")
    
    # Move output files by pattern
    current_dir = Path(".")
    for pattern in output_patterns_to_move:
        matching_files = list(current_dir.glob(pattern))
        for file_path in matching_files:
            if file_path.name not in files_to_keep:
                try:
                    shutil.move(str(file_path), str(older_folder / file_path.name))
                    print(f"  📁 Moved: {file_path.name}")
                    moved_count += 1
                except Exception as e:
                    print(f"  ❌ Error moving {file_path.name}: {e}")
    
    # Move any existing straddle folder
    straddle_folder = Path("straddle")
    if straddle_folder.exists() and straddle_folder.is_dir():
        try:
            shutil.move(str(straddle_folder), str(older_folder / "straddle"))
            print(f"  📁 Moved entire straddle/ folder")
            moved_count += 1
        except Exception as e:
            print(f"  ❌ Error moving straddle folder: {e}")
    
    print(f"\n" + "="*50)
    print("CLEANUP SUMMARY")
    print("="*50)
    print(f"📁 Files moved to OLDER: {moved_count}")
    print(f"✅ Files kept in main directory: {kept_count}")
    
    print(f"\n📋 REMAINING FILES IN MAIN DIRECTORY:")
    remaining_files = [f for f in os.listdir(".") if Path(f).is_file() and not f.startswith('.')]
    for file in sorted(remaining_files):
        print(f"  📄 {file}")
    
    print(f"\n🎯 NEXT STEPS:")
    print(f"1. Verify the core scripts are working:")
    print(f"   - z08_price_path.py (linear price path analysis)")
    print(f"   - z09_random_price_paths.py (random walk version)")
    print(f"   - z17_5_year_simulation.py (5-year trading simulation)")
    print(f"   - z16.py (5-asset portfolio simulation)")
    print(f"   - z18_plots_for_z17.py (visualization script)")
    print(f"")
    print(f"2. Fix the visualization script if needed")
    print(f"3. Run the core analyses to generate fresh outputs")
    
    print(f"\n✨ Project cleanup complete!")

if __name__ == "__main__":
    cleanup_project()