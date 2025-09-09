import os
import shutil
import glob
from pathlib import Path

def move_straddle_plots_to_folder(folder_name="straddle"):
    """
    Move all straddle analysis plot files into a specified folder
    """
    
    print("STRADDLE PLOTS ORGANIZER")
    print("="*50)
    print(f"Moving straddle plot files to '{folder_name}' folder...")
    
    # Create the target folder if it doesn't exist
    target_folder = Path(folder_name)
    
    try:
        target_folder.mkdir(exist_ok=True)
        print(f"✅ Target folder '{folder_name}' ready")
    except Exception as e:
        print(f"❌ Error creating folder '{folder_name}': {str(e)}")
        return
    
    # Find all straddle analysis plot files
    # Pattern: *_straddle_analysis.png
    plot_pattern = "*_straddle_analysis.png"
    plot_files = glob.glob(plot_pattern)
    
    if not plot_files:
        print(f"⚠️  No straddle plot files found matching pattern: {plot_pattern}")
        print("   Make sure you're running this script in the directory containing the plot files")
        return
    
    print(f"\nFound {len(plot_files)} straddle plot files:")
    for plot_file in sorted(plot_files):
        print(f"  📊 {plot_file}")
    
    # Ask for confirmation
    print(f"\n⚠️  About to move {len(plot_files)} files to '{folder_name}' folder")
    response = input("Do you want to proceed? (yes/no): ").lower().strip()
    
    if response not in ['yes', 'y']:
        print("❌ Operation cancelled by user")
        return
    
    # Move the files
    print(f"\n🚀 Moving files...")
    
    moved_count = 0
    failed_count = 0
    
    for plot_file in plot_files:
        try:
            source_path = Path(plot_file)
            target_path = target_folder / source_path.name
            
            # Move the file
            shutil.move(str(source_path), str(target_path))
            print(f"✅ Moved: {plot_file} → {target_path}")
            moved_count += 1
            
        except Exception as e:
            print(f"❌ Failed to move {plot_file}: {str(e)}")
            failed_count += 1
    
    # Summary
    print(f"\n" + "="*50)
    print("MOVE OPERATION COMPLETE")
    print("="*50)
    print(f"Successfully moved: {moved_count} files")
    print(f"Failed to move: {failed_count} files")
    print(f"Target folder: {target_folder.absolute()}")
    
    if moved_count > 0:
        print(f"\n🎉 All straddle plots are now organized in the '{folder_name}' folder!")
    
    # List contents of target folder
    if target_folder.exists():
        files_in_folder = list(target_folder.glob("*.png"))
        if files_in_folder:
            print(f"\nContents of '{folder_name}' folder ({len(files_in_folder)} files):")
            for file_path in sorted(files_in_folder):
                print(f"  📊 {file_path.name}")

def move_additional_files_to_folder(folder_name="straddle", file_patterns=None):
    """
    Move additional analysis files to the straddle folder
    """
    
    if file_patterns is None:
        file_patterns = [
            "*straddle*.csv",
            "*moneyness*.csv", 
            "*analysis*.csv",
            "*removed*.csv",
            "*kept*.csv"
        ]
    
    print(f"\n" + "="*50)
    print("MOVING ADDITIONAL ANALYSIS FILES")
    print("="*50)
    
    target_folder = Path(folder_name)
    
    all_additional_files = []
    for pattern in file_patterns:
        files = glob.glob(pattern)
        all_additional_files.extend(files)
    
    # Remove duplicates and sort
    all_additional_files = sorted(list(set(all_additional_files)))
    
    if not all_additional_files:
        print("⚠️  No additional analysis files found")
        return
    
    print(f"Found {len(all_additional_files)} additional analysis files:")
    for file_name in all_additional_files:
        print(f"  📄 {file_name}")
    
    response = input(f"\nMove these files to '{folder_name}' folder too? (yes/no): ").lower().strip()
    
    if response not in ['yes', 'y']:
        print("⏭️  Skipping additional files")
        return
    
    moved_count = 0
    failed_count = 0
    
    for file_name in all_additional_files:
        try:
            source_path = Path(file_name)
            target_path = target_folder / source_path.name
            
            # Move the file
            shutil.move(str(source_path), str(target_path))
            print(f"✅ Moved: {file_name} → {target_path}")
            moved_count += 1
            
        except Exception as e:
            print(f"❌ Failed to move {file_name}: {str(e)}")
            failed_count += 1
    
    print(f"\nAdditional files moved: {moved_count}")
    print(f"Additional files failed: {failed_count}")

def create_folder_index(folder_name="straddle"):
    """
    Create an index file listing all contents of the straddle folder
    """
    
    target_folder = Path(folder_name)
    
    if not target_folder.exists():
        print(f"⚠️  Folder '{folder_name}' doesn't exist")
        return
    
    # Get all files in the folder
    png_files = sorted(target_folder.glob("*.png"))
    csv_files = sorted(target_folder.glob("*.csv"))
    other_files = sorted([f for f in target_folder.glob("*") if f.is_file() and f.suffix not in ['.png', '.csv']])
    
    # Create index content
    index_content = f"""STRADDLE ANALYSIS FILES INDEX
Generated on: {Path.cwd()}
Date: {os.path.basename(os.getcwd())}

CONTENTS OF '{folder_name}' FOLDER:
{"="*50}

PLOT FILES ({len(png_files)} files):
"""
    
    for png_file in png_files:
        # Extract ticker from filename
        ticker = png_file.stem.replace("_straddle_analysis", "").upper()
        index_content += f"  📊 {png_file.name:<30} - {ticker} Straddle Analysis\n"
    
    if csv_files:
        index_content += f"\nDATA FILES ({len(csv_files)} files):\n"
        for csv_file in csv_files:
            index_content += f"  📄 {csv_file.name}\n"
    
    if other_files:
        index_content += f"\nOTHER FILES ({len(other_files)} files):\n"
        for other_file in other_files:
            index_content += f"  📁 {other_file.name}\n"
    
    index_content += f"\nTOTAL FILES: {len(png_files) + len(csv_files) + len(other_files)}\n"
    
    # Save index file
    index_file = target_folder / "README.txt"
    
    try:
        with open(index_file, 'w') as f:
            f.write(index_content)
        print(f"\n📋 Index file created: {index_file}")
    except Exception as e:
        print(f"❌ Error creating index file: {str(e)}")

def main():
    """
    Main function to organize straddle analysis files
    """
    
    folder_name = "straddle"
    
    print("STRADDLE FILES ORGANIZER")
    print("="*60)
    print("This script will organize your straddle analysis files into a single folder")
    print(f"Target folder: '{folder_name}'")
    print("="*60)
    
    # Step 1: Move plot files
    move_straddle_plots_to_folder(folder_name)
    
    # Step 2: Optionally move additional analysis files
    move_additional_files_to_folder(folder_name)
    
    # Step 3: Create an index file
    create_folder_index(folder_name)
    
    print(f"\n🎉 Organization complete!")
    print(f"📁 All straddle files are now in the '{folder_name}' folder")
    print(f"📋 Check README.txt in the folder for a complete file listing")

if __name__ == "__main__":
    main()