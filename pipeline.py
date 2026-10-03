import argparse
import subprocess
import os
import sys

def run_command(cmd, step_name):
    print(f"\n{'='*50}\nRUNNING: {step_name}\n{'='*50}")
    result = subprocess.run(cmd, shell=True)
    if result.returncode != 0:
        print(f"ERROR in {step_name}")
        return False
    return True

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--e57", type=str, required=True, help="Path to E57 file")
    args = parser.parse_args()
    
    python_exe = sys.executable
    
    stages = [
        (f"{python_exe} extract_e57.py --e57 \"{args.e57}\"", "E57 Extraction"),
        (f"{python_exe} inspect_dataset.py", "Inspect Dataset"),
        (f"{python_exe} calibrate_direction.py --forward_axis=\"-z\"", "Calibrate Direction"),
        (f"{python_exe} build_embeddings.py", "Build Embeddings"),
        (f"{python_exe} train_model.py", "Train Model"),
        (f"{python_exe} evaluate.py", "Rigorous Evaluation")
    ]
    
    results = {}
    for cmd, name in stages:
        success = run_command(cmd, name)
        results[name] = success
        if not success and name == "E57 Extraction":
            print("Extraction failed, aborting pipeline.")
            break
            
    # Simulate a localization query to test the final step
    print(f"\n{'='*50}\nRUNNING: Test Query Localization\n{'='*50}")
    import glob
    test_imgs = glob.glob("dataset/images/*.jpg")
    query_success = False
    if test_imgs:
        res = subprocess.run(f"{python_exe} localize.py --image \"{test_imgs[0]}\"", shell=True)
        query_success = (res.returncode == 0)
    results["Query Localization"] = query_success
    
    # Generate summary report
    import json
    from db.database import get_db_connection
    from config import OUTPUTS_DIR
    
    conn = get_db_connection()
    cursor = conn.cursor()
    
    cursor.execute("SELECT COUNT(*) FROM scans")
    num_scans = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM images")
    num_images = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM images WHERE file_path IS NOT NULL")
    num_extracted = cursor.fetchone()[0]
    
    cursor.execute("SELECT COUNT(*) FROM images WHERE yaw IS NOT NULL")
    num_valid_pose = cursor.fetchone()[0]
    
    print("\n\n" + "="*50)
    print("FINAL PIPELINE REPORT")
    print("="*50)
    for name, success in results.items():
        status = "SUCCESS" if success else "FAILED"
        print(f"Stage: {name.ljust(25)} [{status}]")
        
    print("\nExtraction Statistics:")
    print(f"  Scans: {num_scans}")
    print(f"  Images Found: {num_images}")
    print(f"  Images Extracted: {num_extracted}")
    print(f"  Valid Poses: {num_valid_pose}")
    
    print("\nPipeline completed.")

if __name__ == "__main__":
    main()
