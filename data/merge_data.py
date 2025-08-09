import h5py
import numpy as np
import glob
import os

def merge_h5_rows(file_list, output_filename='merged_rows.h5'):
    """
    Reads a list of H5 files with the same internal structure and merges them
    by concatenating the rows of each corresponding dataset. It automatically
    creates the output directory if it doesn't exist.

    Args:
        file_list (list): A list of strings containing the paths to the H5 files.
        output_filename (str): The name of the output H5 file, which can include a path.
    """
    if not file_list:
        print("File list is empty. Nothing to merge.")
        return

    # Dictionaries to hold the lists of data from each file for each dataset
    # Assumes these are the keys you want to merge.
    data_to_merge = {
        'x_current': [],
        'x_ref': [],
        'u_opt': []
    }
    
    # --- Step 1: Read and collect data from all files ---
    print("--- Reading and collecting data ---")
    for filename in file_list:
        try:
            with h5py.File(filename, 'r') as hf:
                # Check if all required keys exist in the current H5 file
                required_keys = data_to_merge.keys()
                if not all(key in hf for key in required_keys):
                    print(f"--> Skipping file '{filename}' because it is missing one of the required datasets.")
                    continue

                # Append the data from each dataset to the corresponding list
                for key in required_keys:
                    data_to_merge[key].append(hf[key][:])
                
                print(f"Successfully collected data from '{filename}'")

        except Exception as e:
            print(f"--> ERROR: Could not process file '{filename}'. Reason: {e}")
            
    # --- Step 2: Concatenate the collected data ---
    print("\n--- Concatenating datasets ---")
    merged_data = {}
    for key, data_list in data_to_merge.items():
        # Only proceed if we actually collected data for this key
        if data_list:
            merged_data[key] = np.vstack(data_list)
            print(f"Merged dataset '{key}' has shape: {merged_data[key].shape}")
        else:
            print(f"No data was collected for dataset '{key}'.")

    # --- Step 3: Write the merged data to a new H5 file ---
    if not merged_data:
        print("\nNo data was successfully merged. Output file will not be created.")
        return
        
    print(f"\n--- Saving merged data to '{output_filename}' ---")

    # --- FIX: Ensure the output directory exists before trying to save the file ---
    # Get the directory part of the output file path
    output_dir = os.path.dirname(output_filename)
    
    # If output_dir is not an empty string (i.e., the path has a directory),
    # create it. The 'exist_ok=True' prevents an error if it already exists.
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    # --- END FIX ---
    
    try:
        with h5py.File(output_filename, 'w') as hf_out:
            for key, data in merged_data.items():
                hf_out.create_dataset(key, data=data)
        
        print("\nMerging complete. File saved successfully.")
    except Exception as e:
        print(f"\n--> ERROR: Failed to write output file. Reason: {e}")


if __name__ == '__main__':
    # So we define the parent directory of the script.
    script_parent_dir = os.path.dirname(os.path.abspath(__file__))

    # Use glob to find all files starting with 'data_ins_' and ending with '.h5'
    # inside the directory where the script is located.
    # This makes the paths absolute and prevents location-based errors.
    search_path = os.path.join(script_parent_dir, 'data_ins_*.h5')
    h5_files = glob.glob(search_path)
    h5_files.sort() # Sort the files to ensure a consistent merge order

    if not h5_files:
        print(f"No '.h5' files found at path: {search_path}")
        print("Please check the filenames and the script's location.")
    else:
        # Define the name for the final merged file.
        # It will be saved in the same directory as the script.
        output_file = os.path.join(script_parent_dir, 'merged_rows.h5')

        # Run the merging function
        merge_h5_rows(h5_files, output_filename=output_file)