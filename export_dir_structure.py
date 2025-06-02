import os

# Define the root directory to scan
root_dir = r"C:\Users\User\Desktop\tesis\data\corrosion images\corrosion images"

# List to hold all file and directory paths
paths = []

# Traverse the directory tree
for dirpath, dirnames, filenames in os.walk(root_dir):
    # Record the current directory
    relative_dir = os.path.relpath(dirpath, root_dir)
    paths.append(f"{relative_dir}/")
    # Record files in the current directory
    for file in filenames:
        paths.append(os.path.join(relative_dir, file))

# Determine splitting criteria
max_lines_per_file = 500# Adjust as needed
total_lines = len(paths)
num_parts = (total_lines // max_lines_per_file) + 1

# Write output into one or more text files
for part_num in range(num_parts):
    start_index = part_num * max_lines_per_file
    end_index = start_index + max_lines_per_file
    part_paths = paths[start_index:end_index]
    
    output_filename = f"structure_part{part_num + 1}.txt"
    
    with open(output_filename, 'w', encoding='utf-8') as f:
        for line in part_paths:
            f.write(line + '\n')

print(f"Created {num_parts} text file(s) with directory structure.")

