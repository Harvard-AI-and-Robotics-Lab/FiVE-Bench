import csv
import os


result_avg_files = []
for i in range(1, 7):
    result_path = f"8_wan_edit/edit{i}_FiVE_evaluation_result_frame_stride8.csv"

    if not os.path.exists(result_path):
        continue

    # calculate the average of each metric (each column)
    with open(result_path, 'r') as f:
        reader = list(csv.reader(f))
        header, rows = reader[0], reader[1:]

    avg_row = []
    # Process each column by index to handle rows with different lengths
    for col_idx, name in enumerate(header):
        print("processing", name)
        # Extract column values, handling missing values
        col_values = []
        for row in rows:
            if col_idx < len(row):
                col_values.append(row[col_idx])
            else:
                col_values.append("")  # Use empty string for missing values
        
        try:
            # Filter out empty strings and convert to float
            values = [float(x) for x in col_values if x != "" and x != "nan"]
            if values:  # Only calculate average if there are valid values
                avg = sum(values) / len(values)
                if 'structure_distance' in name:
                    avg *= 1000
                elif 'lpips_' in name:
                    avg *= 1000
                elif 'mse_' in name:
                    avg *= 10000
                elif 'ssim_' in name:
                    avg *= 100
                elif 'motion_fidelity_score' in name:
                    avg *= 100
                elif name.startswith('five_acc'):
                    avg *= 100
                avg_row.append(f"{avg:.4f}")
            else:
                avg_row.append("N/A")
        except ValueError:
            avg_row.append("N/A")


    with open(result_path.replace('.csv', '_avg.csv'), 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(header)
        writer.writerow(avg_row)

    result_avg_files.append(result_path.replace('.csv', '_avg.csv'))


# average the results in result_avg_files
if result_avg_files:
    all_avg_rows = []
    
    # Read all average files
    for result_avg_file in result_avg_files:
        with open(result_avg_file, 'r') as f:
            reader = list(csv.reader(f))
            header, rows = reader[0], reader[1:]
            if rows:  # Make sure there's data
                all_avg_rows.append(rows[0])  # Get the average row
    
    # Calculate final averages across all files
    final_avg_row = []
    for col_idx, name in enumerate(header):
        print("final averaging", name)
        
        # Extract values from all average files for this column
        col_values = []
        for avg_row in all_avg_rows:
            if col_idx < len(avg_row) and avg_row[col_idx] != "N/A":
                try:
                    col_values.append(float(avg_row[col_idx]))
                except ValueError:
                    pass  # Skip non-numeric values
        
        # Calculate final average
        if col_values:
            final_avg = sum(col_values) / len(col_values)
            final_avg_row.append(f"{final_avg:.4f}")
        else:
            final_avg_row.append("N/A")
    
    # Write final averaged results
    with open(f"{result_path.split('/')[-2]}/final_averaged_results.csv", 'w', newline='') as f_out:
        writer = csv.writer(f_out)
        writer.writerow(header)
        writer.writerow(final_avg_row)