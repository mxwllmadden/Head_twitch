import os
import h5py
from utils_final import *

# Configuration
folder_name = "1117/h5"
threshold = 5.0
threshold2 = 7.0
save_figure = True
R = 1e-3  # Measurement noise covariance
Q = 9e-5  # Process noise covariance

output_folder = f"results/{folder_name}/{threshold}"
os.makedirs(output_folder, exist_ok=True)

# Initialize a dictionary to store the data for the Excel file
excel_data = {}

# Iterate over all files in the specified folder
for filename in os.listdir(folder_name):
    if filename.endswith(".h5"):
        filename_path = os.path.join(folder_name, filename)
        print(f"Processing file: {filename_path}")

        # Load data from the HDF5 file
        with h5py.File(filename_path, "r") as f:
            dset_names = list(f.keys())
            locations = f["tracks"][:].T
            node_names = [n.decode() for n in f["node_names"][:]]

        # Define indices for body parts
        NOSE_INDEX = 0
        EARL_INDEX = 1
        EARR_INDEX = 2
        BACK_INDEX = 3
        TAIL_BASE_INDEX = 4

        # Extract locations for each body part
        nose_loc = locations[:, NOSE_INDEX, :, :]
        earL_loc = locations[:, EARL_INDEX, :, :]
        earR_loc = locations[:, EARR_INDEX, :, :]
        back_loc = locations[:, BACK_INDEX, :, :]
        tail_loc = locations[:, TAIL_BASE_INDEX, :, :]

        length = len(tail_loc)

                
        # Apply Kalman filter to the data
        nose_loc_filtered_x = kalman_filter(nose_loc[:, 0, 0], R, Q)
        nose_loc_filtered_y = kalman_filter(nose_loc[:, 1, 0], R, Q)
        nose_loc_filtered = np.vstack((nose_loc_filtered_x, nose_loc_filtered_y)).T.reshape(-1, 2, 1)

        back_loc_filtered_x = kalman_filter(back_loc[:, 0, 0], R, Q)
        back_loc_filtered_y = kalman_filter(back_loc[:, 1, 0], R, Q)
        back_loc_filtered = np.vstack((back_loc_filtered_x, back_loc_filtered_y)).T.reshape(-1, 2, 1)

        tail_loc_filtered_x = kalman_filter(tail_loc[:, 0, 0], R, Q)
        tail_loc_filtered_y = kalman_filter(tail_loc[:, 1, 0], R, Q)
        tail_loc_filtered = np.vstack((tail_loc_filtered_x, tail_loc_filtered_y)).T.reshape(-1, 2, 1)

        earL_loc_filtered = earL_loc
        earR_loc_filtered = earR_loc

        # Calculate dynamics using filtered data
        dist_body_to_earL, dist_body_to_earR, velocity_to_earL, velocity_to_earR, acceleration_to_earL, acceleration_to_earR = calculate_dynamics(
            nose_loc_filtered, tail_loc_filtered, back_loc_filtered, earL_loc_filtered, earR_loc_filtered
        )

        dist, vel, acc = calculate_combined_dynamics(dist_body_to_earL, dist_body_to_earR)


        # Sum of absolute accelerations for both ears
        acc_sum = np.abs(acceleration_to_earR) + np.abs(acceleration_to_earL)
        exceeding_frames_acc_sum = [i + 2 for i in range(len(acc_sum)) if abs(acc_sum[i]) > threshold2]
        print("Acc_sum:", exceeding_frames_acc_sum)
        
        # Identify frames where acceleration exceeds ±7.5
        exceeding_frames_L = [i + 2 for i in range(len(acceleration_to_earL)) if abs(acceleration_to_earL[i]) > threshold]
        exceeding_frames_R = [i + 2 for i in range(len(acceleration_to_earR)) if abs(acceleration_to_earR[i]) > threshold]

        # Filter exceeding frames to only those in exceeding_frames_acc_sum
        exceeding_frames_L = [frame for frame in exceeding_frames_L if frame in exceeding_frames_acc_sum]
        exceeding_frames_R = [frame for frame in exceeding_frames_R if frame in exceeding_frames_acc_sum]

        # Combine and sort exceeding frames, removing duplicates
        combined_exceeding_frames = list(set(exceeding_frames_acc_sum))
        combined_exceeding_frames = remove_exceeding_frames_with_nan(nose_loc, combined_exceeding_frames)
        print("Exceeding:", combined_exceeding_frames)
        
        # Calculate angles and filter by those below 100 degrees
        angles = calculate_angles(nose_loc_filtered, back_loc_filtered, tail_loc_filtered)
        angles_below_90_array, new_angles_array = calculate_angles_for_exceeding_frames(angles, combined_exceeding_frames)
        

        if new_angles_array.size == 0:
            print(f"No angles below 90 degrees found for file: {filename}\n")
            new_angles_array = np.empty((0, 2))
            plot_dynamics(acceleration_to_earL, acceleration_to_earR, combined_exceeding_frames, combined_exceeding_frames, output_folder, filename, save_figure)
            continue
        else:
            print("angle:", new_angles_array[:, 0])

        # Calculate distances from nose to earL-earR line for exceeding frames
        threshold_dist = 12
        dist_nose_to_line, filtered_exceeding_frames = calculate_distances_for_exceeding_frames(
            nose_loc_filtered, earL_loc_filtered, earR_loc_filtered, new_angles_array[:, 0], threshold_dist
        )
        if filtered_exceeding_frames.size == 0:
            print(f"No angles below 90 degrees found for file: {filename}\n")
            new_angles_array = np.empty((0, 2))
            plot_dynamics(acceleration_to_earL, acceleration_to_earR, new_angles_array, new_angles_array, output_folder, filename, save_figure)
            continue
        else:
            print("Dist:", filtered_exceeding_frames)

        plot_dynamics(acceleration_to_earL, acceleration_to_earR, filtered_exceeding_frames, filtered_exceeding_frames, output_folder, filename, save_figure)

        # Add the filename and filtered exceeding frames to the Excel data dictionary
        if filename in excel_data:
            excel_data[filename].extend(filtered_exceeding_frames)
        else:
            excel_data[filename] = (filtered_exceeding_frames)

# Prepare data for the DataFrame
excel_rows = [[filename, ", ".join(map(str, frames))] for filename, frames in excel_data.items()]

# Create a DataFrame from the Excel data list
df = pd.DataFrame(excel_rows, columns=["Filename", "Filtered Candidates"])

# Save the DataFrame to an Excel file
output_excel_path = os.path.join(output_folder, "filtered_exceeding_frames.xlsx")
df.to_excel(output_excel_path, index=False)

print(f"Excel file saved to {output_excel_path}")