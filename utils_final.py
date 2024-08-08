import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d

# Kalman filter implementation to smooth noisy data
def kalman_filter(data, R=1e-5, Q=1e-5):
    length = len(data)
    x = np.zeros(length)  # Initialize state vector
    P = np.zeros(length)  # Initialize error covariance
    K = np.zeros(length)  # Initialize Kalman gain

    # Find the first non-NaN value
    idx = 0
    while np.isnan(data[idx]):
        idx += 1

    # Initialize with the first valid data point
    x[0] = data[idx]
    P[0] = 1.0

    for k in range(1, length):
        if np.isnan(data[k]):
            # Skip NaN values, no update
            x[k] = x[k - 1]
            P[k] = P[k - 1]
        else:
            # Prediction step
            x_predict = x[k - 1]
            P_predict = P[k - 1] + Q

            # Update step
            K[k] = P_predict / (P_predict + R)
            x[k] = x_predict + K[k] * (data[k] - x_predict)
            P[k] = (1 - K[k]) * P_predict

    return x

# Function to interpolate NaN values
def interpolate_nans(data):
    nans, x = np.isnan(data), lambda z: z.nonzero()[0]
    data[nans] = np.interp(x(nans), x(~nans), data[~nans])
    return data

def remove_exceeding_frames_with_nan(data, exceeding_frames, window=1):
    """
    Remove frames from exceeding_frames if they have NaN values in adjacent frames.

    Parameters:
    - data: The dataset (e.g., acceleration, velocity) to check for NaNs.
    - exceeding_frames: List of frames where the threshold is exceeded.
    - window: Number of frames before and after the exceeding frame to check for NaNs.

    Returns:
    - filtered_exceeding_frames: List of frames with NaN values adjacent removed.
    """
    filtered_exceeding_frames = []

    # Convert to a numpy array for easier indexing
    data = np.array(data)
    
    # Create a mask for NaNs in the dataset
    nan_mask = np.isnan(data)

    for frame in exceeding_frames:
        # Define the range of adjacent frames to check
        start_idx = max(frame - window, 0)
        end_idx = min(frame + window + 1, len(data))
        
        # Check if any adjacent frame is NaN
        if not np.any(nan_mask[start_idx:end_idx]):
            filtered_exceeding_frames.append(frame)

    return np.array(filtered_exceeding_frames)

# Calculate dynamics (distance, velocity, acceleration) for body parts
def calculate_dynamics(nose_loc, tail_loc, back_loc, earL_loc, earR_loc):
    length = len(tail_loc)

    nose_loc_filtered_x = nose_loc[:, 0, 0]
    nose_loc_filtered_y = nose_loc[:, 1, 0]
        

    # Arrays to store distances and velocities
    dist_body_to_earL, dist_body_to_earR = [], []
    velocity_to_earL, velocity_to_earR = [], []
    acceleration_to_earL, acceleration_to_earR = [], []

    # Compute distances for each frame
    for index in range(length):  
        # nose = nose_loc[index, :, 0]
        tail = tail_loc[index, :, 0]
        back = back_loc[index, :, 0]
        earL = earL_loc[index, :, 0]
        earR = earR_loc[index, :, 0]

        # Define the body vector from nose to tail
        body_vector = np.array([(tail[0] + back[0]) / 2 - nose_loc_filtered_x[index], (tail[1] + back[1]) / 2 - nose_loc_filtered_y[index]])

        # Define the ear vectors from nose to each ear
        earL_vector = np.array([earL[0] - nose_loc_filtered_x[index], earL[1] - nose_loc_filtered_y[index]])
        earR_vector = np.array([earR[0] - nose_loc_filtered_x[index], earR[1] - nose_loc_filtered_y[index]])

        # Calculate projections of ear vectors onto the body vector
        proj_earL = (np.dot(earL_vector, body_vector) / np.dot(body_vector, body_vector)) * body_vector
        proj_earR = (np.dot(earR_vector, body_vector) / np.dot(body_vector, body_vector)) * body_vector

        # Calculate perpendicular distances from the ears to the body vector
        dist_to_earL = np.linalg.norm(earL_vector - proj_earL)
        dist_to_earR = np.linalg.norm(earR_vector - proj_earR)

        dist_body_to_earL.append(dist_to_earL)
        dist_body_to_earR.append(dist_to_earR)

    # Compute velocities (approximated as differences in distance)
    for i in range(1, length):
        vel_to_earL = (dist_body_to_earL[i] - dist_body_to_earL[i - 1]) / 1  # Assuming constant interval of 1 frame
        vel_to_earR = (dist_body_to_earR[i] - dist_body_to_earR[i - 1]) / 1  # Assuming constant interval of 1 frame
        velocity_to_earL.append(vel_to_earL)
        velocity_to_earR.append(vel_to_earR)

    # Compute accelerations (approximated as differences in velocity)
    for i in range(1, len(velocity_to_earL)):
        acc_to_earL = (velocity_to_earL[i] - velocity_to_earL[i - 1]) / 1  # Assuming constant interval of 1 frame
        acc_to_earR = (velocity_to_earR[i] - velocity_to_earR[i - 1]) / 1  # Assuming constant interval of 1 frame
        acceleration_to_earL.append(acc_to_earL)
        acceleration_to_earR.append(acc_to_earR)

    return (np.array(dist_body_to_earL), np.array(dist_body_to_earR), 
            np.array(velocity_to_earL), np.array(velocity_to_earR), 
            np.array(acceleration_to_earL), np.array(acceleration_to_earR))

# Calculate combined dynamics for both ears
def calculate_combined_dynamics(dist_body_to_earL, dist_body_to_earR):
    length = len(dist_body_to_earL)

    # Arrays to store combined distances, velocities, and accelerations
    dist = dist_body_to_earL + dist_body_to_earR
    vel, acc = [], []

    # Compute velocities (approximated as differences in distance)
    for i in range(1, length):
        velocity = (dist[i] - dist[i - 1]) / 1  # Assuming constant interval of 1 frame
        vel.append(velocity)

    # Compute accelerations (approximated as differences in velocity)
    for i in range(1, len(vel)):
        acceleration = (vel[i] - vel[i - 1]) / 1  # Assuming constant interval of 1 frame
        acc.append(acceleration)

    return dist, np.array(vel), np.array(acc)

# Plot dynamics with highlighted exceeding frames
def plot_dynamics(acceleration_to_earL, acceleration_to_earR, exceeding_frames_L, exceeding_frames_R, folder_name, filename, save_figure=False):

    threshold2 = 7

    # Sum of absolute accelerations for both ears
    acc_sum = np.abs(acceleration_to_earR) + np.abs(acceleration_to_earL)
    exceeding_frames_acc_sum = [i + 2 for i in range(len(acc_sum)) if abs(acc_sum[i]) > threshold2]
    
    # Filter exceeding_frames_acc_sum to include only frames present in exceeding_frames_L
    filtered_exceeding_frames_acc_sum = [frame for frame in exceeding_frames_acc_sum if frame in exceeding_frames_L]

    plt.figure(figsize=(10, 15))
    
    # Plot accelerations to EarL
    plt.subplot(3, 1, 1)
    plt.plot(acceleration_to_earL, label='Acceleration to EarL', color='blue')
    plt.ylim(-10, 10)
    plt.xlabel('Frame Index')
    plt.ylabel('Acceleration')
    plt.title(f'Acceleration to EarL Points, @{filtered_exceeding_frames_acc_sum}')
    plt.legend()
    plt.grid(True)

    for frame in exceeding_frames_L:
            # plt.axvline(x=frame, color='red', linestyle='--')
            top_y = plt.ylim()[1]  # Get the maximum y-value
            plt.plot(frame, top_y, 'ro')  # Draw a blue dot at the top of the y-axis
    
    # Plot accelerations to EarR
    plt.subplot(3, 1, 2)
    plt.plot(acceleration_to_earR, label='Acceleration to EarR', color='blue')
    plt.ylim(-10, 10)
    plt.xlabel('Frame Index')
    plt.ylabel('Acceleration')
    plt.title(f'Acceleration to EarR Points, @{filtered_exceeding_frames_acc_sum}')
    plt.legend()
    plt.grid(True)

    for frame in exceeding_frames_R:
        if frame - 5 >= 0:
            # plt.axvline(x=frame, color='red', linestyle='--')
            top_y = plt.ylim()[1]  # Get the maximum y-value
            plt.plot(frame, top_y, 'ro')  # Draw a blue dot at the top of the y-axis
    # Plot combined acceleration sum
    plt.subplot(3, 1, 3)
    plt.plot(acc_sum, label='Acc post_acc_Sum', color='green')
    plt.ylim(0, 20)
    plt.xlabel('Frame Index')
    plt.ylabel('Acceleration')
    plt.title(f"Acceleration of post_acc_Sum @{filtered_exceeding_frames_acc_sum}")
    plt.legend()
    plt.grid(True)

    for frame in filtered_exceeding_frames_acc_sum:
        if frame - 5 >= 0:
            # plt.axvline(x=frame, color='red', linestyle='--')
            top_y = plt.ylim()[1]  # Get the maximum y-value
            plt.plot(frame, top_y, 'ro')  # Draw a blue dot at the top of the y-axis

    plt.suptitle(f"{filename} Candidates", fontsize=16, y=1.02)
    plt.tight_layout()
    plt.subplots_adjust(hspace=0.4)

    if save_figure:
       plt.savefig(os.path.join(folder_name, f"{filename}_plot.png"), bbox_inches='tight')
    plt.show()

# Calculate distances from the nose to the line defined by the ears for exceeding frames
def calculate_distances_for_exceeding_frames(nose_loc, earL_loc, earR_loc, array, threshold):
    length = len(nose_loc)
    dist_nose_to_line = []

    # Calculate the perpendicular distance from the nose to the line formed by the ears
    for index in range(length):
        nose = nose_loc[index, :, 0]
        earL = earL_loc[index, :, 0]
        earR = earR_loc[index, :, 0]

        # Vector from left ear to right ear
        AB = earR - earL
        # Vector from left ear to nose
        AP = nose - earL

        # Cross product gives area of parallelogram formed by AB and AP
        cross_product = np.cross(AB, AP)
        # Normalize cross product to get height (distance) of parallelogram
        norm_cross_product = np.linalg.norm(cross_product)
        norm_AB = np.linalg.norm(AB)

        # Distance from nose to line (height of parallelogram)
        distance = norm_cross_product / norm_AB
        dist_nose_to_line.append(distance)

    dist_nose_to_line = np.array(dist_nose_to_line)
    array = np.asarray(array).astype(int)
    
    # Filter frames where distance exceeds threshold
    filtered_exceeding_frames = [frame for frame in array if 0 <= frame < length and dist_nose_to_line[frame] >= threshold]

    return dist_nose_to_line, np.sort(np.array(filtered_exceeding_frames))

# Calculate the angle between two vectors
def angle_between_vectors(v1, v2):
    # Calculate the angle in degrees between two vectors using arctan2 for better numerical stability
    angle = np.arctan2(np.linalg.norm(np.cross(v1, v2)), np.dot(v1, v2))
    return np.degrees(angle)

# Calculate angles formed by vectors from nose to back and back to tail
def calculate_angles(nose_loc, back_loc, tail_loc):
    angles = []
    for i in range(len(nose_loc)):
        nose = nose_loc[i]
        back = back_loc[i]
        tail = tail_loc[i]

        # Vectors from nose to back and back to tail
        mid_body = nose - back
        body_to_tail = tail - back
        
        # Flatten the vectors to 1D
        mid_body = mid_body.flatten()
        body_to_tail = body_to_tail.flatten()
        
        # Calculate angle between vectors
        angle = angle_between_vectors(mid_body, body_to_tail)
        angles.append(angle)
    
    return angles

# Calculate angles for frames exceeding certain criteria
def calculate_angles_for_exceeding_frames(angles, combined_exceeding_frames):
    # Ensure that combined_exceeding_frames is a list of integers
    combined_exceeding_frames = [int(frame) for frame in combined_exceeding_frames]

    # Filter angles that are below 100 degrees or NaN for the exceeding frames
    angles_below_90 = [
        [frame, angles[frame]] for frame in combined_exceeding_frames
        if frame < len(angles) and (np.isnan(angles[frame]) or angles[frame] < 100)
    ]
    
    angles_below_90_array = np.array(angles_below_90)
    
    # Exclude angles below 90 degrees
    new_angles = [
        [int(frame), angles[int(frame)]] for frame in combined_exceeding_frames
        if int(frame) < len(angles) and all(int(frame) != item[0] for item in angles_below_90_array)
    ]
    
    # Convert the list to a NumPy array with integer indices
    new_angles_array = np.array(new_angles, dtype=int)
    
    return angles_below_90_array, new_angles_array

# Fill missing values in data using interpolation
def fill_missing(Y, kind="linear"):
    initial_shape = Y.shape
    Y = Y.reshape((initial_shape[0], -1))

    # Interpolate missing values for each dimension
    for i in range(Y.shape[-1]):
        y = Y[:, i]
        x = np.flatnonzero(~np.isnan(y))
        f = interp1d(x, y[x], kind=kind, fill_value=np.nan, bounds_error=False)
        xq = np.flatnonzero(np.isnan(y))
        y[xq] = f(xq)
        mask = np.isnan(y)
        y[mask] = np.interp(np.flatnonzero(mask), np.flatnonzero(~mask), y[~mask])
        Y[:, i] = y

    Y = Y.reshape(initial_shape)
    return Y