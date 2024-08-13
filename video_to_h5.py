import os

# Configuration
# Path to TRIAL VIDEOS
video_path = r'C:\Users\Wolff_Lab\Head_twitch\Videos'  # CHANGE!
files = [f for f in os.listdir(video_path) if f.endswith('.mp4')]

# Path to MODEL
model = r'C:\Users\Wolff_Lab\Head_twitch\Mice Master N\models\240716_170449.single_instance.n=1996\training_config.json'  # CHANGE!

# Path to TRAJECTORY STORAGE
output_path = r'C:\Users\Wolff_Lab\Head_twitch\transform_result'  # CHANGE!

# Path to SLEAP ENVIRONMENT on your PC
command = r'C:\Users\Wolff_Lab\anaconda3\envs\sleap\Scripts\sleap-track'  # CHANGE!

# Path and name of the created BATCH FILE with all the tracking commands
batch_file_path = r'C:\Users\Wolff_Lab\Head_twitch\track_list_1.bat'  # CHANGE!

# Create batch file for SLEAP tracking
with open(batch_file_path, 'w') as batch_file:
    output_extension = '.predictions.slp'
    for file_name in files:
        input_file = os.path.join(video_path, file_name)
        output_file = os.path.join(output_path, file_name + output_extension)
        command_line = f'{command} "{input_file}" -m "{model}" --tracking.tracker none -o "{output_file}" --verbosity json --no-empty-frames'
        batch_file.write(f'{command_line}\n')

print(f'Batch file for tracking created at: {batch_file_path}')

# Configuration for SLEAP conversion
# Path to TRAJECTORIES to convert
traj_path = r'C:\Users\Wolff_Lab\Head_twitch\transform_result'  # CHANGE!
files = [f for f in os.listdir(traj_path) if f.endswith('.predictions.slp')]

# Path to H5 FILE STORAGE
h5_output_path = r'C:\Users\Wolff_Lab\Head_twitch\h5_result'  # CHANGE!

# Path to SLEAP ENVIRONMENT on your PC
convert_command = r'C:\Users\Wolff_Lab\anaconda3\envs\sleap\Scripts\sleap-convert --format analysis -o '  # CHANGE!

# Path and name of the created BATCH FILE with all the conversion commands
convert_batch_file_path = r'C:\Users\Wolff_Lab\Head_twitch\convert_list_1.bat'  # CHANGE!

# Create batch file for SLEAP conversion
with open(convert_batch_file_path, 'w') as convert_batch_file:
    output_extension = '.h5'
    for file_name in files:
        input_file = os.path.join(traj_path, file_name)
        output_file = os.path.join(h5_output_path, file_name.replace('.predictions.slp', output_extension))
        convert_command_line = f'{convert_command} "{output_file}" "{input_file}"'
        convert_batch_file.write(f'{convert_command_line}\n')

print(f'Batch file for conversion created at: {convert_batch_file_path}')