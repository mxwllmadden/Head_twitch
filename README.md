# Head Twitch Response tracking
![Overview](overview.png)

# How to use

1. Save all mp4 format videos into the 'Videos' folder.
2. Run 'video_to_h5.py'.
    - Make sure to update the paths according to your local settings.
3. Open the Conda Prompt and activate the Sleap environment.
4. Run 'track_list_1.bat' in the Conda Prompt. This will track the MP4 videos.
5. Run 'video_to_h5.py' one more time.
6. Run 'convert_list_1.bat' in the Conda Prompt. This will convert the tracking data to H5 format.
7. Run 'head_twitch.py'. This will generate an Excel file containing head twitch response candidates.
    - Make sure to update the paths according to your local settings.
    - Set the 'save_figure' option to True if you want to save the figures.
