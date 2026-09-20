import cv2
import numpy as np
print(cv2.__version__)

rng = np.random.default_rng(seed=42)

img = cv2.imread("warmup_photo.jpg")
print(f'Image Shape: {img.shape}')
print(f'Image data type: {type(img)}')

video = cv2.VideoCapture("Minecraft_stitch_test.mp4")

# This finds the total number of frames, and selects a random frame number to save
if video.isOpened():
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_num = rng.integers(1, total_frames)


frame_index = 0
while video.isOpened():
    flag, frame = video.read()

    if not flag:
        print("Error: Can't receive frame (stream end?). Exiting...")
        break

    # if frame_index == frame_num:
    #     filename = f'frame_{frame_num}.png'
    #     cv2.imwrite(filename, frame)
    #     print(f'Saved {filename}')

    # Save every 30th frame as a .png
    if frame_index % 30 == 0:
        filename = f'frames/frame_{frame_index}.png'
        cv2.imwrite(filename, frame)
        print(f'Saved {filename}')

         # Display the resulting frame
    cv2.imshow('Video Feed', frame)

    # Stop the loop if the 'q' key is pressed
    if cv2.waitKey(1) == ord('q'):
        break

    frame_index += 1

# Always release the resource and destroy windows when done
video.release()
cv2.destroyAllWindows()
