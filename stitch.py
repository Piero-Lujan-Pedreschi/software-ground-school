import cv2
import numpy as np

from stitch_utils import find_transform_and_inliers

paths = []
video = cv2.VideoCapture("Minecraft_stitch_test.mp4")

frame_index = 0
while video.isOpened():
    flag, frame = video.read()

    if not flag:
        print("Error: Can't receive frame (stream end?). Exiting...")
        break

    # Save every 30th frame as a .png
    if frame_index % 30 == 0:
        filename = f'frames/frame_{frame_index}.png'
        cv2.imwrite(filename, frame)
        paths.append(filename)
        print(f'Saved {filename}')

    frame_index += 1

frames = []
color_frames = []
for path in paths:
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    img_color = cv2.imread(path) 
    if img is None or img_color is None:
        raise SystemExit(f'Could not load {path}')
    frames.append(img)
    color_frames.append(img_color)  

h, w = frames[0].shape[:2]
corners = np.array([[0, w, w, 0],
                    [0, 0, h, h],
                    [1, 1, 1, 1]], dtype=np.float64)

transforms = [] 

for i in range(len(frames) - 1):
    gray_1 = frames[i]
    gray_2 = frames[i + 1]
    if gray_1 is None or gray_2 is None:
        raise SystemExit('Could not load a frame - check the file paths')

    # M maps the newer frame (gray_2) into the older frame's (gray_1) coordinates
    M, inlier_count = find_transform_and_inliers(gray_1, gray_2)
    ty = M[1, 2]

    if M is None:
        raise SystemExit('Alignment failed for this pair')

    transforms.append(M)

    print(f'Transformation matrix:\n{M}')
    print(f'Inliers: {inlier_count}')
    print(f'ty: {M[1, 2]}\n')


sum_ty = 0
H = np.eye(3)
H_list = [H.copy()]

for M in transforms:
    ty = M[1, 2]
    T = np.vstack([M, [0, 0, 1]])
    H = H @ T
    H_list.append(H.copy())
    sum_ty += ty

print(f'H : {H[1, 2]}')
print(f'sum_of_ty : {sum_ty}')

all_corners = []

for H_k in H_list:
    moved = H_k @ corners              # (3x3) @ (3x4) -> shape (3, 4)
    moved_xy = moved[:2]             # keep only the x and y rows -> shape (2, 4)
    all_corners.append(moved_xy)

all_xy = np.hstack(all_corners)
print(f'all_xy shape: {all_xy.shape}')

min_x = np.floor(all_xy[0].min())
max_x = np.ceil(all_xy[0].max())
min_y = np.floor(all_xy[1].min())
max_y = np.ceil(all_xy[1].max())

width = int(max_x - min_x)
height = int(max_y - min_y)

print(f'min_x={min_x}, max_x={max_x}, min_y={min_y}, max_y={max_y}')
print(f'canvas width={width}, height={height}')

offset = np.array([[1, 0, -min_x],
                   [0, 1, -min_y],
                   [0, 0,    1  ]], dtype=np.float64)

checked = []
for H_k in H_list:
    moved = (offset @ H_k) @ corners
    checked.append(moved[:2])

checked_xy = np.hstack(checked)
print(f'after offset: x in [{checked_xy[0].min()}, {checked_xy[0].max()}]')
print(f'after offset: y in [{checked_xy[1].min()}, {checked_xy[1].max()}]')

canvas = np.zeros((height, width, 3), dtype=np.uint8)

for k, img in enumerate(color_frames):
    full_transform = offset @ H_list[k]
    M_2x3 = full_transform[:2]
    warped = cv2.warpAffine(img, M_2x3, (width, height))

    frame_h, frame_w = img.shape[:2]
    white = np.ones((frame_h, frame_w), dtype=np.uint8) * 255
    mask = cv2.warpAffine(white, M_2x3, (width, height))
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.erode(mask, kernel)

    canvas[mask > 0] = warped[mask > 0]

cv2.imwrite('map.png', canvas)