# Blob Detection Project

Setup:  pip install opencv-python scikit-image numpy
Put the Drive images in an `images/` folder next to these scripts.

| Script | Task |
|---|---|
| part1_simple_blob.py | SimpleBlobDetector on polka dots, size + color filtering |
| challenge1_other_methods.py | LoG / DoG / DoH / contour comparison grid |
| challenge2_classify.py | Cone / cube / ring classification with contours + boxes |
| hsv_tuner.py | Slider tool to find HSV color ranges |

Run order:
  python part1_simple_blob.py --images images --filter dot
  python part1_simple_blob.py --images images --filter dot --color red --min-area 200
  python challenge1_other_methods.py --images images --filter dot
  python hsv_tuner.py images/<objects file>
  python challenge2_classify.py --images images --filter object --debug

Results land in output/. Adjust --filter to match your filenames.
