import sys
import cv2
import numpy as np
from common import resize_to_width

img = resize_to_width(cv2.imread(sys.argv[1]), 800)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.namedWindow("tuner")
for n, mx, v in [("H lo", 179, 0), ("H hi", 179, 179), ("S lo", 255, 0),
                 ("S hi", 255, 255), ("V lo", 255, 0), ("V hi", 255, 255)]:
    cv2.createTrackbar(n, "tuner", v, mx, lambda _: None)

while True:
    g = lambda n: cv2.getTrackbarPos(n, "tuner")
    lo = (g("H lo"), g("S lo"), g("V lo"))
    hi = (g("H hi"), g("S hi"), g("V hi"))
    mask = cv2.inRange(hsv, np.array(lo), np.array(hi))
    cv2.imshow("tuner", np.hstack([img, cv2.bitwise_and(img, img, mask=mask)]))
    if cv2.waitKey(30) & 0xFF == ord("q"):
        break
print(f"[({lo}, {hi})]")
cv2.destroyAllWindows()
