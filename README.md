# Sample main.py
``` python
from oakd import OAKD
import cv2 as cv

camera = OAKD("best.pt")

while True:
    (rgbFrame, dispFrame, depthData) = camera.get_frame()
    tracklets = camera.track()
    if tracklets is not None:
        for (bbox, conf) in tracklets:
            x1, y1, x2, y2 = bbox
            (coordinate, centroid) = camera.get_coordinate(
                (x1 + x2) / 2, (y1 + y2) / 2)
            print(centroid)
            print(coordinate)
            camera.add_bbox_to_frame(x1, y1, x2, y2)
    camera.display_curr_frame("rgb", "disp")
    if cv.waitKey(1) == ord('q'):
        break

cv.destroyAllWindows()
```
