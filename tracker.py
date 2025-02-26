from ultralytics import YOLO
import cv2 as cv


class Tracker:
    def __init__(self, model_path, conf_threshold=0.75):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def track(self, frame):
        """
        Conducts object tracking on the specified input source using the registered trackers

        Args: The frame to use the tracker on

        Returns: A list of all the detections contained as a tuple of ((x1, y1, x2, y2), confidence)
        where data.xyxy = (x1, y1, x2, y2) of the bounding boxes
        """
        results = self.model.track(
            source=frame,
            stream=True,
            tracker="bytetrack.yaml",
            persist=True,
        )

        detections = list()
        if results is None:
            return list()
        for r in results:
            if r.boxes is None:
                continue
            detection = r.boxes.cpu().numpy()
            for box in detection:
                if box.conf < self.conf_threshold:
                    continue
                x1, y1, x2, y2 = box.xyxy[0]
                detections.append(((x1, y1, x2, y2), box.conf))
        return detections
