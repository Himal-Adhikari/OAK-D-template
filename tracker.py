from ultralytics import YOLO


class Tracker:
    def __init__(self, model_path, conf_threshold=0.75):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold

    def track(self, frame):
        """
        Conducts object tracking on the specified input source using the registered trackers

        Args: The frame to use the tracker on

        Returns: A list of all the detections contained as a tuple of (data.xyxy, confidence) where
        data.xyxy = (x1, y1, x2, y2) of the bounding boxes
        """
        results = self.model.track(
            source=frame,
            stream=True,
            tracker="bytetrack.yaml",
            persist=True,
        )

        detections = []
        if results is None:
            return []
        for r in results:
            if r.boxes is None:
                continue
            for box in r.boxes:
                data = box.cpu().numpy()
                if data.conf < self.conf_threshold:
                    continue
                detections.append((data.xyxy, float(data.conf)))
