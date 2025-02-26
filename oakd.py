import depthai as dai
from ultralytics.data.augment import Tuple
from tracker import Tracker
import cv2 as cv
import numpy as np
from ultralytics import YOLO


class OAKD:
    def __init__(
        self,
        model_path: None | str = None,
        preview_size: tuple[int, int] = (600, 600),
        fps=60,
        conf_threshold=0.75,
    ):
        width, height = preview_size
        self.timeout = 1.0 / 1000.0
        pipeline = dai.Pipeline()

        # Define sources and outputs
        monoLeft = pipeline.create(dai.node.MonoCamera)
        monoRight = pipeline.create(dai.node.MonoCamera)
        self.stereo = pipeline.create(dai.node.StereoDepth)
        camRgb = pipeline.create(dai.node.ColorCamera)

        # Properties
        monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
        monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
        monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)
        monoLeft.setFps(fps)
        monoRight.setFps(fps)

        self.stereo.initialConfig.setConfidenceThreshold(255)
        self.stereo.setLeftRightCheck(True)
        self.stereo.setSubpixel(False)

        # Linking
        monoLeft.out.link(self.stereo.left)
        monoRight.out.link(self.stereo.right)

        xoutDepth = pipeline.create(dai.node.XLinkOut)
        xoutDepth.setStreamName("depth")
        self.stereo.depth.link(xoutDepth.input)

        xoutDisparity = pipeline.create(dai.node.XLinkOut)
        xoutDisparity.setStreamName("disp")
        self.stereo.disparity.link(xoutDisparity.input)

        xoutRgb = pipeline.create(dai.node.XLinkOut)
        xoutRgb.setStreamName("rgb")
        camRgb.setPreviewSize(width, height)
        camRgb.setInterleaved(False)
        camRgb.setFps(fps)
        camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.RGB)
        camRgb.preview.link(xoutRgb.input)

        self.device = dai.Device(pipeline, maxUsbSpeed=dai.UsbSpeed.SUPER_PLUS)

        # Print OAK-D camera properties
        print("MxId:", self.device.getDeviceInfo().getMxId())
        print("USB speed:", self.device.getUsbSpeed())
        print("Connected cameras:", self.device.getConnectedCameras())

        # Start pipeline
        self.device.startPipeline()

        self.depthQ = self.device.getOutputQueue(
            name="depth", maxSize=4, blocking=False
        )
        self.dispQ = self.device.getOutputQueue(name="disp", maxSize=4, blocking=False)
        self.rgbQueue = self.device.getOutputQueue(
            name="rgb", maxSize=4, blocking=False
        )

        # Initializing tracker
        if model_path is not None:
            self.tracker = Tracker(model_path, conf_threshold)

    def get_frame(self):
        """Gets the rgb frame, disparity frame and depth data

        Returns: A tuple containing (rgbFrame, dispFrame, depthData)
        Any of the data members of the tuple can be None

        """
        inRgb = self.rgbQueue.get()
        inDisp = self.dispQ.get()
        self.depth_data = self.depthQ.get()
        (rgbFrame, dispFrame) = (None, None)
        if inRgb:
            self.curr_rgbFrame = inRgb.getCvFrame()
            rgbFrame = self.curr_rgbFrame
        if inDisp:
            self.curr_dispFrame = inDisp.getCvFrame()
            self.curr_dispFrame = (
                self.curr_dispFrame
                * (255 / self.stereo.initialConfig.getMaxDisparity())
            ).astype(np.uint8)
            dispFrame = self.curr_dispFrame
        return (rgbFrame, dispFrame, self.depth_data)

    def display_curr_frame(self, rgb_window: str, disp_window: str):
        """Displays the current rgb frame and disparity frame

        This function should be called after the get_frame function to
        display the latest frames

        Args: The window names of the rgb frame and disparity frame
        in that order
        """
        cv.imshow(rgb_window, self.curr_rgbFrame)
        cv.imshow(disp_window, self.curr_dispFrame)

    def track(self):
        """This function must be called only after calling the get_frame function on every
        iteration of the while loop


        Returns: A list of all the detections contained as a tuple of (data.xyxy, confidence) where
        data.xyxy = (x1, y1, x2, y2) of the bounding boxes

        """
        return self.tracker.track(self.curr_rgbFrame)

    def add_bbox_to_frame(self, x1: int, y1: int, x2: int, y2: int):
        """This function adds the bounding box to the rgb frame

        Call the display_curr_frame function after calling this to
        actually display the frame

        """
        cv.rectangle(self.curr_rgbFrame, (x1, y1), (x2, y2), (123, 222, 70), 2)
