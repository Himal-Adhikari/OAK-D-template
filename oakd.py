import depthai as dai
from tracker import Tracker


class OAKD:
    def __init__(
        self, model="best.pt", preview_size=(640, 480), fps=60, conf_threshold=0.75
    ):
        self.timeout = 60.0 / 1000.0
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

        camRgb.setPreviewSize(preview_size)
        camRgb.setInterleaved(False)
        camRgb.setFps(fps)

        # Linking
        monoLeft.out.link(self.stereo.left)
        monoRight.out.link(self.stereo.right)

        xoutDepth = pipeline.create(dai.node.XLinkOut)
        xoutDepth.setStreamName("depth")
        self.stereo.disparity.link(xoutDepth.input)

        xoutDisparity = pipeline.create(dai.node.XLinkOut)
        xoutDisparity.setStreamName("disp")
        self.stereo.disparity.link(xoutDisparity.input)

        xoutRgb = pipeline.create(dai.node.XLinkOut)
        xoutRgb.setStreamName("rgb")
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
        self.tracker = Tracker(model, conf_threshold)

    def get_frame(self):
        inRgb = self.rgbQueue.get()
        inDisp = self.dispQ.get()
        depthData = self.depthQ.get()
        (rgbFrame, dispFrame) = (None, None)
        if inRgb:
            self.curr_rgbFrame = inRgb.CvFrame()
            rgbFrame = self.curr_rgbFrame
        if inDisp:
            self.curr_dispFrame = inDisp.getCvFrame()
            dispFrame = self.curr_dispFrame
        return (rgbFrame, dispFrame, depthData)

    def track(self):
        """This function must be called only after calling the get_frame function on every
        iteration of the while loop


        Returns: A list of all the detections contained as a tuple of (data.xyxy, confidence) where
        data.xyxy = (x1, y1, x2, y2) of the bounding boxes

        """
        return self.tracker.track(self.curr_rgbFrame)
