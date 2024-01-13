import { useState, useRef, useEffect } from "react";

import cv from "@techstark/opencv-js";

import { Tensor, InferenceSession } from "onnxruntime-web";

import Loader from "./components/loader";

import { detectImage } from "./utils/detect";
import { download } from "./utils/download";

import { Typography, Slider, Box, Button, Stack } from "@mui/material";
import FileOpenIcon from "@mui/icons-material/FileOpen";
import CloseIcon from "@mui/icons-material/Close";
import VideocamIcon from "@mui/icons-material/Videocam";

import "./App.css";

const App = () => {
    const videoRef = useRef(null);
    const inputVideo = useRef(null);
    const canvas1Ref = useRef(null);
    const canvasRef = useRef(null);

    const [session, setSession] = useState(null);
    const [video, setVideo] = useState(null);
    const [loading, setLoading] = useState({
        text: "Loading OpenCV.js",
        progress: null,
    });

    // Configs
    const modelName = "drone_detection_v2.onnx";
    const [iouThreshold, setIouThreshold] = useState(0.5);
    const [scoreThreshold, setScoreThreshold] = useState(0.5);
    const modelInputShape = [1, 3, 640, 640];
    const topk = 100;

    useEffect(() => {
        const context = canvas1Ref.current.getContext("2d");
        context.clearRect(0, 0, context.canvas.width, context.canvas.height);
        if (video) {
            const interval = setInterval(() => {
                // Draw video frame on canvas
                context.drawImage(
                    videoRef.current,
                    0,
                    0,
                    canvas1Ref.current.width,
                    canvas1Ref.current.height,
                );

                // Run object detection on the canvas image
                detectImage(
                    canvas1Ref.current,
                    canvasRef.current,
                    session,
                    topk,
                    iouThreshold,
                    scoreThreshold,
                    modelInputShape,
                );
            }, 50);
            // Clear the interval on component unmount
            return () => clearInterval(interval);
        }
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, [session, video, iouThreshold, scoreThreshold]);

    // wait until opencv.js initialized
    cv["onRuntimeInitialized"] = async () => {
        const baseModelURL = `${process.env.PUBLIC_URL}/model`;

        // create session
        const arrBufNet = await download(
            `${baseModelURL}/${modelName}`, // url
            ["Loading YOLOv8 Drone Detection model", setLoading], // logger
        );
        const yolov8 = await InferenceSession.create(arrBufNet);
        const arrBufNMS = await download(
            `${baseModelURL}/nms-yolov8.onnx`, // url
            ["Loading NMS model", setLoading], // logger
        );
        const nms = await InferenceSession.create(arrBufNMS);

        // warmup main model
        setLoading({ text: "Warming up model...", progress: null });
        const tensor = new Tensor(
            "float32",
            new Float32Array(modelInputShape.reduce((a, b) => a * b)),
            modelInputShape,
        );

        await yolov8.run({ images: tensor });
        setSession({ net: yolov8, nms: nms });
        setLoading(null);
    };

    const openWebCam = () => {
        const mediaDevices = navigator.mediaDevices;

        mediaDevices
            .getUserMedia({
                video: true,
                audio: false,
            })
            .then((stream) => {
                videoRef.current.srcObject = stream;
                setVideo("stream");
            })
            .catch(alert);
    };

    const stopWebCam = () => {
        const tracks = videoRef.current.srcObject.getTracks();
        tracks.forEach((track) => {
            track.stop();
        });
        videoRef.current.srcObject = null;
        setVideo(null);
    };

    return (
        <div>
            {loading && (
                <Loader>
                    {loading.progress
                        ? `${loading.text} - ${loading.progress}%`
                        : loading.text}
                </Loader>
            )}
            <div className="header">
                <Typography variant="h3" style={{ margin: 10 }}>
                    Drone Detection App
                </Typography>
            </div>
            <input
                type="file"
                ref={inputVideo}
                accept="video/*"
                style={{ display: "none" }}
                onChange={(e) => {
                    // handle next image to detect
                    if (video) {
                        URL.revokeObjectURL(video);
                        setVideo(null);
                    }
                    const url = URL.createObjectURL(e.target.files[0]); // create image url
                    videoRef.current.src = url; // set image source
                    setVideo(url);
                }}
            />
            <div className="sliders" style={{ margin: 10 }}>
                <Box sx={{ width: 300 }}>
                    <Typography id="input-slider" gutterBottom>
                        IouThreshold
                    </Typography>
                    <Slider
                        defaultValue={0.5}
                        valueLabelDisplay="auto"
                        min={0}
                        max={1}
                        step={0.01}
                        onChange={(e, newVal) => {
                            setIouThreshold(newVal);
                        }}
                    />
                    <Typography id="input-slider" gutterBottom>
                        ScoreThreshold
                    </Typography>
                    <Slider
                        defaultValue={0.5}
                        valueLabelDisplay="auto"
                        min={0}
                        max={1}
                        step={0.01}
                        onChange={(e, newVal) => {
                            setScoreThreshold(newVal);
                        }}
                    />
                </Box>
            </div>
            <div className="btn-container" style={{ margin: 10 }}>
                <Stack spacing={2} direction="row">
                    <Button
                        startIcon={<VideocamIcon />}
                        variant="contained"
                        color="success"
                        onClick={openWebCam}
                    >
                        Connect to camera
                    </Button>
                    <Button
                        startIcon={<FileOpenIcon />}
                        variant="contained"
                        onClick={() => {
                            if (video === "stream") {
                                stopWebCam();
                            }
                            inputVideo.current.click();
                        }}
                    >
                        Open local video
                    </Button>
                    {video && (
                        /* show close btn when there is image */
                        <Button
                            startIcon={<CloseIcon />}
                            variant="contained"
                            color="error"
                            onClick={() => {
                                if (video === "stream") {
                                    stopWebCam();
                                }
                                inputVideo.current.value = "";
                                videoRef.current.src = "#";
                                setVideo(null);
                                const ctx = canvasRef.current.getContext("2d");
                                const ctx1 =
                                    canvas1Ref.current.getContext("2d");
                                ctx.clearRect(
                                    0,
                                    0,
                                    ctx.canvas.width,
                                    ctx.canvas.height,
                                ); // clean canvas
                                ctx1.clearRect(
                                    0,
                                    0,
                                    ctx1.canvas.width,
                                    ctx1.canvas.height,
                                ); // clean canvas
                                URL.revokeObjectURL(video);
                            }}
                        >
                            Close video
                        </Button>
                    )}
                </Stack>
            </div>
            <div className="content">
                <Box width={640}>
                    <Typography variant="h5" gutterBottom textAlign={"center"}>
                        Original video
                    </Typography>
                    <video autoPlay muted ref={videoRef} loop></video>
                </Box>
                <Box width={640} ml={5}>
                    <Typography variant="h5" gutterBottom textAlign={"center"}>
                        Processed video
                    </Typography>
                    <canvas
                        style={{ backgroundColor: "#000" }}
                        ref={canvas1Ref}
                        width={640}
                        height={360}
                    />
                    <canvas ref={canvasRef} width={640} height={640} />
                </Box>
            </div>
        </div>
    );
};

export default App;

