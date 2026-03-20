import os
import json
import time
from typing import Optional, Dict, Any
import supervision as sv
import numpy as np
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
import asyncio
from pathlib import Path
from loguru import logger

from fiber.logging_utils import get_logger
from miner.core.models.config import Config
from miner.core.configuration import factory_config
from miner.dependencies import get_config, verify_request, blacklist_low_stake
from sports.configs.soccer import SoccerPitchConfiguration
from miner.utils.device import get_optimal_device
from miner.utils.model_manager import ModelManager
from miner.utils.video_processor import VideoProcessor
from miner.utils.shared import miner_lock
from miner.utils.video_downloader import download_video

logger = get_logger(__name__)

CONFIG = SoccerPitchConfiguration()

# Validator-expected class IDs
BALL_CLASS_ID = 0
GOALKEEPER_CLASS_ID = 1
PLAYER_CLASS_ID = 2
REFEREE_CLASS_ID = 3

# Global model manager instance
model_manager = None

def get_model_manager(config: Config = Depends(get_config)) -> ModelManager:
    global model_manager
    if model_manager is None:
        model_manager = ModelManager(device=config.device)
        model_manager.load_all_models()
    return model_manager

async def process_soccer_video(
    video_path: str,
    model_manager: ModelManager,
) -> Dict[str, Any]:
    """Process a soccer video and return tracking data."""
    start_time = time.time()

    try:
        # Get actual video FPS so ByteTrack timing is calibrated correctly
        video_info = VideoProcessor.get_video_info(video_path)
        fps = max(video_info.fps, 1)
        logger.info(f"Video FPS: {fps}, total frames: {video_info.total_frames}")

        video_processor = VideoProcessor(
            device=model_manager.device,
            cuda_timeout=10800.0,
            mps_timeout=10800.0,
            cpu_timeout=10800.0,
        )

        if not await video_processor.ensure_video_readable(video_path):
            raise HTTPException(
                status_code=400,
                detail="Video file is not readable or corrupted",
            )

        player_model = model_manager.get_model("player")
        pitch_model = model_manager.get_model("pitch")
        ball_model = model_manager.get_model("ball")

        # Player/person tracker: 2-second buffer handles occlusions between frames
        player_tracker = sv.ByteTrack(
            track_activation_threshold=0.3,
            lost_track_buffer=fps * 2,
            minimum_matching_threshold=0.8,
            frame_rate=fps,
            minimum_consecutive_frames=1,
        )

        # Ball tracker: ball moves fast so IoU drops quickly — lower match threshold
        ball_tracker = sv.ByteTrack(
            track_activation_threshold=0.5,
            lost_track_buffer=fps,
            minimum_matching_threshold=0.7,
            frame_rate=fps,
            minimum_consecutive_frames=1,
        )

        tracking_data = {"frames": []}

        async for frame_number, frame in video_processor.stream_frames(video_path):
            # Pitch keypoints
            pitch_result = pitch_model(frame, verbose=False)[0]
            keypoints = sv.KeyPoints.from_ultralytics(pitch_result)

            # Player/goalkeeper/referee detection — exclude ball class (0) from
            # player model to avoid duplicate ball tracks
            player_result = player_model(frame, imgsz=1280, verbose=False)[0]
            player_detections = sv.Detections.from_ultralytics(player_result)
            player_detections = player_detections[player_detections.class_id > 0]
            player_detections = player_tracker.update_with_detections(player_detections)

            # Ball detection via dedicated model with its own tracker
            ball_result = ball_model(frame, imgsz=640, verbose=False)[0]
            ball_detections = sv.Detections.from_ultralytics(ball_result)
            ball_detections = ball_tracker.update_with_detections(ball_detections)

            objects = []

            if player_detections and player_detections.tracker_id is not None:
                for tracker_id, bbox, class_id in zip(
                    player_detections.tracker_id,
                    player_detections.xyxy,
                    player_detections.class_id,
                ):
                    objects.append({
                        "id": int(tracker_id),
                        "bbox": [float(x) for x in bbox],
                        "class_id": int(class_id),
                    })

            if ball_detections and ball_detections.tracker_id is not None:
                for tracker_id, bbox in zip(
                    ball_detections.tracker_id,
                    ball_detections.xyxy,
                ):
                    objects.append({
                        "id": int(tracker_id),
                        "bbox": [float(x) for x in bbox],
                        "class_id": BALL_CLASS_ID,
                    })

            frame_data = {
                "frame_number": int(frame_number),
                "keypoints": (
                    keypoints.xy[0].tolist()
                    if keypoints and keypoints.xy is not None
                    else []
                ),
                "objects": objects,
            }
            tracking_data["frames"].append(frame_data)

            if frame_number % 100 == 0:
                elapsed = time.time() - start_time
                fps_proc = frame_number / elapsed if elapsed > 0 else 0
                logger.info(
                    f"Processed {frame_number} frames in {elapsed:.1f}s "
                    f"({fps_proc:.2f} fps)"
                )

        processing_time = time.time() - start_time
        tracking_data["processing_time"] = processing_time

        total_frames = len(tracking_data["frames"])
        fps_proc = total_frames / processing_time if processing_time > 0 else 0
        logger.info(
            f"Completed processing {total_frames} frames in {processing_time:.1f}s "
            f"({fps_proc:.2f} fps) on {model_manager.device} device"
        )

        return tracking_data

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Video processing error: {str(e)}")


async def process_challenge(
    request: Request,
    config: Config = Depends(get_config),
    model_manager: ModelManager = Depends(get_model_manager),
):
    logger.info("Attempting to acquire miner lock...")
    async with miner_lock:
        logger.info("Miner lock acquired, processing challenge...")
        try:
            challenge_data = await request.json()
            challenge_id = challenge_data.get("challenge_id")
            video_url = challenge_data.get("video_url")

            logger.info(f"Received challenge data: {json.dumps(challenge_data, indent=2)}")

            if not video_url:
                raise HTTPException(status_code=400, detail="No video URL provided")

            logger.info(f"Processing challenge {challenge_id} with video {video_url}")

            video_path = await download_video(video_url)

            try:
                tracking_data = await process_soccer_video(
                    video_path,
                    model_manager,
                )

                response = {
                    "challenge_id": challenge_id,
                    "frames": tracking_data["frames"],
                    "processing_time": tracking_data["processing_time"],
                }

                logger.info(
                    f"Completed challenge {challenge_id} in "
                    f"{tracking_data['processing_time']:.2f} seconds"
                )
                return response

            finally:
                try:
                    os.unlink(video_path)
                except Exception:
                    pass

        except HTTPException:
            raise
        except Exception as e:
            logger.error(f"Error processing soccer challenge: {str(e)}")
            logger.exception("Full error traceback:")
            raise HTTPException(
                status_code=500,
                detail=f"Challenge processing error: {str(e)}",
            )
        finally:
            logger.info("Releasing miner lock...")


# Create router with dependencies
router = APIRouter()
router.add_api_route(
    "/challenge",
    process_challenge,
    tags=["soccer"],
    dependencies=[Depends(blacklist_low_stake), Depends(verify_request)],
    methods=["POST"],
)
