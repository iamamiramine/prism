from fastapi import APIRouter, Request, WebSocket
from fastapi.responses import JSONResponse
import traceback
import numpy as np
import asyncio
import json
from typing import Dict
import base64
import websockets

from application.audio_reactive.services import generate_service
from domain.models.audio_reactive.generate_models import GenerateAudioReactiveVideoParameters
router = APIRouter()

# Store active WebRTC connections
active_connections: Dict[int, WebSocket] = {}

@router.post("/initialize_model")
def initialize_model(
        config_path: str = "../shared/configs/stable_diffusion/v1-inference.yaml",
        checkpoint_path: str = "../models/ldm/stable_diffusion_v1/model.ckpt"
    ) -> dict:
    """
    Description:
    ------------
        Initialize Model
    """
    try:
        return generate_service.initialize_model(config_path, checkpoint_path)
    except Exception as e:
        traceback.print_exc()
        return {
            "error": True,
            "message": str(e),
            "type": type(e).__name__
        }


@router.post("/generate_audio_reactive_video")
def generate_audio_reactive_video(parameters: GenerateAudioReactiveVideoParameters) -> dict:
    """
    Description:
    ------------
        Generate Audio Reactive Video

    Parameters:
    -----------
        parameters: GeneratorGeneratePromptParameters

    Returns:
    --------
    dict
        A dictionary

    """
    try:
        return {"video_path": str(generate_service.generate_audio_reactive_video(parameters))}
    except Exception as e:
        traceback.print_exc()
        return {
            "error": True,
            "message": str(e),
            "type": type(e).__name__
        }


@router.websocket("/webrtc")
async def webrtc_endpoint(websocket: WebSocket):
    client_id = None
    try:
        await websocket.accept()
        while True:
            try:
                data = await websocket.receive_json()
                msg_type = data.get('type')
                print(f"Received WebRTC message type: {msg_type}", flush=True)
                
                if not msg_type:
                    print(f"Received message without type: {data}", flush=True)
                    continue
                    
                if msg_type == "hello":
                    client_id = data["id"]
                    active_connections[client_id] = websocket
                    print(f"New client connected: {client_id}", flush=True)
                    
                elif msg_type == "video":
                    if not client_id:
                        continue
                        
                    # Convert hex string back to bytes
                    video_data = bytes.fromhex(data["data"])
                    print(f"Received video data from client {client_id}, size: {len(video_data)} bytes", flush=True)
                    
                    # Send acknowledgment
                    await websocket.send_json({
                        "type": "video_received",
                        "size": len(video_data)
                    })
                    
                elif msg_type == "sdp":
                    sdp = data.get('sdp', {})
                    sdp_type = sdp.get('type')
                    print(f"Handling SDP message type: {sdp_type}", flush=True)
                    
                    # Echo back the SDP with our response
                    await websocket.send_json({
                        "type": "sdp_response",
                        "sdp": {
                            "type": "answer",
                            "sdp": sdp.get('sdp', '')
                        }
                    })
                    
                elif msg_type == "ice":
                    ice = data.get('ice', {})
                    print(f"Handling ICE candidate: {ice.get('candidate')}", flush=True)
                    await websocket.send_json({
                        "type": "ice_response",
                        "ice": ice
                    })
                    
            except json.JSONDecodeError as e:
                print(f"Received invalid JSON: {e}", flush=True)
            except websockets.exceptions.ConnectionClosed:
                print(f"Client {client_id} connection closed", flush=True)
                break
            except Exception as e:
                print(f"Error processing message: {str(e)}", flush=True)
                traceback.print_exc()
                
    except Exception as e:
        print(f"WebRTC Error: {str(e)}", flush=True)
        traceback.print_exc()
    finally:
        if client_id and client_id in active_connections:
            del active_connections[client_id]
            print(f"Removed client {client_id} from active connections", flush=True)
