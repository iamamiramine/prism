from fastapi import APIRouter

from src.application.audio_reactive.services import generate_service
from src.domain.models.audio_reactive.generate_models import GenerateAudioReactiveVideoParameters, GenerateImageParameters, GenerateVideoParameters
router = APIRouter()


@router.post("/generate_image")
def generate_image(parameters: GenerateImageParameters) -> dict:
    """
    Description:
    ------------
        Train Generator

    Parameters:
    -----------
        parameters: GeneratorTrainingParameters

    Returns:
    --------
    dict
        A dictionary

    """
    return generate_service.generate_image(parameters)


@router.post("/generate_video")
def generate_video(parameters: GenerateVideoParameters) -> dict:
    """
    Description:
    ------------
        Generate Sample from Prompt

    Parameters:
    -----------
        parameters: GeneratorGeneratePromptParameters

    Returns:
    --------
    dict
        A dictionary

    """
    return generate_service.generate_video(parameters)


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
    return generate_service.generate_audio_reactive_video(parameters)


@router.post("/generate_audio_reactive_video_realtime")
def generate_audio_reactive_video_realtime(parameters: GenerateAudioReactiveVideoParameters) -> dict:
    """
    Description:
    ------------
        Generate Audio Reactive Video in Real-Time

    """
    return generate_service.generate_audio_reactive_video_realtime(parameters)

