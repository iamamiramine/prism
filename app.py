from fastapi import FastAPI

from src.api.controllers import (
    health_controller,
    generate_controller,
)
from src.handlers.exception_handler import add_exception_handlers

tags_metadata = [
    {
        "name": "health",
        "description": "checks the health of the API services",
    },
    {
        "name": "audio_reactive",
        "description": "Audio Reactive",
    }
]


app = FastAPI(
    version="1.0",
    title="Audio Reactive LDM API",
    description="API for Audio Reactive LDM",
    openapi_tags=tags_metadata,
)

app.include_router(
    health_controller.router,
    prefix="/health",
    tags=["health"],
    responses={404: {"description": "Not found"}},
)
app.include_router(
    generate_controller.router,
    prefix="/audio_reactive",
    tags=["audio_reactive"],
    responses={404: {"description": "Not found"}},
)

add_exception_handlers(app=app)


# if __name__ == "__main__":
#     import uvicorn
#
#     uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
