"""
Endpoint for checking the health of the API services.

The prefix endpoint is '/health'
"""

from fastapi import APIRouter
from starlette.responses import Response

router = APIRouter()


@router.get("")
def health():
    """
    Description:
    ------------
        Check services health. Endpoint: '/health'

    """
    return Response(status_code=200)