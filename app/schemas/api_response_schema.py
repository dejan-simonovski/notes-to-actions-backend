from typing import Generic, TypeVar
from pydantic import BaseModel

T = TypeVar("T")

class ApiResponse(BaseModel, Generic[T]):
    """
    Global response wrapper to standardize all API responses.
    """
    success: bool
    message: str
    data: T | None = None
