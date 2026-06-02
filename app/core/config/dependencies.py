from functools import lru_cache
from fastapi import Depends
from app.core.config.meeting_config import MeetingConfig
from app.services.ai_service import AiService
from app.services.meeting_service import MeetingService

@lru_cache
def get_meeting_config() -> MeetingConfig:
    """
    Dependency provider for app configuration. Cached for singleton behavior.
    """
    return MeetingConfig()

@lru_cache
def get_ai_service() -> AiService:
    """
    Dependency provider for AI Service. Cached for singleton behavior.
    """
    config = get_meeting_config()
    return AiService(api_key=config.OPENAI_API_KEY)

def get_meeting_service(
    ai_service: AiService = Depends(get_ai_service),
    config: MeetingConfig = Depends(get_meeting_config)
) -> MeetingService:
    """
    Dependency provider for Meeting Service.
    Wired properly by injecting AiService and MeetingConfig.
    """
    return MeetingService(ai_service=ai_service, config=config)
