from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class MeetingConfig(BaseSettings):
    """
    Application configuration class using Pydantic Settings.
    It reads environment variables automatically.
    """
    SLACK_WEBHOOK_URL: str = Field(
        default="https://hooks.slack.com/services/MOCK/URL",
        validation_alias="SLACK_WEBHOOK_URL"
    )
    OPENAI_API_KEY: str = Field(
        default="",
        validation_alias="OPENAI_API_KEY"
    )

    # Configure Pydantic Settings to load from local .env files
    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore"
    )
