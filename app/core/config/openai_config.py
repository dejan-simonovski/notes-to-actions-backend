from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

class OpenaiSettings(BaseSettings): 
    OPENAI_API_KEY: str = Field(
        default="",
        validation_alias="OPENAI_API_KEY"
    )
    OPENAI_MODEL: str = Field(
        default="gpt-4o",
        validation_alias="OPENAI_MODEL"
    )
    OPENAI_TEMPERATURE: float = Field(
        default=0.7,
        validation_alias="OPENAI_TEMPERATURE"
    )
    OPENAI_MAX_TOKENS: int = Field(
        default=1024,
        validation_alias="OPENAI_MAX_TOKENS"
    )

    GEMINI_API_KEY: str = Field(
        default="",
        validation_alias="GEMINI_API_KEY"
    )

    model_config = SettingsConfigDict(
        env_file=".env.chatbot",
        extra="ignore"
    )

settings = OpenaiSettings()