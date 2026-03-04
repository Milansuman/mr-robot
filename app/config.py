from pydantic_settings import BaseSettings, SettingsConfigDict

class Config(BaseSettings):
    """Application settings loaded from environment variables."""
    
    GROQ_API_KEY: str
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=True,
    )


env = Config() #type: ignore