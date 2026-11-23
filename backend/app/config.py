from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    database_url: str = "sqlite:///./restaurant_ai.db"
    meta_app_secret: str = ""
    meta_verify_token: str = "restaurant-ai-dev"
    meta_access_token: str = ""
    gemini_api_key: str = ""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
