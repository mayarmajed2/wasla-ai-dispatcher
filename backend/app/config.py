from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_env: str = "development"
    restaurant_timezone: str = "Africa/Cairo"
    dashboard_public_url: str = "http://127.0.0.1:8000"
    database_url: str = "sqlite:///./restaurant_ai.db"
    db_pool_size: int = 10
    db_max_overflow: int = 20
    meta_app_secret: str = ""
    meta_verify_token: str = "restaurant-ai-dev"
    meta_access_token: str = ""
    meta_phone_number_id: str = ""
    meta_graph_version: str = "v27.0"
    meta_is_official_business_account: bool = False
    meta_billable_messages_authorized: bool = False
    meta_monthly_free_service_limit: int = 1000
    gemini_api_key: str = ""
    groq_api_key: str = ""
    rate_limit_per_minute: int = 120

    # Zero-cost Cashier Group Notifications (Telegram Bot API)
    telegram_bot_token: str = ""
    cashier_telegram_chat_id: str = ""

    # Owner Escalation Settings
    restaurant_owner_whatsapp: str = ""
    restaurant_owner_telegram_chat_id: str = ""
    complaint_escalation_enabled: bool = True

    # Sales Assistant & Review Scheduling
    upsell_enabled: bool = True
    review_delay_minutes: int = 60
    arrival_check_delay_minutes: int = 90

    # Authentication & RBAC
    auth_secret_key: str = "change-me-in-production-min-32-chars-secret"
    auth_token_ttl_hours: int = 12
    bootstrap_admin_username: str = "admin"
    bootstrap_admin_password: str = "WaslaAdmin2026!"

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")


settings = Settings()
