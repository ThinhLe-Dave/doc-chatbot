import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class DatabaseConfig:
    """PostgreSQL connection configuration."""
    host: str = "localhost"
    port: int = 5432
    name: str = "docchatbot"
    user: str = "docuser"
    password: str = ""
    url: Optional[str] = None
    embedding_dimension: int = 384

    @classmethod
    def from_env(cls) -> "DatabaseConfig":
        """Load configuration from environment variables."""
        url = os.environ.get("DATABASE_URL")
        if url:
            return cls(url=url)
        return cls(
            host=os.environ.get("DB_HOST", "localhost"),
            port=int(os.environ.get("DB_PORT", "5432")),
            name=os.environ.get("DB_NAME", "docchatbot"),
            user=os.environ.get("DB_USER", "docuser"),
            password=os.environ.get("DB_PASSWORD", ""),
        )

    @classmethod
    def from_config_file(cls, config_path: str = "config/config.cfg") -> "DatabaseConfig":
        """Load configuration from config.cfg file."""
        import configparser
        config = configparser.ConfigParser()
        config.read(config_path)

        if "database" in config:
            section = config["database"]
            url = section.get("url") or os.environ.get("DATABASE_URL")
            if url:
                return cls(url=url)
            return cls(
                host=section.get("host", os.environ.get("DB_HOST", "localhost")),
                port=int(section.get("port", os.environ.get("DB_PORT", "5432"))),
                name=section.get("name", os.environ.get("DB_NAME", "docchatbot")),
                user=section.get("user", os.environ.get("DB_USER", "docuser")),
                password=section.get("password", os.environ.get("DB_PASSWORD", "")),
            )
        return cls.from_env()

    def get_connection_string(self) -> str:
        """Build PostgreSQL connection string."""
        if self.url:
            return self.url
        return f"postgresql://{self.user}:{self.password}@{self.host}:{self.port}/{self.name}"

    def is_configured(self) -> bool:
        """Check if database connection is fully configured."""
        if self.url:
            return True
        return bool(self.host and self.name and self.user)


def get_db_config() -> Optional[DatabaseConfig]:
    """Get database configuration if available."""
    return DatabaseConfig.from_config_file()