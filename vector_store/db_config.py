import os
from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class SearchConfig:
    """Search configuration parameters."""
    top_k: int = 10
    chunk_k: int = 3
    min_score: float = 0.01
    hybrid: bool = False
    hybrid_weight: float = 0.1
    categories: Optional[List[str]] = field(default_factory=lambda: None)

    @classmethod
    def from_config_file(cls, config_path: str = "config/config.cfg") -> "SearchConfig":
        """Load configuration from config.cfg file."""
        import configparser
        config = configparser.ConfigParser()
        config.read(config_path)

        if "search" in config:
            section = config["search"]
            return cls(
                top_k=int(section.get("top_k", "10")),
                chunk_k=int(section.get("chunk_k", "3")),
                min_score=float(section.get("min_score", "0.01")),
                hybrid=section.get("hybrid", "false").lower() in ("true", "1", "yes"),
                hybrid_weight=float(section.get("hybrid_weight", "0.1")),
                categories=None,
            )
        return cls()


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