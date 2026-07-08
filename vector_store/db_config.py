from typing import Optional, List
from dataclasses import dataclass, field

from utils.config import (
    get_search_top_k,
    get_search_chunk_k,
    get_search_min_score,
    get_search_hybrid,
    get_search_hybrid_weight,
    get_db_host,
    get_db_port,
    get_db_name,
    get_db_user,
    get_db_password,
    get_db_url,
)


# SearchConfig and DatabaseConfig are maintained for compatibility with legacy callers.
@dataclass
class SearchConfig:
    """Search configuration parameters."""
    top_k: int = 10
    chunk_k: int = 3
    min_score: float = 0.01
    hybrid: bool = True
    hybrid_weight: float = 0.1
    categories: Optional[List[str]] = field(default_factory=lambda: None)

    @classmethod
    def from_config_file(cls, config_path: str = None) -> "SearchConfig":
        """Load configuration from config.cfg file."""
        return cls(
            top_k=get_search_top_k(),
            chunk_k=get_search_chunk_k(),
            min_score=get_search_min_score(),
            hybrid=get_search_hybrid(),
            hybrid_weight=get_search_hybrid_weight(),
            categories=None,
        )


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
        url = get_db_url()
        if url:
            return cls(url=url)
        return cls(
            host=get_db_host(),
            port=get_db_port(),
            name=get_db_name(),
            user=get_db_user(),
            password=get_db_password(),
        )

    @classmethod
    def from_config_file(cls, config_path: str = None) -> "DatabaseConfig":
        """Load configuration from config.cfg file."""
        url = get_db_url()
        if url:
            return cls(url=url)
        return cls(
            host=get_db_host(),
            port=get_db_port(),
            name=get_db_name(),
            user=get_db_user(),
            password=get_db_password(),
        )

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