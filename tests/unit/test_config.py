import pytest

from finsight.core.config import Settings, load_app_config, load_companies
from finsight.core.exceptions import ConfigError


def test_companies_load():
    companies = load_companies()
    tickers = {c.ticker for c in companies}
    assert {"NVDA", "AMD", "AAPL", "TSLA"} <= tickers
    assert all(len(c.cik) == 10 for c in companies)


def test_app_config_has_core_sections():
    cfg = load_app_config()
    assert {"models", "chunking", "retrieval", "agent"} <= set(cfg)


def test_postgres_dsn():
    s = Settings(_env_file=None, postgres_host="db", postgres_port=5433)
    assert s.postgres_dsn.endswith("@db:5433/finsight")


def test_sec_user_agent_required():
    with pytest.raises(ConfigError):
        Settings(_env_file=None, sec_user_agent="").require_sec_user_agent()