from pathlib import Path
from dotenv import load_dotenv

# Load .env from the same directory as this script
_env_path = Path(__file__).resolve().parent / ".env"
load_dotenv(_env_path)

from src.agent import create_data_agent
from src.cli import start_cli


def main():
    agent, _ = create_data_agent()
    start_cli(agent)


if __name__ == "__main__":
    main()
