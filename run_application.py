import asyncio
import os
import sys

# Ensure project root is in PYTHONPATH
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from agents.application_agent import ApplicationAgent
from core.config import get_settings
from core.database import init_db
from core.logger import logger, setup_logging


async def main():
    settings = get_settings()
    setup_logging(settings)
    await init_db(settings.storage.database_url)
    
    agent = ApplicationAgent()
    
    logger.info("Starting Autonomous Execution Phase 2 test run...")
    await agent.process_queue()
    logger.info("Finished execution run.")

if __name__ == "__main__":
    asyncio.run(main())
