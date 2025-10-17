#!/usr/bin/env python3
"""
Seed data script for the Ontology-Aware Memory System.
This script runs the seed data migration to populate the database with initial data.
"""

import logging
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def run_seed_migration():
    """Run the seed data migration."""
    try:
        logger.info("Running seed data migration...")
        
        # Run alembic upgrade to ensure all migrations are applied
        result = subprocess.run(
            ["alembic", "upgrade", "head"],
            cwd=Path(__file__).parent.parent,
            capture_output=True,
            text=True,
            check=True
        )
        
        logger.info("Migration completed successfully")
        logger.info(f"Migration output: {result.stdout}")
        
        return True
        
    except subprocess.CalledProcessError as e:
        logger.error(f"Migration failed: {e}")
        logger.error(f"Error output: {e.stderr}")
        return False
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        return False


def main():
    """Main function."""
    logger.info("Starting seed data script...")
    
    success = run_seed_migration()
    
    if success:
        logger.info("Seed data script completed successfully")
        sys.exit(0)
    else:
        logger.error("Seed data script failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
