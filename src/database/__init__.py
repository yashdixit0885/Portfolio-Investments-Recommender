"""
Database package for Portfolio Investments Recommender

This package provides database functionality for storing and retrieving:
- Securities data
- Historical price data
- Analysis results
"""

from .db_manager import DatabaseManager
from .config import DB_PATH, DB_SETTINGS, TABLE_SCHEMAS, INDEXES

__all__ = ['DatabaseManager', 'DB_PATH', 'DB_SETTINGS', 'TABLE_SCHEMAS', 'INDEXES'] 