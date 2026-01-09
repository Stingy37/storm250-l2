"""
Automatically picked up by pytest. 
"""
import logging

def pytest_configure():
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
