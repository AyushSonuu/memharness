# memharness - Framework-agnostic memory infrastructure for AI agents
# Copyright (c) 2026 Ayush Sonuu
# Licensed under MIT License

"""
Custom exceptions for memharness.

This module defines all custom exceptions used throughout the memharness package.
"""

from __future__ import annotations


class MemharnessError(Exception):
    """Base exception for all memharness errors."""

    pass


class BackendError(MemharnessError):
    """Exception raised for backend storage errors."""

    pass


class ConnectionError(MemharnessError):
    """Exception raised for backend connection errors."""

    pass


class ConfigError(MemharnessError):
    """Exception raised for configuration errors."""

    pass
