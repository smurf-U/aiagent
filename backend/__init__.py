"""
DeepAgents backend.

Two entry points share these modules:

- API:    ``uvicorn backend.main:app``  (HTTP + WebSocket; never runs the agent)
- Worker: ``python -m backend.worker``  (runs agent tasks from the Redis queue)

The package import is kept side-effect free so the worker does not load the
web app, and the API does not build agents at import time.
"""
