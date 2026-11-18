"""Google Colab with MCP & CLI suite.

Provides:
- ColabMCPClient: Official MCP JSON-RPC client
- ColabRESTClient: REST API client
- JupyterProxy: Direct file transfer via X-Colab-Runtime-Proxy-Token
- ColabConfig: Configuration loader
"""

from .config import ColabConfig
from .mcp_client import ColabMCPClient
from .api import ColabRESTClient
from .jupyter_proxy import JupyterProxy

__version__ = "0.1.0"
__all__ = ["ColabConfig", "ColabMCPClient", "ColabRESTClient", "JupyterProxy"]
