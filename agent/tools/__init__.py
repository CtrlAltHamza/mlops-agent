from agent.tools.base import Registry, Tool


def build_registry() -> Registry:
    from agent.tools.airflow_tools import AIRFLOW_TOOLS
    from agent.tools.dvc_tools import DVC_TOOLS
    from agent.tools.kubernetes_tools import KUBERNETES_TOOLS
    from agent.tools.mlflow_tools import MLFLOW_TOOLS

    registry = Registry()
    for tool in (*DVC_TOOLS, *MLFLOW_TOOLS, *AIRFLOW_TOOLS, *KUBERNETES_TOOLS):
        registry.add(tool)
    return registry


__all__ = ["Registry", "Tool", "build_registry"]
