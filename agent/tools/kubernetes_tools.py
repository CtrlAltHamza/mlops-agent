"""Kubernetes inspection tools.

State-changing calls require confirm=true and the approved namespace.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Protocol

from agent.safety import approved_namespace, ensure_confirmed, ensure_namespace
from agent.schemas import ToolUnavailable
from agent.tools.base import Tool


class ClusterGateway(Protocol):
    def list_pods(self, namespace: str) -> list[dict[str, Any]]: ...

    def list_deployments(self, namespace: str) -> list[dict[str, Any]]: ...

    def list_services(self, namespace: str) -> list[dict[str, Any]]: ...

    def list_events(self, namespace: str) -> list[dict[str, Any]]: ...

    def pod_logs(self, namespace: str, pod: str, tail: int) -> str: ...

    def restart_deployment(self, namespace: str, name: str) -> dict[str, Any]: ...


class KubernetesGateway:
    """Lazy wrapper around the official Kubernetes client."""

    def _apis(self) -> tuple[Any, Any]:
        try:
            from kubernetes import client, config
        except ImportError as exc:
            raise ToolUnavailable("kubernetes package is not installed") from exc
        try:
            config.load_kube_config()
        except Exception as exc:  # noqa: BLE001
            try:
                config.load_incluster_config()
            except Exception as inner:  # noqa: BLE001
                raise ToolUnavailable(f"kubeconfig is unavailable: {exc}") from inner
        return client.CoreV1Api(), client.AppsV1Api()

    def list_pods(self, namespace: str) -> list[dict[str, Any]]:
        core, _apps = self._apis()
        try:
            pods = core.list_namespaced_pod(namespace, _request_timeout=10)
        except Exception as exc:  # noqa: BLE001
            raise ToolUnavailable(f"pod list failed: {exc.__class__.__name__}") from exc
        return [_pod_dict(item) for item in pods.items]

    def list_deployments(self, namespace: str) -> list[dict[str, Any]]:
        _core, apps = self._apis()
        try:
            deployments = apps.list_namespaced_deployment(namespace, _request_timeout=10)
        except Exception as exc:  # noqa: BLE001
            raise ToolUnavailable(f"deployment list failed: {exc.__class__.__name__}") from exc
        payload = []
        for item in deployments.items:
            containers = item.spec.template.spec.containers or []
            images = [container.image for container in containers]
            labels = item.metadata.labels or {}
            payload.append(
                {
                    "name": item.metadata.name,
                    "replicas": item.spec.replicas,
                    "ready_replicas": item.status.ready_replicas or 0,
                    "images": images,
                    "version": labels.get("app.kubernetes.io/version")
                    or (item.spec.template.metadata.labels or {}).get("app.kubernetes.io/version"),
                }
            )
        return payload

    def list_services(self, namespace: str) -> list[dict[str, Any]]:
        core, _apps = self._apis()
        try:
            services = core.list_namespaced_service(namespace, _request_timeout=10)
        except Exception as exc:  # noqa: BLE001
            raise ToolUnavailable(f"service list failed: {exc.__class__.__name__}") from exc
        return [
            {
                "name": item.metadata.name,
                "type": item.spec.type,
                "cluster_ip": item.spec.cluster_ip,
                "ports": [port.port for port in (item.spec.ports or [])],
            }
            for item in services.items
        ]

    def list_events(self, namespace: str) -> list[dict[str, Any]]:
        core, _apps = self._apis()
        try:
            events = core.list_namespaced_event(namespace, _request_timeout=10)
        except Exception as exc:  # noqa: BLE001
            raise ToolUnavailable(f"event list failed: {exc.__class__.__name__}") from exc
        return [
            {
                "reason": item.reason,
                "message": item.message,
                "type": item.type,
                "involved": getattr(item.involved_object, "name", None),
                "count": item.count,
            }
            for item in events.items
        ]

    def pod_logs(self, namespace: str, pod: str, tail: int) -> str:
        core, _apps = self._apis()
        try:
            return core.read_namespaced_pod_log(
                name=pod,
                namespace=namespace,
                tail_lines=tail,
                _request_timeout=10,
            )
        except Exception as exc:  # noqa: BLE001
            raise ToolUnavailable(f"log read failed: {exc.__class__.__name__}") from exc

    def restart_deployment(self, namespace: str, name: str) -> dict[str, Any]:
        _core, apps = self._apis()
        now = datetime.now(timezone.utc).isoformat()
        body = {
            "spec": {
                "template": {
                    "metadata": {"annotations": {"kubectl.kubernetes.io/restartedAt": now}}
                }
            }
        }
        try:
            apps.patch_namespaced_deployment(name=name, namespace=namespace, body=body)
        except Exception as exc:  # noqa: BLE001
            raise ToolUnavailable(f"restart failed: {exc.__class__.__name__}") from exc
        return {"restarted": True, "deployment": name, "namespace": namespace, "at": now}


_GATEWAY: ClusterGateway = KubernetesGateway()


def set_gateway(gateway: ClusterGateway) -> None:
    global _GATEWAY
    _GATEWAY = gateway


def _namespace(namespace: str | None) -> str:
    return namespace or approved_namespace()


def list_workloads(namespace: str | None = None) -> dict[str, Any]:
    ns = _namespace(namespace)
    return {
        "namespace": ns,
        "pods": _GATEWAY.list_pods(ns),
        "deployments": _GATEWAY.list_deployments(ns),
    }


def get_pod_logs(pod: str, namespace: str | None = None, tail: int = 80) -> dict[str, Any]:
    if not pod:
        from agent.schemas import InvalidRequest

        raise InvalidRequest("pod name is required")
    ns = _namespace(namespace)
    return {"namespace": ns, "pod": pod, "logs": _GATEWAY.pod_logs(ns, pod, tail)}


def list_services(namespace: str | None = None) -> dict[str, Any]:
    ns = _namespace(namespace)
    return {"namespace": ns, "services": _GATEWAY.list_services(ns)}


def get_deployed_version(namespace: str | None = None) -> dict[str, Any]:
    ns = _namespace(namespace)
    deployments = _GATEWAY.list_deployments(ns)
    return {"namespace": ns, "deployments": deployments}


def diagnose_application(namespace: str | None = None, tail: int = 40) -> dict[str, Any]:
    ns = _namespace(namespace)
    pods = _GATEWAY.list_pods(ns)
    deployments = _GATEWAY.list_deployments(ns)
    services = _GATEWAY.list_services(ns)
    try:
        events = _GATEWAY.list_events(ns)
    except ToolUnavailable as exc:
        events = [{"reason": "EventLookupFailed", "message": str(exc), "type": "Warning"}]
    unhealthy = [pod for pod in pods if not _healthy(pod)]
    logs: dict[str, str] = {}
    for pod in unhealthy[:3]:
        try:
            logs[pod["name"]] = _GATEWAY.pod_logs(ns, pod["name"], tail)
        except ToolUnavailable as exc:
            logs[pod["name"]] = f"logs unavailable: {exc}"
    return {
        "namespace": ns,
        "pods": pods,
        "deployments": deployments,
        "services": services,
        "events": events[:20],
        "unhealthy_pods": unhealthy,
        "logs": logs,
        "healthy": not unhealthy and all((item.get("ready_replicas") or 0) >= 1 for item in deployments),
    }


def restart_deployment(
    name: str = "churn-inference",
    namespace: str | None = None,
    confirm: bool = False,
) -> dict[str, Any]:
    ensure_confirmed(confirm, "restart_deployment")
    ns = ensure_namespace(_namespace(namespace))
    return _GATEWAY.restart_deployment(ns, name)


def _healthy(pod: dict[str, Any]) -> bool:
    if pod.get("phase") not in {"Running", "Succeeded"}:
        return False
    if pod.get("ready") is False:
        return False
    waiting = pod.get("waiting_reason")
    return waiting in {None, ""}


def _pod_dict(item: Any) -> dict[str, Any]:
    statuses = item.status.container_statuses or []
    restarts = sum(status.restart_count or 0 for status in statuses)
    waiting_reason = None
    ready = True
    for status in statuses:
        if not status.ready:
            ready = False
        waiting = getattr(status.state, "waiting", None)
        if waiting is not None:
            waiting_reason = waiting.reason
    return {
        "name": item.metadata.name,
        "phase": item.status.phase,
        "ready": ready and item.status.phase == "Running",
        "restarts": restarts,
        "waiting_reason": waiting_reason,
        "node": item.spec.node_name,
    }


KUBERNETES_TOOLS = [
    Tool("list_workloads", list_workloads),
    Tool("get_pod_logs", get_pod_logs),
    Tool("list_services", list_services),
    Tool("get_deployed_version", get_deployed_version),
    Tool("diagnose_application", diagnose_application),
    Tool("restart_deployment", restart_deployment),
]
