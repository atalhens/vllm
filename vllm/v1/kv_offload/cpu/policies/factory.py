# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
import importlib
import inspect
from collections.abc import Callable
from collections.abc import Mapping
from typing import Any

from vllm.logger import init_logger
from vllm.v1.kv_offload.cpu.policies.base import CachePolicy

logger = init_logger(__name__)


class CachePolicyFactory:
    """Registry for CachePolicy implementations, resolved by name.

    Mirrors OffloadingSpecFactory (vllm/v1/kv_offload/factory.py): built-in
    policies are pre-registered below. External policies can either
    register_cache_policy() a friendly short name up front, or skip
    registration entirely and pass a module path at lookup time (out-of-tree,
    no vLLM fork/patch required) -- see get_cache_policy_cls.
    """

    _registry: dict[str, Callable[[], type[CachePolicy]]] = {}

    @classmethod
    def register_cache_policy(
        cls, name: str, module_path: str, class_name: str
    ) -> None:
        """Register a cache policy with a lazy-loading module and class name."""
        if name in cls._registry:
            raise ValueError(f"Cache policy '{name}' is already registered.")

        def loader() -> type[CachePolicy]:
            module = importlib.import_module(module_path)
            return getattr(module, class_name)

        cls._registry[name] = loader

    @classmethod
    def get_cache_policy_cls(
        cls, name: str, module_path: str | None = None
    ) -> type[CachePolicy]:
        """Get a cache policy class by name.

        Args:
            name: Name of the cache policy. Checked against the registry
                first; if it's not registered and `module_path` is given,
                `name` is imported from there instead -- an out-of-tree
                policy needs no register_cache_policy() call at all, just
                this module path passed through config (mirrors
                OffloadingSpecFactory.get_spec_cls's spec_module_path
                fallback).
            module_path: Python import path to load `name` from when it is
                not a registered policy.

        Returns:
            The cache policy class.

        Raises ValueError if the cache policy is neither registered nor
        resolvable via `module_path`.
        """
        if name in cls._registry:
            return cls._registry[name]()
        if module_path is None:
            raise ValueError(
                f"Unknown cache policy: {name!r}. Supported: {list(cls._registry)}. "
                "For an out-of-tree policy, also set cache_policy_module_path."
            )
        logger.warning_once(
            "Loading out-of-tree cache policy. This API is "
            "experimental and subject to change in the future "
            "as we iterate the design."
        )
        logger.info(
            "Loading out-of-tree cache policy '%s' from '%s'.",
            name,
            module_path,
        )
        module = importlib.import_module(module_path)
        policy_cls = getattr(module, name)
        assert issubclass(policy_cls, CachePolicy)
        return policy_cls


# Register built-in policies here.
CachePolicyFactory.register_cache_policy(
    "lru", "vllm.v1.kv_offload.cpu.policies.lru", "LRUCachePolicy"
)
CachePolicyFactory.register_cache_policy(
    "arc", "vllm.v1.kv_offload.cpu.policies.arc", "ARCCachePolicy"
)
CachePolicyFactory.register_cache_policy(
    "prefix_pruning_on_sessions",
    "vllm.v1.kv_offload.cpu.policies.prefix_pruning",
    "PrefixPruningOnSessionsCachePolicy",
)
CachePolicyFactory.register_cache_policy(
    "prefix_pruning",
    "vllm.v1.kv_offload.cpu.policies.prefix_pruning",
    "PrefixPruningCachePolicy",
)


def cache_policy_config_from_extra(extra_config: Mapping[str, Any]) -> dict[str, Any]:
    nested = extra_config.get("eviction_policy_config")
    config: dict[str, Any] = dict(nested) if isinstance(nested, dict) else {}
    for key in (
        "first_tier_tokens",
        "tier_step_tokens",
        "max_tiers",
        "session_dead_ttl_s",
        "session_ewa_alpha",
        "session_dead_k",
        "session_min_gaps",
        "session_detector",
        "session_quantile",
        "session_gap_window",
    ):
        if key in extra_config:
            config[key] = extra_config[key]
    return config


def instantiate_cache_policy(
    policy_cls: type[CachePolicy],
    cache_capacity: int,
    policy_config: Mapping[str, Any] | None = None,
) -> CachePolicy:
    accepted = inspect.signature(policy_cls.__init__).parameters
    kwargs = {
        key: value
        for key, value in (policy_config or {}).items()
        if key in accepted
    }
    return policy_cls(cache_capacity=cache_capacity, **kwargs)
