"""One model call through the installed Vibe runtime; never starts its tool loop.

Stdio is a private NDJSON transport. Credentials are resolved by Vibe and never
leave this process. This adapter targets mistral-vibe 3.1.x's Python interfaces.
"""

import asyncio
import json
import sys
from importlib.metadata import version


def emit(value):
    print(json.dumps(value, separators=(",", ":")), flush=True)


async def main():
    release = version("mistral-vibe")
    if not release.startswith("3.1."):
        emit(
            {
                "type": "error",
                "message": "This integration requires mistral-vibe 3.1.x. Install it with uv tool install mistral-vibe==3.1.0.",
            }
        )
        return

    from pydantic import TypeAdapter
    from vibe.core.config import build_default_orchestrator, load_dotenv_values
    from vibe.core.config.harness_files import init_harness_files_manager
    from vibe.utils.api_keys import resolve_api_key

    request = json.loads(sys.stdin.readline())
    init_harness_files_manager("user")
    load_dotenv_values()
    # Only user model/connection settings; project tools, hooks and instructions
    # belong to OpenCode. No Vibe session or agent manager is constructed.
    orchestrator = await build_default_orchestrator()
    config = orchestrator.config
    from vibe.core.config.layers.model_catalog import apply_model_catalog
    from vibe.core.model_catalog import (
        fetch_and_cache_model_catalog,
        load_cached_model_catalog,
    )

    # Building Vibe's config leaves its model-catalog layer empty. Populate it
    # exactly as the CLI does, without persisting an experimental flag in Vibe.
    catalog = load_cached_model_catalog(config)
    source = "cache" if catalog is not None else "local"
    if request["type"] == "probe" or catalog is None:
        try:
            async with asyncio.timeout(5):
                latest = await fetch_and_cache_model_catalog(
                    config.model_copy(
                        update={"experimental_enable_model_catalog": True}
                    )
                )
        except TimeoutError:
            latest = None
        if latest is not None:
            catalog = latest
            source = "remote"
    if catalog is not None:
        await apply_model_catalog(orchestrator, catalog)
        config = orchestrator.config

    models = {
        alias: model
        for alias, model in config.available_models().items()
        if str(config.get_provider_for_model(model).backend) == "mistral"
    }
    if request["type"] == "probe":
        emit(
            {
                "type": "catalog",
                "version": release,
                "source": source,
                "models": [
                    {
                        "id": alias,
                        "name": model.display_name or alias,
                        "context": model.max_context_length or 0,
                        "output": model.max_output_tokens or 0,
                        "images": bool(model.supports_images),
                        "efforts": model.thinking_levels or [],
                        "authenticated": bool(
                            resolve_api_key(
                                config.get_provider_for_model(model).api_key_env_var
                            )
                        ),
                    }
                    for alias, model in models.items()
                ],
            }
        )
        return

    from mistralai_vibe_local_harness.protocol import RustMessage, RustToolDefinition
    from mistralai_vibe_local_harness.vibe import (
        LocalModelRoute,
        LocalRuntimeAdapterConfig,
        ProviderCredentialSnapshot,
    )
    from mistralai_vibe_local_harness.vibe.adapters.mistral import (
        execute_mistral_completion,
    )
    from vibe.core.llm.completion import _compile_reasoning
    from vibe.core.llm.reasoning import profile_thinking

    model = models.get(request["model"])
    if model is None:
        emit(
            {
                "type": "error",
                "message": "Model is no longer available in Vibe. Reconnect Mistral Vibe to refresh the model list.",
            }
        )
        return
    provider = config.get_provider_for_model(model)
    token = resolve_api_key(provider.api_key_env_var)
    if not token:
        emit(
            {
                "type": "error",
                "message": "Mistral Vibe is not logged in. Run vibe --setup on the OpenCode server, then reconnect Mistral Vibe.",
            }
        )
        return
    effort = request.get("effort")
    if effort is not None:
        if effort not in (model.thinking_levels or []):
            emit(
                {
                    "type": "error",
                    "message": "The selected reasoning effort is not supported by this Vibe model.",
                }
            )
            return
        model = model.model_copy(update={"thinking": effort})
    adapter = LocalRuntimeAdapterConfig(
        provider=provider.name,
        backend="mistral",
        base_url=provider.api_base,
        max_tokens=request.get("maxTokens"),
        extra_headers=provider.extra_headers,
        emits_finish_reason=provider.emits_finish_reason,
        timeout_s=config.api_timeout,
        stream_idle_timeout_s=config.api_stream_idle_timeout or None,
        # OpenCode owns retries; replaying a partially emitted stream duplicates it.
        retry_max_elapsed_time_s=0,
        enable_system_trust_store=config.enable_system_trust_store,
    )
    compiled = _compile_reasoning(model, adapter)
    route = LocalModelRoute(
        model=model.name,
        temperature=request.get("temperature", model.temperature),
        top_p=request.get("topP", model.top_p),
        thinking=profile_thinking(compiled),
        reasoning=compiled.profile.format.reasoning,
        max_output_tokens=model.max_output_tokens,
    )

    async def delta(value):
        if value.text:
            emit({"type": "text", "text": value.text})
        if value.reasoning:
            emit({"type": "reasoning", "text": value.reasoning})

    result = await execute_mistral_completion(
        TypeAdapter(list[RustMessage]).validate_python(request["messages"]),
        TypeAdapter(list[RustToolDefinition]).validate_python(request["tools"]),
        adapter,
        ProviderCredentialSnapshot(token=token, headers={}, revision="vibe"),
        route,
        on_delta=delta,
    )
    emit({"type": "result", **result.model_dump(mode="json", by_alias=True)})


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except Exception as error:  # noqa: BLE001 - sanitize every exception at the process boundary
        # Provider/config exceptions can contain headers, credentials and prompts.
        # Only the class and numeric HTTP status cross the process boundary.
        status = getattr(error, "status_code", None)
        emit(
            {
                "type": "error",
                "message": "Mistral Vibe bridge failed ("
                + type(error).__name__
                + "). Check the Vibe installation and run vibe --setup.",
                **({"status": status} if isinstance(status, int) else {}),
            }
        )
        sys.exit(1)
