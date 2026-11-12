"""Vibe's catalog APIs changed within 3.1.x; keep both runtimes supported."""

import asyncio


async def load_catalog(config, *, refresh):
    from vibe.core import model_catalog

    fetch = getattr(model_catalog, "fetch_and_cache_model_catalog", None)
    if fetch is None:
        return await _service_catalog(config, refresh=refresh)

    # Vibe 3.1.0/3.1.1: the cache and fetch helpers accept the user config.
    catalog = model_catalog.load_cached_model_catalog(config)
    source = "cache" if catalog is not None else "local"
    if refresh or catalog is None:
        try:
            async with asyncio.timeout(5):
                latest = await fetch(
                    config.model_copy(
                        update={"experimental_enable_model_catalog": True}
                    )
                )
        except TimeoutError:
            latest = None
        if latest is not None:
            catalog = latest
            source = "remote"
    return catalog, source


async def _service_catalog(config, *, refresh):
    # Vibe 3.1.2 moved fetching into a process-owned service and changed the
    # disk cache to take an account/endpoint key, returning (catalog, timestamp).
    # Use that service rather than reimplementing credential or cache handling.
    from vibe.app_server._model_catalog_service import (
        ModelCatalogKey,
        ModelCatalogService,
    )

    config = config.model_copy(update={"experimental_enable_model_catalog": True})
    service = ModelCatalogService()
    catalog = None
    source = "local"
    try:
        key = ModelCatalogKey.current(config, None)
        if key is not None:
            catalog = await service.load(key)
            if catalog is not None:
                source = "cache"
        if refresh or catalog is None:
            try:
                # Principal resolution (including an OIDC refresh) shares the
                # same overall budget as the HTTP request.
                async with asyncio.timeout(5):
                    if key is None:
                        key = await ModelCatalogKey.of(config, None)
                    latest = await service.refresh(key) if key is not None else None
            except TimeoutError:
                latest = None
            if latest is not None:
                catalog = latest
                source = "remote"
        return catalog, source
    finally:
        # A timed-out waiter does not cancel Vibe's shared fetch by itself.
        # This bridge owns the service; do not leave its tasks running.
        await service.close()
