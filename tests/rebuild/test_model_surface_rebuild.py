from __future__ import annotations


def test_rebuilt_model_surface_imports() -> None:
    import xrpl_router.amm  # noqa: F401
    import xrpl_router.book_offers  # noqa: F401
    import xrpl_router.book_step  # noqa: F401
    import xrpl_router.core  # noqa: F401
    import xrpl_router.execution_semantics  # noqa: F401
    import xrpl_router.flow  # noqa: F401
    import xrpl_router.tx_intent  # noqa: F401
