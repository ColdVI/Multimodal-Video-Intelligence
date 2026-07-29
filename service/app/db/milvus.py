"""Optional Milvus adapter; enabled only by the compose `milvus` profile."""

SUPPORTED_STRATEGIES = ("ann",)


def available() -> bool:
    try:
        import pymilvus  # noqa: F401
        return True
    except ImportError:
        return False
