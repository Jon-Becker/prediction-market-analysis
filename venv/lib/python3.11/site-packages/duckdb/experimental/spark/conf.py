from duckdb.experimental.spark.exception import ContributionsAcceptedError


class SparkConf:  # noqa: D101
    def __init__(self) -> None:  # noqa: D107
        raise NotImplementedError

    def contains(self, key: str) -> bool:  # noqa: D102
        raise ContributionsAcceptedError

    def get(self, key: str, defaultValue: str | None = None) -> str | None:  # noqa: D102
        raise ContributionsAcceptedError

    def getAll(self) -> list[tuple[str, str]]:  # noqa: D102
        raise ContributionsAcceptedError

    def set(self, key: str, value: str) -> "SparkConf":  # noqa: D102
        raise ContributionsAcceptedError

    def setAll(self, pairs: list[tuple[str, str]]) -> "SparkConf":  # noqa: D102
        raise ContributionsAcceptedError

    def setAppName(self, value: str) -> "SparkConf":  # noqa: D102
        raise ContributionsAcceptedError

    def setExecutorEnv(  # noqa: D102
        self, key: str | None = None, value: str | None = None, pairs: list[tuple[str, str]] | None = None
    ) -> "SparkConf":
        raise ContributionsAcceptedError

    def setIfMissing(self, key: str, value: str) -> "SparkConf":  # noqa: D102
        raise ContributionsAcceptedError

    def setMaster(self, value: str) -> "SparkConf":  # noqa: D102
        raise ContributionsAcceptedError

    def setSparkHome(self, value: str) -> "SparkConf":  # noqa: D102
        raise ContributionsAcceptedError

    def toDebugString(self) -> str:  # noqa: D102
        raise ContributionsAcceptedError


__all__ = ["SparkConf"]
