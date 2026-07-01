from typing import NamedTuple

from .session import SparkSession


class Database(NamedTuple):  # noqa: D101
    name: str
    description: str | None
    locationUri: str


class Table(NamedTuple):  # noqa: D101
    name: str
    database: str | None
    description: str | None
    tableType: str
    isTemporary: bool


class Column(NamedTuple):  # noqa: D101
    name: str
    description: str | None
    dataType: str
    nullable: bool
    isPartition: bool
    isBucket: bool


class Function(NamedTuple):  # noqa: D101
    name: str
    description: str | None
    className: str
    isTemporary: bool


class Catalog:  # noqa: D101
    def __init__(self, session: SparkSession) -> None:  # noqa: D107
        self._session = session

    def listDatabases(self) -> list[Database]:  # noqa: D102
        res = self._session.conn.sql("select database_name from duckdb_databases()").fetchall()

        def transform_to_database(x: list[str]) -> Database:
            return Database(name=x[0], description=None, locationUri="")

        databases = [transform_to_database(x) for x in res]
        return databases

    def listTables(self) -> list[Table]:  # noqa: D102
        res = self._session.conn.sql("select table_name, database_name, sql, temporary from duckdb_tables()").fetchall()

        def transform_to_table(x: list[str]) -> Table:
            return Table(name=x[0], database=x[1], description=x[2], tableType="", isTemporary=x[3])

        tables = [transform_to_table(x) for x in res]
        return tables

    def listColumns(self, tableName: str, dbName: str | None = None) -> list[Column]:  # noqa: D102
        query = f"""
			select column_name, data_type, is_nullable from duckdb_columns() where table_name = '{tableName}'
		"""
        if dbName:
            query += f" and database_name = '{dbName}'"
        res = self._session.conn.sql(query).fetchall()

        def transform_to_column(x: list[str | bool]) -> Column:
            return Column(name=x[0], description=None, dataType=x[1], nullable=x[2], isPartition=False, isBucket=False)

        columns = [transform_to_column(x) for x in res]
        return columns

    def listFunctions(self, dbName: str | None = None) -> list[Function]:  # noqa: D102
        raise NotImplementedError

    def setCurrentDatabase(self, dbName: str) -> None:  # noqa: D102
        raise NotImplementedError


__all__ = ["Catalog", "Column", "Database", "Function", "Table"]
