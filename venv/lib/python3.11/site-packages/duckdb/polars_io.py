from __future__ import annotations  # noqa: D100

import datetime
import io
import json
import typing
from decimal import Decimal

import polars as pl
from polars.io.plugins import register_io_source

import duckdb

if typing.TYPE_CHECKING:
    from collections.abc import Iterator

    import typing_extensions

_ExpressionTree: typing_extensions.TypeAlias = typing.Dict[str, typing.Union[str, int, "_ExpressionTree", typing.Any]]  # noqa: UP006


def _predicate_to_expression(predicate: pl.Expr) -> duckdb.Expression | None:
    """Convert a Polars predicate expression to a DuckDB-compatible SQL expression.

    Parameters:
        predicate (pl.Expr): A Polars expression (e.g., col("foo") > 5)

    Returns:
        SQLExpression: A DuckDB SQL expression string equivalent.
        None: If conversion fails.

    Example:
        >>> _predicate_to_expression(pl.col("foo") > 5)
        SQLExpression("(foo > 5)")
    """
    # Serialize the Polars expression tree to JSON
    tree = json.loads(predicate.meta.serialize(format="json"))
    return _tree_to_sql_expression(tree)


def _tree_to_sql_expression(tree: _ExpressionTree) -> duckdb.Expression | None:
    """Convert an already-parsed Polars expression tree to a DuckDB expression.

    Returns None if the tree contains a node we cannot translate to SQL.
    """
    try:
        return duckdb.SQLExpression(_pl_tree_to_sql(tree))
    except Exception:
        # If the conversion fails, we return None
        return None


# Polars "dynamic predicates"
# ---------------------------
# When a slice / TOP-N sits above a scan, polars' optimizer pushes a *dynamic
# predicate* into the scan, AND-ed onto any real filter. It is an internal
# optimizer node, not a materializable expression: it serializes as a `Display`
# node with `fmt_str` "dynamic_pred: <uuid>", and feeding it back into
# `DataFrame.filter()` panics polars (`unreachable!` in `expr_to_ir`). It is only
# an early-pruning hint -- the limit above the scan still runs -- so we drop it
# and keep the real predicate, which we MUST still apply (polars trusts the
# source and does not re-filter above it). Reported as duckdb-python#460; for
# polars-side context see:
#   - https://github.com/pola-rs/polars/issues/21665  (why real + dynamic arrive AND-ed)
#   - https://github.com/pola-rs/polars/issues/22252  (filter() panicking on un-lowerable nodes)
def _is_dynamic_predicate_node(node: typing.Any) -> bool:  # noqa: ANN401
    """Return True if a serialized node is a dynamic predicate (see the note above).

    Detected by shape: a ``Display`` node whose ``fmt_str`` starts with ``dynamic_pred``.
    """
    if not isinstance(node, dict):
        return False
    display = node.get("Display")
    return (
        isinstance(display, dict)
        and isinstance(display.get("fmt_str"), str)
        and display["fmt_str"].startswith("dynamic_pred")
    )


def _tree_contains_dynamic_predicate(node: typing.Any) -> bool:  # noqa: ANN401
    """Return True if the serialized expression tree contains a dynamic predicate anywhere."""
    if _is_dynamic_predicate_node(node):
        return True
    if isinstance(node, dict):
        return any(_tree_contains_dynamic_predicate(child) for child in node.values())
    if isinstance(node, list):
        return any(_tree_contains_dynamic_predicate(child) for child in node)
    return False


def _strip_dynamic_predicates(tree: typing.Any) -> tuple[_ExpressionTree | None, bool]:  # noqa: ANN401
    """Remove dynamic-predicate conjuncts from a serialized predicate tree.

    See the note above for what a dynamic predicate is and why we drop it.

    Returns ``(stripped_tree, removed)``. ``stripped_tree`` is ``None`` when the
    predicate was purely dynamic. Raises ``NotImplementedError`` if a dynamic
    predicate appears anywhere other than a top-level ``And`` conjunct — a shape
    polars does not produce today, where the hint can neither be safely dropped
    nor applied.
    """
    if _is_dynamic_predicate_node(tree):
        return None, True
    if isinstance(tree, dict) and "BinaryExpr" in tree:
        bin_expr = tree["BinaryExpr"]
        if isinstance(bin_expr, dict) and bin_expr.get("op") == "And":
            left, left_removed = _strip_dynamic_predicates(bin_expr["left"])
            right, right_removed = _strip_dynamic_predicates(bin_expr["right"])
            removed = left_removed or right_removed
            if left is None:
                return right, removed
            if right is None:
                return left, removed
            return {"BinaryExpr": {**bin_expr, "left": left, "right": right}}, removed
    if _tree_contains_dynamic_predicate(tree):
        msg = "Cannot handle a polars dynamic predicate outside a top-level AND conjunct"
        raise NotImplementedError(msg)
    return tree, False


def _expression_from_tree(tree: _ExpressionTree) -> pl.Expr:
    """Rebuild a polars expression from a serialized tree (inverse of meta.serialize)."""
    return pl.Expr.deserialize(io.BytesIO(json.dumps(tree).encode()), format="json")


def _pl_operation_to_sql(op: str) -> str:
    """Map Polars binary operation strings to SQL equivalents.

    Example:
        >>> _pl_operation_to_sql("Eq")
        '='
    """
    try:
        return {
            "Lt": "<",
            "LtEq": "<=",
            "Gt": ">",
            "GtEq": ">=",
            "Eq": "=",
            "Modulus": "%",
            "And": "AND",
            "Or": "OR",
        }[op]
    except KeyError:
        raise NotImplementedError(op)  # noqa: B904


def _escape_sql_identifier(identifier: str) -> str:
    """Escape SQL identifiers by doubling any double quotes and wrapping in double quotes.

    Example:
        >>> _escape_sql_identifier('column"name')
        '"column""name"'
    """
    escaped = identifier.replace('"', '""')
    return f'"{escaped}"'


def _pl_tree_to_sql(tree: _ExpressionTree) -> str:
    """Recursively convert a Polars expression tree (as JSON) to a SQL string.

    Parameters:
        tree (dict): JSON-deserialized expression tree from Polars

    Returns:
        str: SQL expression string

    Example:
        Input tree:
        {
            "BinaryExpr": {
                "left": { "Column": "foo" },
                "op": "Gt",
                "right": { "Literal": { "Int": 5 } }
            }
        }
        Output: "(foo > 5)"
    """
    [node_type] = tree.keys()

    if node_type == "BinaryExpr":
        # Binary expressions: left OP right
        bin_expr_tree = tree[node_type]
        assert isinstance(bin_expr_tree, dict), f"A {node_type} should be a dict but got {type(bin_expr_tree)}"
        lhs, op, rhs = bin_expr_tree["left"], bin_expr_tree["op"], bin_expr_tree["right"]
        assert isinstance(lhs, dict), f"LHS of a {node_type} should be a dict but got {type(lhs)}"
        assert isinstance(op, str), f"The op of a {node_type} should be a str but got {type(op)}"
        assert isinstance(rhs, dict), f"RHS of a {node_type} should be a dict but got {type(rhs)}"
        return f"({_pl_tree_to_sql(lhs)} {_pl_operation_to_sql(op)} {_pl_tree_to_sql(rhs)})"
    if node_type == "Column":
        # A reference to a column name
        # Wrap in quotes to handle special characters
        col_name = tree[node_type]
        assert isinstance(col_name, str), f"The col name of a {node_type} should be a str but got {type(col_name)}"
        return _escape_sql_identifier(col_name)

    if node_type in ("Literal", "Dyn"):
        # Recursively process dynamic or literal values
        val_tree = tree[node_type]
        assert isinstance(val_tree, dict), f"A {node_type} should be a dict but got {type(val_tree)}"
        return _pl_tree_to_sql(val_tree)

    if node_type == "Int":
        # Direct integer literals
        int_literal = tree[node_type]
        assert isinstance(int_literal, (int, str)), (
            f"The value of an Int should be an int or str but got {type(int_literal)}"
        )
        return str(int_literal)

    if node_type == "Float":
        # Direct float literals
        float_literal = tree[node_type]
        assert isinstance(float_literal, (float, int, str)), (
            f"The value of a Float should be a float, int or str but got {type(float_literal)}"
        )
        return str(float_literal)

    if node_type == "Function":
        # Handle boolean functions like IsNull, IsNotNull
        func_tree = tree[node_type]
        assert isinstance(func_tree, dict), f"A {node_type} should be a dict but got {type(func_tree)}"
        inputs = func_tree["input"]
        assert isinstance(inputs, list), f"A {node_type} should have a list of dicts as input but got {type(inputs)}"
        input_tree = inputs[0]
        assert isinstance(input_tree, dict), (
            f"A {node_type} should have a list of dicts as input but got {type(input_tree)}"
        )
        func_dict = func_tree["function"]
        assert isinstance(func_dict, dict), (
            f"A {node_type} should have a function dict as input but got {type(func_dict)}"
        )

        if "Boolean" in func_dict:
            func = func_dict["Boolean"]
            arg_sql = _pl_tree_to_sql(inputs[0])

            if func == "IsNull":
                return f"({arg_sql} IS NULL)"
            if func == "IsNotNull":
                return f"({arg_sql} IS NOT NULL)"
            msg = f"Boolean function not supported: {func}"
            raise NotImplementedError(msg)

        msg = f"Unsupported function type: {func_dict}"
        raise NotImplementedError(msg)

    if node_type == "Cast":
        cast_tree = tree[node_type]
        assert isinstance(cast_tree, dict), f"A {node_type} should be a dict but got {type(cast_tree)}"
        options = cast_tree.get("options")
        if options == "Strict":
            # Strict casts on literals (e.g. pl.lit(1, dtype=pl.Int8)) are safe to unwrap —
            # the value is known at expression creation time. Strict casts on columns
            # (e.g. pl.col("a").cast(pl.Int64)) are semantically meaningful and must not be dropped.
            cast_expr = cast_tree.get("expr", {})
            if not isinstance(cast_expr, dict) or "Literal" not in cast_expr:
                msg = "Strict cast on non-literal expression cannot be pushed down"
                raise NotImplementedError(msg)
        elif options != "NonStrict":
            msg = f"Only NonStrict/Strict casts can be safely unwrapped, got {options!r}"
            raise NotImplementedError(msg)
        cast_expr = cast_tree["expr"]
        assert isinstance(cast_expr, dict), f"A {node_type} should be a dict but got {type(cast_expr)}"
        return _pl_tree_to_sql(cast_expr)

    if node_type == "Scalar":
        # Detect format: old style (dtype/value) or new style (direct type key)
        scalar_tree = tree[node_type]
        assert isinstance(scalar_tree, dict), f"A {node_type} should be a dict but got {type(scalar_tree)}"
        if "dtype" in scalar_tree and "value" in scalar_tree:
            dtype = str(scalar_tree["dtype"])
            value = scalar_tree["value"]
        else:
            # New style: dtype is the single key in the dict
            dtype = next(iter(scalar_tree.keys()))
            value = scalar_tree
        assert isinstance(dtype, str), f"A {node_type} should have a str dtype but got  {type(dtype)}"
        assert isinstance(value, dict), f"A {node_type} should have a dict value but got {type(value)}"

        # Decimal support
        if dtype.startswith("{'Decimal'") or dtype == "Decimal":
            decimal_value = value["Decimal"]
            assert isinstance(decimal_value, list), (
                f"A {dtype} should be a two or three member list but got {type(decimal_value)}"
            )
            assert 2 <= len(decimal_value) <= 3, (
                f"A {dtype} should be a two or three member list but got {len(decimal_value)} member list"
            )
            return str(Decimal(decimal_value[0]) / Decimal(10 ** decimal_value[-1]))

        # Datetime with microseconds since epoch
        if dtype.startswith("{'Datetime'") or dtype == "Datetime":
            micros = value["Datetime"]
            assert isinstance(micros, list), f"A {dtype} should be a one member list but got {type(micros)}"
            dt_timestamp = datetime.datetime.fromtimestamp(micros[0] / 1_000_000, tz=datetime.timezone.utc)
            return f"'{dt_timestamp!s}'::TIMESTAMP"

        # Match simple numeric/boolean types
        if dtype in (
            "Int8",
            "Int16",
            "Int32",
            "Int64",
            "Int128",
            "UInt8",
            "UInt16",
            "UInt32",
            "UInt64",
            "UInt128",
            "Float32",
            "Float64",
            "Boolean",
        ):
            return str(value[dtype])

        # Time type
        if dtype == "Time":
            nanoseconds = value["Time"]
            assert isinstance(nanoseconds, int), f"A {dtype} should be an int but got {type(nanoseconds)}"
            seconds = nanoseconds // 1_000_000_000
            microseconds = (nanoseconds % 1_000_000_000) // 1_000
            dt_time = (datetime.datetime.min + datetime.timedelta(seconds=seconds, microseconds=microseconds)).time()
            return f"'{dt_time}'::TIME"

        # Date type
        if dtype == "Date":
            days_since_epoch = value["Date"]
            assert isinstance(days_since_epoch, (float, int)), (
                f"A {dtype} should be a number but got {type(days_since_epoch)}"
            )
            date = datetime.date(1970, 1, 1) + datetime.timedelta(days=days_since_epoch)
            return f"'{date}'::DATE"

        # Binary type
        if dtype == "Binary":
            bin_value = value["Binary"]
            assert isinstance(bin_value, list), f"A {dtype} should be a list but got {type(bin_value)}"
            binary_data = bytes(bin_value)
            escaped = "".join(f"\\x{b:02x}" for b in binary_data)
            return f"'{escaped}'::BLOB"

        # String type
        if dtype == "String" or dtype == "StringOwned":
            # Some new formats may store directly under StringOwned
            string_val = value.get("StringOwned", value.get("String", None))
            # the string must be a string constant
            return str(duckdb.ConstantExpression(string_val))

        msg = f"Unsupported scalar type {dtype!s}, with value {value}"
        raise NotImplementedError(msg)

    msg = f"Node type: {node_type} is not implemented. {tree[node_type]}"
    raise NotImplementedError(msg)


def duckdb_source(relation: duckdb.DuckDBPyRelation, schema: pl.schema.Schema) -> pl.LazyFrame:
    """A polars IO plugin for DuckDB."""

    def source_generator(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        duck_predicate = None
        fallback_predicate = None
        relation_final = relation
        if with_columns is not None:
            cols = ",".join(map(_escape_sql_identifier, with_columns))
            relation_final = relation_final.project(cols)
        if n_rows is not None:
            relation_final = relation_final.limit(n_rows)
        if predicate is not None:
            # Strip any dynamic-predicate hint (see the dynamic-predicate note
            # above); the real predicate must still be applied.
            tree = json.loads(predicate.meta.serialize(format="json"))
            real_tree, had_dynamic = _strip_dynamic_predicates(tree)
            if real_tree is not None:
                # We have a real predicate; if possible, push it down to DuckDB.
                duck_predicate = _tree_to_sql_expression(real_tree)
                if duck_predicate is None:
                    # Could not push it down: re-apply it polars-side. Rebuild the
                    # expression from the stripped tree so we never hand polars the
                    # dynamic node it cannot lower.
                    fallback_predicate = _expression_from_tree(real_tree) if had_dynamic else predicate
        # Try to pushdown filter, if one exists
        if duck_predicate is not None:
            relation_final = relation_final.filter(duck_predicate)
        results = relation_final.to_arrow_reader() if batch_size is None else relation_final.to_arrow_reader(batch_size)

        for record_batch in iter(results.read_next_batch, None):
            if fallback_predicate is not None:
                # We have a predicate, but did not manage to push it down, we fallback here
                yield pl.from_arrow(record_batch).filter(fallback_predicate)  # type: ignore[arg-type,misc,unused-ignore]
            else:
                yield pl.from_arrow(record_batch)  # type: ignore[misc,unused-ignore]

    return register_io_source(source_generator, schema=schema, is_pure=True)
