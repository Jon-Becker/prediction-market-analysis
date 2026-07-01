import argparse  # noqa: D100
import json
import re
import webbrowser
from functools import reduce
from pathlib import Path

from duckdb import DuckDBPyConnection

qgraph_css = """
:root {
  --text-primary-color: #0d0d0d;
  --text-secondary-color: #444;
  --doc-codebox-border-color: #e6e6e6;
  --doc-codebox-background-color: #f7f7f7;
  --doc-scrollbar-bg: #e6e6e6;
  --doc-scrollbar-slider: #ccc;
  --duckdb-accent: #009982;
  --duckdb-accent-light: #00b89a;
  --card-bg: #fff;
  --border-radius: 8px;
  --shadow: 0 4px 14px rgba(0,0,0,0.05);
}

html, body {
  margin: 0;
  padding: 0;
  font-family: Inter, system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
  color: var(--text-primary-color);
  background: #fafafa;
  line-height: 1.55;
}

.container {
  max-width: 1000px;
  margin: 40px auto;
  padding: 0 20px;
}

header {
  display: flex;
  align-items: center;
  gap: 10px;
  margin-bottom: 5px;
}

header img {
  width: 100px;
  height: 100px;
}

header h1 {
  font-size: 1.5rem;
  font-weight: 600;
  margin: 0;
  color: var(--text-primary-color);
}

/* === Table Styling (DuckDB documentation style, flat header) === */
table {
  border-collapse: collapse;
  width: 100%;
  margin-bottom: 20px;
  text-align: left;
  font-variant-numeric: tabular-nums;
  border: 1px solid var(--doc-codebox-border-color);
  border-radius: var(--border-radius);
  overflow: hidden;
  box-shadow: var(--shadow);
  background: var(--card-bg);
}

thead {
  background-color: var(--duckdb-accent);
  color: white;
}

th, td {
  padding: 10px 12px;
  font-size: 14px;
  vertical-align: top;
}

th {
  font-weight: 700;
}

tbody tr {
  border-bottom: 1px solid var(--doc-codebox-border-color);
}

tbody tr:last-child td {
  border-bottom: none;
}

tbody tr:hover {
  background: var(--doc-codebox-border-color);
}

tbody tr.phase-details-row {
  border-bottom: none;
}

tbody tr.phase-details-row:hover {
  background: transparent;
}

tbody tr.phase-details-row details summary {
  font-size: 12px;
  padding: 4px 0;
}

tbody tr.phase-details-row details[open] summary {
  margin-bottom: 4px;
}

/* === Chart/Card Section === */
.chart {
  padding: 20px;
  border: 1px solid var(--doc-codebox-border-color);
  border-radius: var(--border-radius);
  background: var(--card-bg);
  box-shadow: var(--shadow);
  overflow: visible;
}

/* === Tree Layout Styling === */
.tf-tree {
  overflow-x: visible;
  overflow-y: visible;
  padding-top: 20px;
}

.tf-nc {
  background: var(--card-bg);
  border: 1px solid var(--doc-codebox-border-color);
  border-radius: var(--border-radius);
  padding: 6px;
  display: inline-block;
}

.node-body {
  font-size: 13px;
  text-align: left;
  padding: 10px;
  white-space: nowrap;
}

.node-body p {
  margin: 2px 0;
}

.node-details {
  white-space: nowrap;
  overflow: visible;
  display: inline-block;
}

/* === Metric Boxes === */
.chart .metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 16px;
  margin-bottom: 20px;
}

.chart .metric-box {
  background: var(--card-bg);
  border: 1px solid var(--doc-codebox-border-color);
  border-radius: var(--border-radius);
  box-shadow: var(--shadow);
  padding: 12px 16px;
  text-align: center;
  transition: transform 0.2s ease, box-shadow 0.2s ease;
}

.chart .metric-box:hover {
  transform: translateY(-2px);
  box-shadow: 0 6px 18px rgba(0, 0, 0, 0.08);
}

.chart .metric-title {
  font-size: 13px;
  color: var(--text-secondary-color);
  margin-bottom: 4px;
  text-transform: uppercase;
  letter-spacing: 0.5px;
}

.chart .metric-value {
  font-size: 18px;
  font-weight: 600;
  color: var(--duckdb-accent);
}


/* === SQL Query Block === */
.chart.sql-block {
  background: var(--doc-codebox-background-color);
  border: 1px solid var(--doc-codebox-border-color);
  border-radius: var(--border-radius);
  box-shadow: var(--shadow);
  padding: 16px;
  overflow-x: auto;
  margin-top: 20px;
}

.chart.sql-block pre {
  margin: 0;
  font-family: "JetBrains Mono", "Fira Code", Consolas, monospace;
  font-size: 13.5px;
  line-height: 1.5;
  color: var(--text-primary-color);
  white-space: pre;
}

.chart.sql-block code {
  color: var(--duckdb-accent);
  font-weight: 500;
}


/* === Links, Typography, and Consistency === */
a {
  color: var(--duckdb-accent);
  text-decoration: underline;
  transition: color 0.3s;
}

a:hover {
  color: black;
}

strong {
  font-weight: 600;
}

/* === Dark Mode Support === */
@media (prefers-color-scheme: dark) {
  :root {
    --text-primary-color: #e6e6e6;
    --text-secondary-color: #b3b3b3;
    --doc-codebox-border-color: #2a2a2a;
    --doc-codebox-background-color: #1e1e1e;
    --card-bg: #111;
  }
  body {
    background: #0b0b0b;
  }
  thead {
    background-color: var(--duckdb-accent);
  }
  tbody tr:hover {
    background: #222;
  }
  
  /* Fix tree node text visibility in dark mode */
  .tf-nc .node-body,
  .tf-nc .node-body p,
  .tf-nc .node-details {
    color: #1a1a1a !important;
  }
  
  /* Fix metric title visibility in dark mode */
  .chart .metric-title {
    color: #b3b3b3;
  }
}
"""  # noqa: W293


class NodeTiming:  # noqa: D101
    def __init__(self, phase: str, time: float, depth: int) -> None:  # noqa: D107
        self.phase = phase
        self.time = time
        self.depth = depth
        # percentage is determined later.
        self.percentage = 0

    def calculate_percentage(self, total_time: float) -> None:  # noqa: D102
        self.percentage = self.time / total_time

    def combine_timing(self, r: "NodeTiming") -> "NodeTiming":  # noqa: D102
        # TODO: can only add timings for same-phase nodes  # noqa: TD002, TD003
        total_time = self.time + r.time
        return NodeTiming(self.phase, total_time, self.depth)


class AllTimings:  # noqa: D101
    def __init__(self) -> None:  # noqa: D107
        self.phase_to_timings = {}

    def add_node_timing(self, node_timing: NodeTiming) -> None:  # noqa: D102
        if node_timing.phase in self.phase_to_timings:
            self.phase_to_timings[node_timing.phase].append(node_timing)
        else:
            self.phase_to_timings[node_timing.phase] = [node_timing]

    def get_phase_timings(self, phase: str) -> list[NodeTiming]:  # noqa: D102
        return self.phase_to_timings[phase]

    def get_summary_phase_timings(self, phase: str) -> NodeTiming:  # noqa: D102
        return reduce(NodeTiming.combine_timing, self.phase_to_timings[phase])

    def get_phases(self) -> list[NodeTiming]:  # noqa: D102
        phases = list(self.phase_to_timings.keys())
        phases.sort(key=lambda x: (self.get_summary_phase_timings(x)).time)
        phases.reverse()
        return phases

    def get_sum_of_all_timings(self) -> float:  # noqa: D102
        total_timing_sum = 0
        for phase in self.phase_to_timings:
            total_timing_sum += self.get_summary_phase_timings(phase).time
        return total_timing_sum


def open_utf8(fpath: str, flags: str) -> object:  # noqa: D103
    return Path(fpath).open(mode=flags, encoding="utf8")


class ProfilingInfo:  # noqa: D101
    def __init__(self, conn: DuckDBPyConnection | None = None, from_file: str | None = None) -> None:  # noqa: D107
        self.conn = conn
        self.from_file = from_file

    def to_json(self) -> str:  # noqa: D102
        if self.from_file is not None:
            with open_utf8(self.from_file, "r") as f:
                return f.read()

        return self.conn.get_profiling_information(format="json")

    def to_pydict(self) -> dict:  # noqa: D102
        return json.loads(self.to_json())

    def to_html(self, output_file: str = "profile.html") -> str:  # noqa: D102
        profiling_info_text = self.to_json()
        html_output = self._translate_json_to_html(input_text=profiling_info_text, output_file=output_file)
        return html_output

    def _get_child_timings(self, top_node: object, query_timings: object, depth: int = 0) -> str:
        node_timing = NodeTiming(top_node["operator_type"], float(top_node["operator_timing"]), depth)
        query_timings.add_node_timing(node_timing)
        for child in top_node["children"]:
            self._get_child_timings(child, query_timings, depth + 1)

    @staticmethod
    def _get_f7fff0_shade_hex(fraction: float) -> str:
        """Returns a shade between very light (#f7fff0) and a slightly darker green-yellow,
        depending on the fraction (0..1).
        """  # noqa: D205
        fraction = max(0, min(1, fraction))

        # Define RGB for light and dark end
        light_color = (247, 255, 240)  # #f7fff0
        dark_color = (200, 255, 150)  # slightly darker/more saturated green-yellow

        # Interpolate RGB channels
        r = int(light_color[0] + (dark_color[0] - light_color[0]) * fraction)
        g = int(light_color[1] + (dark_color[1] - light_color[1]) * fraction)
        b = int(light_color[2] + (dark_color[2] - light_color[2]) * fraction)

        return f"#{r:02x}{g:02x}{b:02x}"

    def _get_node_body(
        self, name: str, result: str, cpu_time: float, card: int, est: int, result_size: int, extra_info: str
    ) -> str:
        """Generate the HTML body for a single node in the tree."""
        node_style = f"background-color: {self._get_f7fff0_shade_hex(float(result) / cpu_time)};"
        new_name = "BRIDGE" if (name == "INVALID") else name.replace("_", " ")
        formatted_num = f"{float(result):.4f}"

        body = f'<span class="tf-nc" style="{node_style}">'
        body += '<div class="node-body">'
        body += f"<p><b>{new_name}</b></p>"
        if result_size > 0:
            body += f"<p>time: {formatted_num}s</p>"
            body += f"<p>cardinality: {card}</p>"
            body += f"<p>estimate: {est}</p>"
            body += f"<p>result size: {result_size} bytes</p>"
        body += "<details>"
        body += "<summary>Extra info</summary>"
        body += '<div class="node-details">'
        body += f"<p>{extra_info}</p>"
        # TODO: Expand on timing. Usually available from a detailed profiling  # noqa: TD002, TD003
        body += "</div>"
        body += "</details>"
        body += "</div>"
        body += "</span>"
        return body

    def _generate_tree_recursive(self, json_graph: object, cpu_time: float) -> str:
        node_prefix_html = "<li>"
        node_suffix_html = "</li>"

        extra_info = ""
        estimate = 0
        for key in json_graph["extra_info"]:
            value = json_graph["extra_info"][key]
            if key == "Estimated Cardinality":
                estimate = int(value)
            else:
                extra_info += f"{key}: {value} <br>"

        # get rid of some typically long names
        extra_info = re.sub(r"__internal_\s*", "__", extra_info)
        extra_info = re.sub(r"compress_integral\s*", "compress", extra_info)

        node_body = self._get_node_body(
            json_graph["operator_type"],
            json_graph["operator_timing"],
            cpu_time,
            json_graph["operator_cardinality"],
            estimate,
            json_graph["result_set_size"],
            re.sub(r",\s*", ", ", extra_info),
        )

        children_html = ""
        if len(json_graph["children"]) >= 1:
            children_html += "<ul>"
            for child in json_graph["children"]:
                children_html += self._generate_tree_recursive(child, cpu_time)
            children_html += "</ul>"
        return node_prefix_html + node_body + children_html + node_suffix_html

    # For generating the table in the top left with expandable phases
    def _generate_timing_html(self, graph_json: object, query_timings: object) -> object:
        """Generates timing HTML table with expandable phases."""
        json_graph = json.loads(graph_json)
        self._gather_timing_information(json_graph, query_timings)
        table_head = """
      <table>
        <thead>
          <tr>
            <th>Phase</th>
            <th>Time (s)</th>
            <th>Percentage</th>
          </tr>
        </thead>"""

        table_body = "<tbody>"
        table_end = "</tbody></table>"

        execution_time = query_timings.get_sum_of_all_timings()

        all_phases = query_timings.get_phases()
        query_timings.add_node_timing(NodeTiming("Execution Time (CPU)", execution_time, None))
        all_phases = ["Execution Time (CPU)", *all_phases]

        for phase in all_phases:
            summarized_phase = query_timings.get_summary_phase_timings(phase)
            summarized_phase.calculate_percentage(execution_time)
            phase_column = f"<b>{phase}</b>" if phase == "Execution Time (CPU)" else phase

            # Main phase row
            table_body += f"""
      <tr>
          <td>{phase_column}</td>
                <td>{round(summarized_phase.time, 8)}</td>
                <td>{str(summarized_phase.percentage * 100)[:6]}%</td>
        </tr>
    """

            # Add expandable details for individual nodes (except for Execution Time)
            if phase != "Execution Time (CPU)":
                phase_timings = query_timings.get_phase_timings(phase)
                if len(phase_timings) > 1:  # Only show details if there are multiple nodes
                    table_body += f"""
        <tr class="phase-details-row">
            <td colspan="3">
                <details>
                    <summary style="cursor: pointer; padding: 4px 0; color: var(--text-secondary-color);">
                        Show {len(phase_timings)} nodes
                    </summary>
                    <table style="margin: 8px 0; width: 100%; border: none; box-shadow: none;">
                        <tbody>
    """
                    for node_timing in sorted(phase_timings, key=lambda x: x.time, reverse=True):
                        node_timing.calculate_percentage(execution_time)
                        depth_indent = "&nbsp;" * (node_timing.depth * 4)
                        table_body += f"""
                            <tr style="background: var(--doc-codebox-background-color);">
                                <td style="padding: 4px 12px; border: none;">{depth_indent}↳ Depth {node_timing.depth}</td>
                                <td style="padding: 4px 12px; border: none;">{round(node_timing.time, 8)}</td>
                                <td style="padding: 4px 12px; border: none;">{str(node_timing.percentage * 100)[:6]}%</td>
                            </tr>
    """  # noqa: E501
                    table_body += """
                        </tbody>
                    </table>
                </details>
            </td>
        </tr>
    """

        table_body += table_end
        return table_head + table_body

    @staticmethod
    def _generate_metric_grid_html(graph_json: str) -> str:
        json_graph = json.loads(graph_json)
        metrics = {
            "Execution Time (s)": f"{float(json_graph.get('latency', 'N/A')):.4f}",
            "Total GB Read": f"{float(json_graph.get('total_bytes_read', 'N/A')) / (1024**3):.4f}"
            if json_graph.get("total_bytes_read", "N/A") != "N/A"
            else "N/A",
            "Total GB Written": f"{float(json_graph.get('total_bytes_written', 'N/A')) / (1024**3):.4f}"
            if json_graph.get("total_bytes_written", "N/A") != "N/A"
            else "N/A",
            "Peak Memory (GB)": f"{float(json_graph.get('system_peak_buffer_memory', 'N/A')) / (1024**3):.4f}"
            if json_graph.get("system_peak_buffer_memory", "N/A") != "N/A"
            else "N/A",
            "Rows Scanned": f"{json_graph.get('cumulative_rows_scanned', 'N/A'):,}"
            if json_graph.get("cumulative_rows_scanned", "N/A") != "N/A"
            else "N/A",
        }
        metric_grid_html = """<div class="metrics-grid">"""
        for key in metrics:
            metric_grid_html += f"""
            <div class="metric-box">
                <div class="metric-title">{key}</div>
                <div class="metric-value">{metrics[key]}</div>
            </div>
            """
        metric_grid_html += "</div>"
        return metric_grid_html

    @staticmethod
    def _generate_sql_query_html(graph_json: str) -> str:
        json_graph = json.loads(graph_json)
        sql_query = json_graph.get("query_name", "N/A")
        sql_html = f"""
        <details><summary><b>SQL Query</b></summary>
        <div class="chart sql-block">
            <pre><code>
    {sql_query}
            </code></pre>
        </div>
        </details><br>
        """
        return sql_html

    def _generate_tree_html(self, graph_json: object) -> str:
        json_graph = json.loads(graph_json)
        cpu_time = float(json_graph["cpu_time"])
        tree_prefix = '<div class="tf-tree tf-gap-sm"> \n <ul>'
        tree_suffix = "</ul> </div>"
        # first level of json is general overview
        # TODO: make sure json output first level always has only 1 level  # noqa: TD002, TD003
        tree_body = self._generate_tree_recursive(json_graph["children"][0], cpu_time)
        return tree_prefix + tree_body + tree_suffix

    def _generate_ipython(self, json_input: str) -> str:
        from IPython.core.display import HTML

        html_output = self._generate_html(json_input, False)

        return HTML(
            (
                '\n	${CSS}\n	${LIBRARIES}\n	<div class="chart" id="query-profile"></div>\n	${CHART_SCRIPT}\n	'
            )
            .replace("${CSS}", html_output["css"])
            .replace("${CHART_SCRIPT}", html_output["chart_script"])
            .replace("${LIBRARIES}", html_output["libraries"])
        )

    @staticmethod
    def _generate_style_html(graph_json: str, include_meta_info: bool) -> None:  # noqa: FBT001
        treeflex_css = '<link rel="stylesheet" href="https://unpkg.com/treeflex/dist/css/treeflex.css">\n'
        libraries = '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap" rel="stylesheet">\n'  # noqa: E501
        return {"treeflex_css": treeflex_css, "duckdb_css": qgraph_css, "libraries": libraries, "chart_script": ""}

    def _gather_timing_information(self, json: str, query_timings: object) -> None:
        # add up all of the times
        # measure each time as a percentage of the total time.
        # then you can return a list of [phase, time, percentage]
        self._get_child_timings(json["children"][0], query_timings)

    def _translate_json_to_html(
        self, input_file: str | None = None, input_text: str | None = None, output_file: str = "profile.html"
    ) -> None:
        query_timings = AllTimings()
        if input_text is not None:
            text = input_text
        elif input_file is not None:
            with open_utf8(input_file, "r") as f:
                text = f.read()
        else:
            print("please provide either input file or input text")
            exit(1)
        html_output = self._generate_style_html(text, True)
        highlight_metric_grid = self._generate_metric_grid_html(text)
        timing_table = self._generate_timing_html(text, query_timings)
        tree_output = self._generate_tree_html(text)
        sql_query_html = self._generate_sql_query_html(text)
        # finally create and write the html
        with open_utf8(output_file, "w+") as f:
            html = """<!DOCTYPE html>
    <html>
      <head>
      <meta charset="utf-8">
      <meta name="viewport" content="width=device-width">
      <title>Query Profile Graph for Query</title>
      ${TREEFLEX_CSS}
      <style>
        ${DUCKDB_CSS}
        </style>
    </head>
    <body>
        <div class="container">
            <header>
                <img src="https://raw.githubusercontent.com/duckdb/duckdb/refs/heads/main/logo/DuckDB_Logo-horizontal.svg" alt="DuckDB Logo">
                <h1>Query Profile Graph</h1>
            </header>
        <div class="chart" id="query-overview">
            ${METRIC_GRID}
        </div>
      <div class="chart" id="query-profile">
            ${SQL_QUERY}
            ${TIMING_TABLE}
      </div>
      ${TREE}
    </body>
    </html>
    """  # noqa: E501
            html = html.replace("${TREEFLEX_CSS}", html_output["treeflex_css"])
            html = html.replace("${DUCKDB_CSS}", html_output["duckdb_css"])
            html = html.replace("${METRIC_GRID}", highlight_metric_grid)
            html = html.replace("${SQL_QUERY}", sql_query_html)
            html = html.replace("${TIMING_TABLE}", timing_table)
            html = html.replace("${TREE}", tree_output)
            f.write(html)


def main() -> None:  # noqa: D103
    parser = argparse.ArgumentParser(
        prog="Query Graph Generator",
        description="""Given a json profile output, generate a html file showing the query graph and
        timings of operators""",
    )
    parser.add_argument("--profile_input", help="profile input in json")
    parser.add_argument("--out", required=False, default=False)
    parser.add_argument("--open", required=False, action="store_true", default=True)
    args = parser.parse_args()

    input = args.profile_input
    output = args.out
    if not args.out:
        if ".json" in input:
            output = input.replace(".json", ".html")
        else:
            print("please provide profile output in json")
            exit(1)
    else:
        if ".html" in args.out:
            output = args.out
        else:
            print("please provide valid .html file for output name")
            exit(1)

    open_output = args.open
    profiling_info = ProfilingInfo(from_file=input)
    profiling_info.to_html(output_file=output)

    if open_output:
        webbrowser.open(f"file://{Path(output).resolve()}", new=2)


if __name__ == "__main__":
    main()
