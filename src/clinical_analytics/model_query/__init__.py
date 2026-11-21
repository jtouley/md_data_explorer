# SPDX-License-Identifier: MIT
"""Model-writes-the-query: the default question path.

The local model reads the table's columns, descriptions, values and grain plus similar worked
examples and writes the query (a small JSON plan the app compiles to SQL, or SQL itself); the app
runs it in the SQL sandbox, feeds an error back for a bounded number of retries, and shows the SQL
and a one-sentence plain reading with every answer. Statistical tests are a structured call
routed to the tested statistics code (``analysis.compute``), never statistics written in SQL.
"""
