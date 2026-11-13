"""Read-only files shipped inside the package (config YAML, the synthetic demo cohort).

Read them through ``clinical_analytics.core.resources``; never write here. Anything the app
writes goes to the per-user data home (``clinical_analytics.core.data_home``).
"""
