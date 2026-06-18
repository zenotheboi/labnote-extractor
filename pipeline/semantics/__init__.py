"""
pipeline.semantics — experiment-specific validator plugins.

Each module registers one family's validator with ``pipeline.registry`` via the
``@plugin`` decorator. Import a semantics module to auto-register it:

    import pipeline.semantics.electrodeposition   # registers "electrodeposition"
    import pipeline.semantics.spectroscopy        # registers "spectroscopy"

The pipeline core and perception layers never need to change when a new
experiment type is added — only a new plugin module.
"""
