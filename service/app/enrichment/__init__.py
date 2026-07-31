"""Run-scoped detector/tracking/caption enrichment (plan Sec.8/16/18).

Nothing here imports ultralytics/torch/a caption model at module level -- detector.py and
caption.py lazy-import their heavy dependency exactly like app/embedding/text_cpu.py
already does, so DETECTOR_ENRICHMENT_ENABLED=false (the default) never pays for it.
"""
