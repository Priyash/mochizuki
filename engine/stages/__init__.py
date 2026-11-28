"""The stages of making a film, chained by what each consumes and produces (chain.py), and what a
stage is (stage.py). The scheduler takes its GPU stages from here; `python -m engine chain` shows
where a chapter, or the whole film, stands."""
from .chain import BY_NAME, CHAIN, FILM_INPUTS, dependencies, gpu_stages, next_gpu_stage, order, report
from .stage import ANIMATIC_GATE, ASSEMBLE, BOARD_GATE, CGI, REFS_GATE, SCHEDULER, Stage

__all__ = ["ANIMATIC_GATE", "ASSEMBLE", "BOARD_GATE", "BY_NAME", "CGI", "CHAIN", "FILM_INPUTS", "REFS_GATE",
           "SCHEDULER", "Stage", "dependencies", "gpu_stages", "next_gpu_stage", "order", "report"]
