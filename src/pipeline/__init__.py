"""
Pipeline package for end-to-end fake news detection orchestration.
"""

from src.pipeline.analysis_pipeline import AnalysisPipeline, analyze_input
from src.pipeline.result_builder import ConsensusStatus, HybridResultBuilder

__all__ = [
    "AnalysisPipeline",
    "analyze_input",
    "HybridResultBuilder",
    "ConsensusStatus",
]
