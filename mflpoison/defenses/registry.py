from mflpoison.core.registry import Registry

from .robust_aggregation import CoordinateMedian, Krum, TrimmedMean, WeightedMean
from .update_filter import NormClipper


AGGREGATOR_REGISTRY = Registry("aggregator")
AGGREGATOR_REGISTRY.register("weighted_mean", WeightedMean)
AGGREGATOR_REGISTRY.register("fedavg", WeightedMean)
AGGREGATOR_REGISTRY.register("coordinate_median", CoordinateMedian)
AGGREGATOR_REGISTRY.register("trimmed_mean", TrimmedMean)
AGGREGATOR_REGISTRY.register("krum", Krum)

UPDATE_FILTER_REGISTRY = Registry("update filter")
UPDATE_FILTER_REGISTRY.register("norm_clipping", NormClipper)
