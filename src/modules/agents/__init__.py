REGISTRY = {}

from .rnn_agent import RNNAgent
REGISTRY["rnn"] = RNNAgent

from .updet_agent import UPDeT
REGISTRY['updet'] = UPDeT

from .dasen_agent import DASEN_v1
REGISTRY['dasen_v1'] = DASEN_v1

from .dasen_agent import DASEN_v2
REGISTRY['dasen_v2'] = DASEN_v2

from .dasen_agent import DASEN_v3
REGISTRY['dasen_v3'] = DASEN_v3

from .transformer_agg_agent import TransformerAggregationAgent
REGISTRY['transformer_aggregation'] = TransformerAggregationAgent