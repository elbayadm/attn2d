import importlib
import os

from fairseq import registry
# ConvNets:
(build_convnet, register_convnet, CONVNET_REGISTRY) = registry.setup_registry('--convnet')
# Aggregators:
(build_aggregator, register_aggregator, AGGREGATOR_REGISTRY) = registry.setup_registry('--aggregator')

from .masked_convolution import MaskedConvolution
from .pa_controller import PAController

# automatically import any Python files in the criterions/ directory
for file in os.listdir(os.path.dirname(__file__)):
    if file.endswith('.py') and not file.startswith('_'):
        module = file[:file.find('.py')]
        importlib.import_module('examples.pervasive.modules.' + module)
