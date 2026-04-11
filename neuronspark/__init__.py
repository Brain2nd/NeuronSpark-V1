from .configuration_neuronspark import NeuronSparkConfig
from .modeling_neuronspark import NeuronSparkForCausalLM

from transformers import AutoConfig, AutoModelForCausalLM

AutoConfig.register("neuronspark", NeuronSparkConfig)
AutoModelForCausalLM.register(NeuronSparkConfig, NeuronSparkForCausalLM)
