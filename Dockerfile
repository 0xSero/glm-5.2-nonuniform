FROM voipmonitor/vllm:eldritch-enlightenment-v9e8b0ab-b12x21e5cd4-cu132-20260624
# Non-uniform MoE fork: DeepseekV2MoE reads config.num_routed_experts_per_layer
# (12-line change; see deepseek_v2_nonuniform.patch for the diff)
COPY deepseek_v2_nu.py /opt/venv/lib/python3.12/site-packages/vllm/model_executor/models/deepseek_v2.py
# arXiv:2606.00206 overthinking penalty — enabled via --logits-processors
COPY overthink_logits_processor.py /opt/venv/lib/python3.12/site-packages/overthink_logits_processor.py
