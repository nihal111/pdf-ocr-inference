# Lazy exports so the scheduler/block manager can be imported without torch or
# flash-attn (lets the lab's mock engine run on machines without a GPU).
def __getattr__(name):
    if name == "LLM":
        from nanovllm.llm import LLM
        return LLM
    if name == "SamplingParams":
        from nanovllm.sampling_params import SamplingParams
        return SamplingParams
    raise AttributeError(name)
