from model.sharing_strategy.llama import sharing_strategy as sharing_strategy_llama


SHARING_STRATEGY = {
    "smollm": sharing_strategy_llama,
    "smollm2": sharing_strategy_llama,
}