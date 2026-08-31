from .schemas import ConditionNode, Operand, StrategyDefinition

RSI_OVERSOLD_BOUNCE = StrategyDefinition(
    name="RSI Aşırı Satım Sıçraması",
    direction="long",
    entry=ConditionNode(type="compare", left=Operand(indicator="rsi"), op="lt", right=Operand(value=30)),
    exit=ConditionNode(type="compare", left=Operand(indicator="rsi"), op="gt", right=Operand(value=55)),
    take_profit_pct=4.0,
    stop_loss_pct=3.0,
)

EMA_GOLDEN_CROSS = StrategyDefinition(
    name="EMA Altın Kesişim",
    direction="long",
    entry=ConditionNode(
        type="cross", left=Operand(indicator="ema_fast"), right=Operand(indicator="ema_slow"), direction="above"
    ),
    exit=ConditionNode(
        type="cross", left=Operand(indicator="ema_fast"), right=Operand(indicator="ema_slow"), direction="below"
    ),
    stop_loss_pct=5.0,
)

MACD_MOMENTUM_SHORT = StrategyDefinition(
    name="MACD Momentum Short",
    direction="short",
    entry=ConditionNode(
        type="and",
        conditions=[
            ConditionNode(type="compare", left=Operand(indicator="macd_hist"), op="lt", right=Operand(value=0)),
            ConditionNode(type="compare", left=Operand(indicator="rsi"), op="lt", right=Operand(value=45)),
        ],
    ),
    take_profit_pct=3.0,
    stop_loss_pct=2.5,
)

EXAMPLES: dict[str, StrategyDefinition] = {
    "rsi_oversold_bounce": RSI_OVERSOLD_BOUNCE,
    "ema_golden_cross": EMA_GOLDEN_CROSS,
    "macd_momentum_short": MACD_MOMENTUM_SHORT,
}
