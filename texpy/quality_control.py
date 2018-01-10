"""
Quality control
"""
from typing import List


class QualityControlDecision():
    """
    @should_approve: should we approve this response or not?
    @short_reason: a short reason for update (used when short on space)
    @reason: a longer reason the explanation
    @qualification_update: how much the qualification should be updated by
    This CANNOT be a NamedTuple so that the json encoder will call the custom default()
    function. Otherwise it will just serialize this as a list.
    """
    should_approve: bool
    short_reason: str
    reason: str
    qualification_update: int = 0
    def __init__(
            self,
            should_approve: bool,
            short_reason: str,
            reason: str,
            qualification_update: int = 0,
            ):
        self.should_approve = should_approve
        self.short_reason = short_reason
        self.reason = reason
        self.qualification_update = qualification_update


def generate_explanation(decisions: List[QualityControlDecision], char_limit: int = 0) -> str:
    """
    Generates an explanation given a list of decisions that will fit in the character limit.
    """
    ret = "\n".join(f"* {decision.reason}" for decision in decisions)

    if char_limit == 0 or len(ret) < char_limit:
        return ret

    ret = "\n".join(f"* {decision.short_reason}" for decision in decisions)
    if len(ret) < char_limit:
        return ret
    suffix = "... (for more details contact us)"

    ret = ret[:char_limit - len(suffix)] + suffix

    return ret


__all__ = ["QualityControlDecision", "generate_explanation"]
