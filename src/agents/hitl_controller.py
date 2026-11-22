
def collect_human_decisions(valid_proposals, decisions):
    """
    Apply operation-level human decisions.

    decisions:
        Dictionary mapping operation index to
        'approve' or 'reject'.
    """

    approved = []
    rejected = []
    decision_log = []

    for index, proposal in enumerate(valid_proposals):

        decision = decisions.get(index, "reject")

        if decision not in ("approve", "reject"):
            raise ValueError(
                f"Invalid decision for operation {index}: {decision}"
            )

        log_entry = {
            "operation_id": index + 1,
            "column": proposal.get("column", "contact"),
            "original_value": proposal["original_value"],
            "suggested_value": proposal["suggested_value"],
            "decision": decision
        }

        decision_log.append(log_entry)

        if decision == "approve":
            approved.append(proposal)
        else:
            rejected.append(proposal)

    return approved, rejected, decision_log
