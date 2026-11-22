
import pandas as pd


def execute_categorical_proposals(
    dataframe,
    column,
    valid_proposals,
    approved_proposals=None
):
    """
    Apply approved categorical value mappings to a copy
    of the dataframe.

    approved_proposals:
        None = execute all valid proposals (Autonomous).
        List = execute only the approved proposals (HITL).
    """

    cleaned_df = dataframe.copy()
    execution_log = []

    if approved_proposals is None:
        proposals_to_execute = valid_proposals
    else:
        approved_keys = {
            (
                str(p["original_value"]).strip().lower(),
                str(p["suggested_value"]).strip().lower()
            )
            for p in approved_proposals
        }

        proposals_to_execute = [
            p for p in valid_proposals
            if (
                str(p["original_value"]).strip().lower(),
                str(p["suggested_value"]).strip().lower()
            ) in approved_keys
        ]

    for proposal in proposals_to_execute:

        original = proposal["original_value"]
        suggested = proposal["suggested_value"]

        mask = (
            cleaned_df[column]
            .astype("string")
            .str.strip()
            .str.lower()
            .eq(str(original).strip().lower())
        )

        affected_rows = int(mask.sum())

        cleaned_df.loc[mask, column] = suggested

        execution_log.append({
            "column": column,
            "original_value": original,
            "suggested_value": suggested,
            "affected_rows": affected_rows,
            "reason": proposal.get("reason"),
            "confidence": proposal.get("confidence")
        })

    return cleaned_df, execution_log
