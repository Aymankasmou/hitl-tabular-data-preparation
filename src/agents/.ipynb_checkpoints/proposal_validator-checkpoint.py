
def validate_proposals(proposals, suspicious_values, allowed_values):
    """
    Validate LLM correction proposals without modifying the dataset.

    Returns:
        valid_proposals: proposals that pass validation
        rejected_proposals: proposals rejected with reasons
    """

    valid_proposals = []
    rejected_proposals = []

    # Build a lookup of suspicious values
    suspicious_lookup = {
        str(item["suspicious_value"]).strip().lower()
        for item in suspicious_values
    }

    seen_originals = set()

    for proposal in proposals:

        original = proposal.get("original_value")
        suggested = proposal.get("suggested_value")

        reason = None

        if not isinstance(original, str) or not isinstance(suggested, str):
            reason = "Missing or invalid original/suggested value"

        elif not original.strip() or not suggested.strip():
            reason = "Empty original or suggested value"

        elif original.strip().lower() not in suspicious_lookup:
            reason = "Original value is not in suspicious values"

        elif original.strip().lower() in seen_originals:
            reason = "Duplicate proposal for the same original value"

        elif suggested.strip().lower() not in {
            str(value).strip().lower() for value in allowed_values
        }:
            reason = "Suggested value is not in allowed values"

        elif original.strip().lower() == suggested.strip().lower():
            reason = "No actual correction proposed"

        if reason:
            rejected_proposals.append({
                "proposal": proposal,
                "rejection_reason": reason
            })
            continue

        seen_originals.add(original.strip().lower())

        valid_proposals.append(proposal)

    return valid_proposals, rejected_proposals
