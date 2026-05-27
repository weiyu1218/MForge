"""Stage 2: extracted dict → ChemicalIntentGraph."""
import uuid

from mf_core.types.cig import ChemicalIntentGraph, ObjectiveNode, ObjectiveType

PROPERTY_DEFAULTS = {
    "qed": (ObjectiveType.CONTINUOUS_MAXIMIZE, "rdkit"),
    "sa_score": (ObjectiveType.CONTINUOUS_MINIMIZE, "rdkit"),
    "binding_affinity": (ObjectiveType.CONTINUOUS_MAXIMIZE, "boltz2"),
    "solubility": (ObjectiveType.CONTINUOUS_MAXIMIZE, "admet_ai"),
    "logp": (ObjectiveType.MULTI_CONSTRAINT_SATISFY, "rdkit"),
}


def build_cig(extracted: dict, source: str) -> ChemicalIntentGraph:
    intent_id = f"cig-{uuid.uuid4().hex[:12]}"

    objectives = []
    props = extracted.get("properties", [])

    for prop_info in props:
        prop_name = prop_info["name"]
        obj_type, oracle = PROPERTY_DEFAULTS.get(
            prop_name, (ObjectiveType.CONTINUOUS_MAXIMIZE, "rdkit")
        )
        direction = prop_info.get("direction", "maximize")
        if direction == "minimize":
            obj_type = ObjectiveType.CONTINUOUS_MINIMIZE

        objectives.append(
            ObjectiveNode(
                id=f"obj_{prop_name}",
                name=prop_name,
                type=obj_type,
                oracle=oracle,
                weight=1.0 / max(1, len(props)),
            )
        )

    # Add affinity objective if targets present
    targets = extracted.get("targets", [])
    if targets:
        objectives.append(
            ObjectiveNode(
                id="obj_affinity",
                name="binding_affinity",
                type=ObjectiveType.CONTINUOUS_MAXIMIZE,
                oracle="boltz2",
                weight=1.0 / max(1, len(objectives) + 1),
            )
        )

    # Add ADMET bundle if ADMET constraints present
    admet = extracted.get("admet_constraints", {})
    if admet.get("cyp3a4_ic50_min") or admet.get("oral_bioavailability_min"):
        objectives.append(
            ObjectiveNode(
                id="obj_admet_bundle",
                name="admet_bundle",
                type=ObjectiveType.MULTI_CONSTRAINT_SATISFY,
                oracle="admet_ai",
                weight=1.0 / max(1, len(objectives) + 1),
                constraints={"CYP3A4_IC50": admet.get("cyp3a4_ic50_min", 10.0)},
            )
        )

    # Add FTO objective if IP constraints present
    ip = extracted.get("ip_constraints", {})
    if ip.get("fto_required"):
        objectives.append(
            ObjectiveNode(
                id="obj_fto",
                name="fto",
                type=ObjectiveType.CONSTRAINT,
                oracle="fto_patent",
                weight=1.0 / max(1, len(objectives) + 1),
                pareto_tier=1,
            )
        )

    # Normalize weights
    total_w = sum(o.weight for o in objectives)
    if total_w > 0:
        for o in objectives:
            o.weight = o.weight / total_w

    # Build generative priors from constraints
    generative_priors: dict = {}
    constraints = extracted.get("constraints", {})
    if "max_mw" in constraints:
        generative_priors["mw_range"] = (0, float(constraints["max_mw"]))

    # Build target context
    target_names = [t["name"] for t in targets]
    grounded_uniprot_ids = extracted.get("_grounded_uniprot_ids", [])
    grounded_pdb_ids = extracted.get("_grounded_pdb_ids", [])
    grounding_evidence = extracted.get("_grounding_evidence", [])
    target_context: dict = {}
    if grounded_uniprot_ids:
        target_context["uniprot_ids"] = list(grounded_uniprot_ids)
    elif target_names:
        target_context["uniprot_ids"] = target_names
    if grounded_pdb_ids:
        target_context["pdb_ids"] = list(grounded_pdb_ids)
    if grounding_evidence:
        target_context["grounding_evidence"] = list(grounding_evidence)

    return ChemicalIntentGraph(
        intent_id=intent_id,
        objective_nodes=objectives,
        source_user_input=source,
        target_context=target_context,
        generative_priors=generative_priors,
        created_by="cig_compiler_svc",
    )
