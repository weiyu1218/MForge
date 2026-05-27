"""USPTO-MIT + RetroPath reaction tree ingestion pipeline.

Reads USPTO-MIT and RetroPath reaction data, builds AND-OR reaction trees,
and outputs training data for the HUMU route encoder.

Usage:
    python data/ingestion/reaction_ingestion.py \
        --uspto zzzzz/USPTO-MIT/ \
        --retropath zzzzz/RetroPath/ \
        --output data/processing/reaction_trees/
"""

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_uspto_mit(data_dir: str) -> list[dict]:
    """Parse USPTO-MIT reaction SMILES files.

    Expected format: one reaction SMILES per line.
    Reaction SMILES format: reactants >> products
    """
    reactions = []

    for fname in os.listdir(data_dir):
        if not fname.endswith((".txt", ".smi", ".csv")):
            continue
        filepath = os.path.join(data_dir, fname)
        with open(filepath) as f:
            for line_num, line in enumerate(f):
                line = line.strip()
                if not line or line.startswith("#"):
                    continue

                # Handle CSV format
                if "," in line:
                    parts = line.split(",")
                    rxn_smiles = parts[0].strip()
                else:
                    rxn_smiles = line

                if ">>" not in rxn_smiles:
                    continue

                reactants, products = rxn_smiles.split(">>")
                reactions.append({
                    "reactants": [r.strip() for r in reactants.split(".")],
                    "products": [p.strip() for p in products.split(".")],
                    "reaction_smiles": rxn_smiles,
                    "source": "USPTO-MIT",
                })

    logger.info("uspto.parsed", n_reactions=len(reactions))
    return reactions


def parse_retropath(data_dir: str) -> list[dict]:
    """Parse RetroPath reverse reaction data.

    RetroPath outputs paths as rule-based transformations.
    """
    reactions = []

    for fname in os.listdir(data_dir):
        if not fname.endswith((".csv", ".tsv")):
            continue
        filepath = os.path.join(data_dir, fname)
        with open(filepath) as f:
            header = f.readline()  # skip header
            for line in f:
                line = line.strip()
                if not line:
                    continue

                sep = "\t" if "\t" in line else ","
                parts = line.split(sep)
                if len(parts) < 2:
                    continue

                # Try to extract reaction SMILES from common columns
                rxn_smiles = None
                for p in parts:
                    if ">>" in p:
                        rxn_smiles = p.strip()
                        break

                if rxn_smiles is None:
                    continue

                reactants, products = rxn_smiles.split(">>")
                reactions.append({
                    "reactants": [r.strip() for r in reactants.split(".")],
                    "products": [p.strip() for p in products.split(".")],
                    "reaction_smiles": rxn_smiles,
                    "source": "RetroPath",
                })

    logger.info("retropath.parsed", n_reactions=len(reactions))
    return reactions


def build_reaction_tree(reactions: list[dict]) -> dict:
    """Build an AND-OR reaction tree from a list of reactions.

    AND node: a reaction (all reactants needed)
    OR node: a product (any reaction that produces it)

    Returns a tree structure suitable for route encoder training.
    """
    # Product → reactions that produce it
    product_to_rxns: dict[str, list[dict]] = {}
    # Reactant → reactions that consume it
    reactant_to_rxns: dict[str, list[dict]] = {}

    for rxn in reactions:
        for prod in rxn["products"]:
            product_to_rxns.setdefault(prod, []).append(rxn)
        for react in rxn["reactants"]:
            reactant_to_rxns.setdefault(react, []).append(rxn)

    # Build tree from each product as root
    trees = []
    visited = set()

    for target_smiles in product_to_rxns:
        if target_smiles in visited:
            continue
        tree = _build_tree_recursive(target_smiles, product_to_rxns, reactant_to_rxns, visited, depth=0, max_depth=5)
        if tree:
            trees.append(tree)

    return {"n_trees": len(trees), "trees": trees}


def _build_tree_recursive(
    target: str,
    product_to_rxns: dict[str, list[dict]],
    reactant_to_rxns: dict[str, list[dict]],
    visited: set[str],
    depth: int = 0,
    max_depth: int = 5,
) -> dict | None:
    """Recursively build AND-OR tree from target molecule."""
    if depth > max_depth or target in visited:
        return None

    visited.add(target)

    reactions = product_to_rxns.get(target, [])
    if not reactions:
        return {"type": "leaf", "smiles": target}

    or_node = {"type": "or", "smiles": target, "children": []}

    for rxn in reactions[:3]:  # Limit branching factor
        and_node = {
            "type": "and",
            "reaction_smiles": rxn["reaction_smiles"],
            "children": [],
        }
        for reactant in rxn["reactants"]:
            child = _build_tree_recursive(
                reactant, product_to_rxns, reactant_to_rxns,
                visited, depth + 1, max_depth,
            )
            if child:
                and_node["children"].append(child)
            else:
                and_node["children"].append({"type": "leaf", "smiles": reactant})

        or_node["children"].append(and_node)

    return or_node


def save_trees(trees: list[dict], output_dir: str) -> None:
    """Save reaction trees to JSONL format."""
    os.makedirs(output_dir, exist_ok=True)

    with open(os.path.join(output_dir, "reaction_trees.jsonl"), "w") as f:
        for tree in trees:
            f.write(json.dumps(tree) + "\n")

    # Count statistics
    n_and_nodes = 0
    n_or_nodes = 0
    n_leaves = 0
    max_depth = 0

    def _count(node, depth=0):
        nonlocal n_and_nodes, n_or_nodes, n_leaves, max_depth
        if node is None:
            return
        max_depth = max(max_depth, depth)
        if node["type"] == "and":
            n_and_nodes += 1
        elif node["type"] == "or":
            n_or_nodes += 1
        elif node["type"] == "leaf":
            n_leaves += 1
        for child in node.get("children", []):
            _count(child, depth + 1)

    for tree_data in trees:
        for tree in tree_data.get("trees", []):
            _count(tree)

    manifest = {
        "source": "USPTO-MIT + RetroPath",
        "n_trees": len(trees),
        "n_and_nodes": n_and_nodes,
        "n_or_nodes": n_or_nodes,
        "n_leaves": n_leaves,
        "max_depth": max_depth,
    }
    with open(os.path.join(output_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Reaction tree ingestion pipeline")
    parser.add_argument("--uspto", required=True, help="USPTO-MIT data directory")
    parser.add_argument("--retropath", default=None, help="RetroPath data directory")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--max_trees", type=int, default=0)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    all_reactions = []

    # Parse USPTO-MIT
    if os.path.isdir(args.uspto):
        all_reactions.extend(parse_uspto_mit(args.uspto))

    # Parse RetroPath
    if args.retropath and os.path.isdir(args.retropath):
        all_reactions.extend(parse_retropath(args.retropath))

    if not all_reactions:
        logger.error("reaction_ingestion.no_data_found")
        return

    logger.info("reaction_ingestion.total_reactions", n=len(all_reactions))

    # Build trees
    tree_data = build_reaction_tree(all_reactions)

    if args.max_trees > 0:
        tree_data["trees"] = tree_data["trees"][:args.max_trees]

    # Save
    save_trees([tree_data], args.output)

    logger.info("reaction_ingestion.complete", n_trees=tree_data["n_trees"])


if __name__ == "__main__":
    main()
