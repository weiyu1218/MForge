"""SQLite-backed persistence for runs, designs and known-molecule catalog.

Single-file DB at $MF_DB_PATH (default ./data/moleculeforge.db) so the app
survives restarts without docker. Schema is intentionally small and readable.
"""
from __future__ import annotations

import json
import os
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

_DB_PATH_ENV = "MF_DB_PATH"
_DEFAULT_DB = "/workspace/MForge/moleculeforge/data/moleculeforge.db"

_lock = threading.RLock()


def db_path() -> str:
    return os.environ.get(_DB_PATH_ENV, _DEFAULT_DB)


def _connect() -> sqlite3.Connection:
    path = db_path()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path, timeout=30, isolation_level=None, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL;")
    conn.execute("PRAGMA foreign_keys=ON;")
    return conn


@contextmanager
def cursor() -> Iterator[sqlite3.Cursor]:
    with _lock:
        conn = _connect()
        try:
            cur = conn.cursor()
            yield cur
        finally:
            conn.close()


SCHEMA = """
CREATE TABLE IF NOT EXISTS projects (
    project_id      TEXT PRIMARY KEY,
    name            TEXT NOT NULL,
    description     TEXT DEFAULT '',
    created_at      TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS runs (
    run_id          TEXT PRIMARY KEY,
    project_id      TEXT,
    intent          TEXT NOT NULL,
    objectives      TEXT NOT NULL,
    status          TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    finished_at     TEXT,
    summary         TEXT,
    devices_used    TEXT,
    n_candidates    INTEGER DEFAULT 0,
    n_novel         INTEGER DEFAULT 0,
    n_known         INTEGER DEFAULT 0,
    FOREIGN KEY (project_id) REFERENCES projects(project_id) ON DELETE SET NULL
);

CREATE TABLE IF NOT EXISTS reasoning_steps (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL,
    step_index      INTEGER NOT NULL,
    stage           TEXT NOT NULL,
    title           TEXT NOT NULL,
    detail          TEXT,
    payload         TEXT,
    timestamp       TEXT NOT NULL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS run_results (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id          TEXT NOT NULL,
    rank            INTEGER NOT NULL,
    smiles          TEXT NOT NULL,
    canonical_smiles TEXT NOT NULL,
    inchi_key       TEXT,
    is_novel        INTEGER NOT NULL,
    known_match     TEXT,
    properties      TEXT NOT NULL,
    pareto_optimal  INTEGER DEFAULT 0,
    composite_score REAL,
    FOREIGN KEY (run_id) REFERENCES runs(run_id) ON DELETE CASCADE
);

CREATE TABLE IF NOT EXISTS known_molecules (
    inchi_key       TEXT PRIMARY KEY,
    canonical_smiles TEXT NOT NULL,
    name            TEXT NOT NULL,
    drugbank_id     TEXT,
    indications     TEXT,
    target          TEXT,
    note            TEXT
);

CREATE INDEX IF NOT EXISTS idx_runs_project ON runs(project_id);
CREATE INDEX IF NOT EXISTS idx_steps_run    ON reasoning_steps(run_id, step_index);
CREATE INDEX IF NOT EXISTS idx_results_run  ON run_results(run_id, rank);
CREATE INDEX IF NOT EXISTS idx_known_smiles ON known_molecules(canonical_smiles);
"""


def init_db() -> None:
    """Create schema and seed the known-molecule catalog if empty."""
    with cursor() as cur:
        for stmt in SCHEMA.split(";"):
            stmt = stmt.strip()
            if stmt:
                cur.execute(stmt + ";")
        n = cur.execute("SELECT COUNT(*) FROM known_molecules").fetchone()[0]
    if n == 0:
        _seed_known_molecules()


def insert_project(project_id: str, name: str, description: str, created_at: str) -> None:
    with cursor() as cur:
        cur.execute(
            "INSERT OR REPLACE INTO projects (project_id, name, description, created_at) VALUES (?,?,?,?)",
            (project_id, name, description, created_at),
        )


def list_projects() -> list[dict]:
    with cursor() as cur:
        rows = cur.execute(
            "SELECT * FROM projects ORDER BY created_at DESC LIMIT 200"
        ).fetchall()
    return [dict(r) for r in rows]


def insert_run(run: dict) -> None:
    with cursor() as cur:
        cur.execute(
            """
            INSERT OR REPLACE INTO runs
            (run_id, project_id, intent, objectives, status, created_at, finished_at,
             summary, devices_used, n_candidates, n_novel, n_known)
            VALUES (?,?,?,?,?,?,?,?,?,?,?,?)
            """,
            (
                run["run_id"], run.get("project_id"), run["intent"],
                json.dumps(run.get("objectives") or {}),
                run["status"], run["created_at"], run.get("finished_at"),
                run.get("summary"),
                json.dumps(run.get("devices_used") or []),
                int(run.get("n_candidates") or 0),
                int(run.get("n_novel") or 0),
                int(run.get("n_known") or 0),
            ),
        )


def update_run(run_id: str, **fields: Any) -> None:
    if not fields:
        return
    cols = []
    vals = []
    for k, v in fields.items():
        cols.append(f"{k}=?")
        if isinstance(v, (dict, list)):
            v = json.dumps(v)
        vals.append(v)
    vals.append(run_id)
    with cursor() as cur:
        cur.execute(f"UPDATE runs SET {', '.join(cols)} WHERE run_id=?", vals)


def get_run(run_id: str) -> dict | None:
    with cursor() as cur:
        row = cur.execute("SELECT * FROM runs WHERE run_id=?", (run_id,)).fetchone()
    if not row:
        return None
    out = dict(row)
    out["objectives"] = json.loads(out.get("objectives") or "{}")
    out["devices_used"] = json.loads(out.get("devices_used") or "[]")
    return out


def list_runs(limit: int = 100) -> list[dict]:
    with cursor() as cur:
        rows = cur.execute(
            "SELECT * FROM runs ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["objectives"] = json.loads(d.get("objectives") or "{}")
        d["devices_used"] = json.loads(d.get("devices_used") or "[]")
        out.append(d)
    return out


def insert_reasoning_step(
    run_id: str, step_index: int, stage: str, title: str,
    detail: str | None, payload: dict | None, timestamp: str,
) -> None:
    with cursor() as cur:
        cur.execute(
            """
            INSERT INTO reasoning_steps
            (run_id, step_index, stage, title, detail, payload, timestamp)
            VALUES (?,?,?,?,?,?,?)
            """,
            (run_id, step_index, stage, title, detail,
             json.dumps(payload or {}), timestamp),
        )


def get_reasoning_steps(run_id: str) -> list[dict]:
    with cursor() as cur:
        rows = cur.execute(
            "SELECT * FROM reasoning_steps WHERE run_id=? ORDER BY step_index",
            (run_id,),
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["payload"] = json.loads(d.get("payload") or "{}")
        out.append(d)
    return out


def insert_run_results(run_id: str, results: list[dict]) -> None:
    if not results:
        return
    with cursor() as cur:
        cur.executemany(
            """
            INSERT INTO run_results
            (run_id, rank, smiles, canonical_smiles, inchi_key,
             is_novel, known_match, properties, pareto_optimal, composite_score)
            VALUES (?,?,?,?,?,?,?,?,?,?)
            """,
            [
                (
                    run_id, r["rank"], r["smiles"], r["canonical_smiles"],
                    r.get("inchi_key"),
                    1 if r.get("is_novel") else 0,
                    json.dumps(r.get("known_match")) if r.get("known_match") else None,
                    json.dumps(r.get("properties") or {}),
                    1 if r.get("pareto_optimal") else 0,
                    r.get("composite_score"),
                )
                for r in results
            ],
        )


def get_run_results(run_id: str) -> list[dict]:
    with cursor() as cur:
        rows = cur.execute(
            "SELECT * FROM run_results WHERE run_id=? ORDER BY rank",
            (run_id,),
        ).fetchall()
    out = []
    for r in rows:
        d = dict(r)
        d["is_novel"] = bool(d["is_novel"])
        d["pareto_optimal"] = bool(d["pareto_optimal"])
        d["properties"] = json.loads(d.get("properties") or "{}")
        if d.get("known_match"):
            d["known_match"] = json.loads(d["known_match"])
        out.append(d)
    return out


# ---------------------------------------------------------------------------
# Known-molecule catalog
# ---------------------------------------------------------------------------


def _seed_known_molecules() -> None:
    """Seed with 80+ well-known drug molecules — used for novelty labelling."""
    try:
        from rdkit import Chem
    except ImportError:
        return

    seeds: list[tuple[str, str, str | None, str | None, str | None]] = [
        # (name, smiles, drugbank_id, indications, target)
        ("Aspirin", "CC(=O)Oc1ccccc1C(=O)O", "DB00945", "analgesic, anti-inflammatory", "COX-1/2"),
        ("Ibuprofen", "CC(C)Cc1ccc(C(C)C(=O)O)cc1", "DB01050", "anti-inflammatory", "COX-1/2"),
        ("Paracetamol", "CC(=O)Nc1ccc(O)cc1", "DB00316", "analgesic", "COX-2"),
        ("Caffeine", "Cn1cnc2c1c(=O)n(C)c(=O)n2C", "DB00201", "stimulant", "Adenosine receptor"),
        ("Nicotine", "CN1CCC[C@H]1c1cccnc1", "DB00184", "stimulant", "nAChR"),
        ("Morphine", "CN1CC[C@]23c4c5ccc(O)c4O[C@H]2[C@@H](O)C=C[C@H]3[C@H]1C5", "DB00295", "analgesic", "Mu opioid"),
        ("Codeine", "COc1ccc2c3c1O[C@H]1[C@@H](O)C=C[C@H]4[C@@H]3CC([C@@H]2C4)N(C)C1", "DB00318", "analgesic", "Mu opioid"),
        ("Ethanol", "CCO", "DB00898", "antiseptic", None),
        ("Methanol", "CO", "DB02327", "industrial solvent", None),
        ("Penicillin G", "CC1(C)S[C@@H]2[C@H](NC(=O)Cc3ccccc3)C(=O)N2[C@H]1C(=O)O", "DB01053", "antibacterial", "PBP"),
        ("Amoxicillin", "CC1(C)S[C@@H]2[C@H](NC(=O)[C@H](N)c3ccc(O)cc3)C(=O)N2[C@H]1C(=O)O", "DB01060", "antibacterial", "PBP"),
        ("Cephalexin", "CC1=C(C(=O)O)N2C(=O)[C@@H](NC(=O)[C@H](N)c3ccccc3)[C@H]2SC1", "DB00567", "antibacterial", "PBP"),
        ("Ciprofloxacin", "O=C(O)c1cn(C2CC2)c2cc(N3CCNCC3)c(F)cc2c1=O", "DB00537", "antibacterial", "DNA gyrase"),
        ("Azithromycin", "CC[C@H]1OC(=O)[C@H](C)[C@@H](O[C@H]2C[C@@](C)(OC)[C@@H](O)[C@H](C)O2)[C@H](C)[C@@H](O[C@@H]2O[C@H](C)C[C@@H]([C@H]2O)N(C)C)[C@@](C)(O)C[C@@H](C)CN(C)[C@H](C)[C@@H](O)[C@]1(C)O", "DB00207", "antibacterial", "50S ribosome"),
        ("Doxycycline", "CC1c2cccc(O)c2C(=O)C2=C(O)[C@]3(O)C(=O)C(C(N)=O)=C(O)[C@@H](N(C)C)[C@@H]3C[C@@H]12", "DB00254", "antibacterial", "30S ribosome"),
        ("Metformin", "CN(C)C(=N)NC(N)=N", "DB00331", "antidiabetic", "AMPK"),
        ("Glipizide", "Cc1ncc(C(=O)NCCc2ccc(S(=O)(=O)NC(=O)NC3CCCCC3)cc2)cn1", "DB01067", "antidiabetic", "K-ATP channel"),
        ("Atorvastatin", "CC(C)c1c(C(=O)Nc2ccccc2)c(-c2ccccc2)c(-c2ccc(F)cc2)n1CC[C@@H](O)C[C@@H](O)CC(=O)O", "DB01076", "hypolipidemic", "HMG-CoA reductase"),
        ("Simvastatin", "CCC(C)(C)C(=O)O[C@H]1C[C@H](C)C=C2C=C[C@H](C)[C@H](CC[C@@H]3C[C@@H](O)CC(=O)O3)[C@@H]12", "DB00641", "hypolipidemic", "HMG-CoA reductase"),
        ("Lisinopril", "NCCCC[C@@H](N[C@@H](CCc1ccccc1)C(=O)O)C(=O)N1CCC[C@H]1C(=O)O", "DB00722", "antihypertensive", "ACE"),
        ("Losartan", "CCCCc1nc(Cl)c(CO)n1Cc1ccc(-c2ccccc2-c2nnn[nH]2)cc1", "DB00678", "antihypertensive", "AT1 receptor"),
        ("Amlodipine", "CCOC(=O)C1=C(COCCN)NC(C)=C(C(=O)OC)C1c1ccccc1Cl", "DB00381", "antihypertensive", "L-type Ca channel"),
        ("Hydrochlorothiazide", "NS(=O)(=O)c1cc2c(cc1Cl)NCNS2(=O)=O", "DB00999", "antihypertensive", "NCC"),
        ("Furosemide", "NS(=O)(=O)c1cc(C(=O)O)c(NCc2ccco2)cc1Cl", "DB00695", "diuretic", "NKCC2"),
        ("Warfarin", "CC(=O)CC(c1ccccc1)c1c(O)c2ccccc2oc1=O", "DB00682", "anticoagulant", "VKORC1"),
        ("Heparin (frag)", "OCC1OC(O)C(O)C1OC1OC(C(=O)O)C(O)C1OS(=O)(=O)O", "DB01109", "anticoagulant", "antithrombin"),
        ("Omeprazole", "COc1ccc2[nH]c(S(=O)Cc3ncc(C)c(OC)c3C)nc2c1", "DB00338", "PPI", "H+/K+-ATPase"),
        ("Ranitidine", "CNC(=C[N+](=O)[O-])NCCSCc1ccc(CN(C)C)o1", "DB00863", "H2 blocker", "H2 receptor"),
        ("Loratadine", "CCOC(=O)N1CCC(=C2c3ccc(Cl)cc3CCc3cccnc32)CC1", "DB00455", "antihistamine", "H1 receptor"),
        ("Diphenhydramine", "CN(C)CCOC(c1ccccc1)c1ccccc1", "DB01075", "antihistamine", "H1 receptor"),
        ("Sildenafil", "CCCc1nn(C)c2c1NC(=NC2=O)c1cc(S(=O)(=O)N2CCN(C)CC2)ccc1OCC", "DB00203", "PDE5 inhibitor", "PDE5"),
        ("Tadalafil", "CN1CC(=O)N2[C@@H](Cc3c2[nH]c2ccccc32)[C@@H]1c1ccc2c(c1)OCO2", "DB00820", "PDE5 inhibitor", "PDE5"),
        ("Diazepam", "CN1C(=O)CN=C(c2ccccc2)c2cc(Cl)ccc21", "DB00829", "anxiolytic", "GABA-A"),
        ("Alprazolam", "Cc1nnc2n1-c1ccc(Cl)cc1C(c1ccccc1)=NC2", "DB00404", "anxiolytic", "GABA-A"),
        ("Fluoxetine", "CNCCC(c1ccc(C(F)(F)F)cc1)Oc1ccccc1", "DB00472", "antidepressant", "SERT"),
        ("Sertraline", "CN[C@H]1CC[C@H](c2ccc(Cl)c(Cl)c2)c2ccccc21", "DB01104", "antidepressant", "SERT"),
        ("Paroxetine", "Fc1ccc([C@@H]2CCNC[C@H]2COc2ccc3OCOc3c2)cc1", "DB00715", "antidepressant", "SERT"),
        ("Citalopram", "N#CC1=CC=C(C2(c3ccc(F)cc3)OCC2CCN(C)C)C=C1", "DB00215", "antidepressant", "SERT"),
        ("Risperidone", "Cc1c(CCN2CCC(c3noc4cc(F)ccc34)CC2)c(=O)n2CCCCc2n1", "DB00734", "antipsychotic", "D2/5HT2A"),
        ("Olanzapine", "CN1CCN(C2=Nc3cc(C)sc3Nc3ccccc32)CC1", "DB00334", "antipsychotic", "D2/5HT2A"),
        ("Quetiapine", "OCCOCCN1CCN(C2=Nc3ccccc3Sc3ccccc32)CC1", "DB01224", "antipsychotic", "D2/5HT2A"),
        ("Levodopa", "N[C@@H](Cc1ccc(O)c(O)c1)C(=O)O", "DB01235", "antiparkinson", "DDC"),
        ("Donepezil", "COc1cc2c(cc1OC)C(=O)C(CC1CCN(Cc3ccccc3)CC1)C2", "DB00843", "antialzheimer", "AChE"),
        ("Memantine", "CC12CC3CC(C1)CC(N)(C3)C2", "DB01043", "antialzheimer", "NMDAR"),
        ("Insulin (frag)", "CC[C@H](C)[C@H](NC(=O)CN)C(=O)O", "DB00030", "antidiabetic", "Insulin receptor"),
        ("Acetaminophen", "CC(=O)Nc1ccc(O)cc1", "DB00316", "analgesic", "COX-2"),
        ("Naproxen", "COc1ccc2cc([C@@H](C)C(=O)O)ccc2c1", "DB00788", "anti-inflammatory", "COX-1/2"),
        ("Diclofenac", "OC(=O)Cc1ccccc1Nc1c(Cl)cccc1Cl", "DB00586", "anti-inflammatory", "COX-1/2"),
        ("Celecoxib", "Cc1ccc(-c2cc(C(F)(F)F)nn2-c2ccc(S(N)(=O)=O)cc2)cc1", "DB00482", "anti-inflammatory", "COX-2"),
        ("Salbutamol", "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1", "DB01001", "bronchodilator", "Beta-2 receptor"),
        ("Albuterol", "CC(C)(C)NCC(O)c1ccc(O)c(CO)c1", "DB01001", "bronchodilator", "Beta-2 receptor"),
        ("Theophylline", "Cn1c(=O)c2[nH]cnc2n(C)c1=O", "DB00277", "bronchodilator", "PDE/AdR"),
        ("Prednisone", "O=C1C=CC2(C)C3CCC4(C)C(C(=O)CO)(O)CCC4C3CCC2C1", "DB00635", "anti-inflammatory", "GR"),
        ("Hydrocortisone", "CC12CCC3C(CCC4=CC(=O)CCC34C)C1CCC2(O)C(=O)CO", "DB00741", "anti-inflammatory", "GR"),
        ("Dexamethasone", "CC1CC2C3CCC4=CC(=O)C=CC4(C)C3(F)C(O)CC2(C)C1(O)C(=O)CO", "DB01234", "anti-inflammatory", "GR"),
        ("Methotrexate", "CN(Cc1cnc2nc(N)nc(N)c2n1)c1ccc(C(=O)N[C@@H](CCC(=O)O)C(=O)O)cc1", "DB00563", "antineoplastic", "DHFR"),
        ("Doxorubicin", "COc1cccc2c1C(=O)c1c(O)c3c(c(O)c1C2=O)C[C@@](O)(C(=O)CO)C[C@@H]3O[C@H]1C[C@H](N)[C@H](O)[C@H](C)O1", "DB00997", "antineoplastic", "Topo II"),
        ("Imatinib", "Cc1ccc(NC(=O)c2ccc(CN3CCN(C)CC3)cc2)cc1Nc1nccc(-c2cccnc2)n1", "DB00619", "antineoplastic", "Bcr-Abl"),
        ("Erlotinib", "COCCOc1cc2ncnc(Nc3cccc(C#C)c3)c2cc1OCCOC", "DB00530", "antineoplastic", "EGFR"),
        ("Gefitinib", "COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN1CCOCC1", "DB00317", "antineoplastic", "EGFR"),
        ("Sorafenib", "CNC(=O)c1cc(Oc2ccc(NC(=O)Nc3ccc(Cl)c(C(F)(F)F)c3)cc2)ccn1", "DB00398", "antineoplastic", "VEGFR/PDGFR"),
        ("Pazopanib", "Cc1ccc(N(C)c2nccc(N(C)c3ccc4[nH]nc(C)c4c3)n2)cc1S(N)(=O)=O", "DB06589", "antineoplastic", "VEGFR"),
        ("Tamoxifen", "CC/C(=C(\\c1ccccc1)c1ccc(OCCN(C)C)cc1)c1ccccc1", "DB00675", "antineoplastic", "ER"),
        ("Olaparib", "O=C1c2ccccc2Cc2cc(CC(=O)N3CCN(C(=O)C4CC4)CC3)ccc21", "DB09074", "antineoplastic", "PARP"),
        ("Adenosine", "Nc1ncnc2c1ncn2[C@@H]1O[C@H](CO)[C@@H](O)[C@H]1O", "DB00640", "antiarrhythmic", "AdR"),
        ("ATP", "Nc1ncnc2c1ncn2[C@@H]1O[C@H](COP(=O)(O)OP(=O)(O)OP(=O)(O)O)[C@@H](O)[C@H]1O", "DB00171", "energy", None),
        ("Glucose", "OC[C@H]1O[C@H](O)[C@H](O)[C@@H](O)[C@@H]1O", "DB00131", None, None),
        ("Cholesterol", "C[C@H](CCCC(C)C)[C@H]1CC[C@H]2[C@@H]3CC=C4CC(O)CC[C@]4(C)[C@H]3CC[C@]12C", "DB04540", None, None),
        ("Thiamine", "Cc1ncc(C[n+]2csc(CCO)c2C)c(N)n1", "DB00152", "vitamin", None),
        ("Vitamin C", "OC1=C(O)C(=O)O[C@@H]1[C@@H](O)CO", "DB00126", "vitamin", None),
        ("Folic Acid", "Nc1nc2ncc(CNc3ccc(C(=O)N[C@@H](CCC(=O)O)C(=O)O)cc3)nc2c(=O)[nH]1", "DB00158", "vitamin", "DHFR"),
        ("Riboflavin", "Cc1cc2nc3c(=O)[nH]c(=O)nc-3n(C[C@H](O)[C@H](O)[C@H](O)CO)c2cc1C", "DB00140", "vitamin", None),
        ("Niacin", "O=C(O)c1cccnc1", "DB00627", "vitamin", None),
        ("Histamine", "NCCc1c[nH]cn1", "DB05381", "biogenic amine", "H receptors"),
        ("Serotonin", "NCCc1c[nH]c2ccc(O)cc12", "DB08839", "biogenic amine", "5HT receptors"),
        ("Dopamine", "NCCc1ccc(O)c(O)c1", "DB00988", "biogenic amine", "D receptors"),
        ("Adrenaline", "CNC[C@@H](O)c1ccc(O)c(O)c1", "DB00668", "biogenic amine", "Adrenergic"),
        ("GABA", "NCCCC(=O)O", "DB02530", "neurotransmitter", "GABA receptors"),
        ("Glutamate", "N[C@@H](CCC(=O)O)C(=O)O", "DB00142", "neurotransmitter", "NMDA/AMPA"),
        ("Salicylic Acid", "OC(=O)c1ccccc1O", "DB00936", "topical", "COX"),
        ("Benzene", "c1ccccc1", None, None, None),
        ("Pyridine", "c1ccncc1", None, None, None),
        ("Sulfanilamide", "Nc1ccc(S(N)(=O)=O)cc1", "DB00259", "antibacterial", "DHPS"),
    ]

    rows = []
    for name, smiles, db_id, ind, target in seeds:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            continue
        canon = Chem.MolToSmiles(mol, canonical=True)
        ikey = Chem.InchiToInchiKey(Chem.MolToInchi(mol)) if hasattr(Chem, "InchiToInchiKey") else None
        if ikey is None:
            try:
                from rdkit.Chem import inchi as _inchi
                ikey = _inchi.MolToInchiKey(mol)
            except Exception:
                ikey = canon  # fallback unique key
        rows.append((ikey, canon, name, db_id, ind, target, ""))

    with cursor() as cur:
        cur.executemany(
            """
            INSERT OR IGNORE INTO known_molecules
            (inchi_key, canonical_smiles, name, drugbank_id, indications, target, note)
            VALUES (?,?,?,?,?,?,?)
            """,
            rows,
        )


def lookup_known(inchi_key: str | None, canonical_smiles: str) -> dict | None:
    with cursor() as cur:
        if inchi_key:
            row = cur.execute(
                "SELECT * FROM known_molecules WHERE inchi_key=?", (inchi_key,)
            ).fetchone()
            if row:
                return dict(row)
        row = cur.execute(
            "SELECT * FROM known_molecules WHERE canonical_smiles=?",
            (canonical_smiles,),
        ).fetchone()
    return dict(row) if row else None


def known_count() -> int:
    with cursor() as cur:
        return int(cur.execute("SELECT COUNT(*) FROM known_molecules").fetchone()[0])
