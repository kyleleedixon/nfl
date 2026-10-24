"""Call-type experiment (2023 test, trained on 2022: FTN call tags start in 2022). Never touches 2024+.

Five classes from features_v5 `call`: 0 run, 1 RPO kept, 2 RPO thrown, 3 play-action pass, 4 other pass.
P(RPO call) = p1 + p2 is the play-caller's decision; P(pass) = p2 + p3 + p4 is the outcome the binary model predicts.
Compares against the binary screen predictions (OUT dir, `rpo` arm) and reports RPO-aware scoring.

usage: calltype.py [full|live]
"""
import os
import sys
import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss
sys.path.insert(0, os.path.dirname(__file__))
from backtest_v3 import chosen, CATS, BASEP, params
from features_v4 import CATS as CATS4
from features_v5 import HABIT

FEAT = os.environ.get("FEAT", "data/features_v5.parquet")
OUT = os.environ.get("OUT", "data/screen_rpo")
NAMES = ["run", "RPO kept", "RPO thrown", "PA pass", "other pass"]


def fit(tr, va, te, cols):
    p = {**BASEP, **params()[0], "objective": "multiclass", "num_class": 5, "metric": "multi_logloss", "seed": 0}
    p.pop("is_unbalance", None)
    dtr = lgb.Dataset(tr[cols], tr.call)
    m = lgb.train(p, dtr, 5000, valid_sets=[lgb.Dataset(va[cols], va.call, reference=dtr)],
                  callbacks=[lgb.early_stopping(100, verbose=False)])
    full = pd.concat([tr, va])
    m2 = lgb.train(p, lgb.Dataset(full[cols], full.call), int(m.best_iteration * 1.1))
    return m2.predict(te[cols])


def ll(y, p):
    return -(y * np.log(np.clip(p, 1e-6, 1)) + (1 - y) * np.log(np.clip(1 - p, 1e-6, 1)))


def main(model="full"):
    d = pd.read_parquet(FEAT)
    d = d[(d.is_target == 1) & d.call.notna()].reset_index(drop=True)
    for c in CATS + CATS4:
        d[c] = d[c].astype("category")
    full, live, _ = chosen()
    cols = (full if model == "full" else live) + HABIT
    y22 = d[d.season == 2022]
    tr, va, te = y22[y22.week <= 14], y22[y22.week > 14], d[d.season == 2023].copy()
    P = fit(tr, va, te, cols)
    te["p_rpo"], te["p_pa"], te["p_pass_mc"] = P[:, 1] + P[:, 2], P[:, 3], P[:, 2] + P[:, 3] + P[:, 4]
    print(f"[{model}] 2023 test, {len(te)} plays; multiclass log loss {log_loss(te.call, P, labels=range(5)):.4f}")
    rpo, pa = te.call.isin([1, 2]), te.call.eq(3)
    print(f"RPO call:  AUC {roc_auc_score(rpo, te.p_rpo):.3f}  mean P(RPO) on RPOs {te.p_rpo[rpo].mean():.2f}, "
          f"on the rest {te.p_rpo[~rpo].mean():.3f}; plays with P(RPO)>50%: {(te.p_rpo > .5).sum()} "
          f"({rpo[te.p_rpo > .5].mean():.0%} were RPOs); max P(RPO) {te.p_rpo.max():.2f}")
    for lo in [.1, .2, .3]:
        s = te.p_rpo > lo
        print(f"  P(RPO)>{lo:.0%}: {s.mean():.1%} of plays, {rpo[s].mean():.0%} RPOs, catches {rpo[s].sum() / rpo.sum():.0%} of RPOs")
    print(f"PA call:   AUC {roc_auc_score(pa, te.p_pa):.3f}  mean P(PA) on PA {te.p_pa[pa].mean():.2f}, rest {te.p_pa[~pa].mean():.3f}")

    b = pd.read_parquet(f"{OUT}/2023_{model}_base.parquet").rename(columns={"p": "p_base"})
    r = pd.read_parquet(f"{OUT}/2023_{model}_rpo.parquet").rename(columns={"p": "p_rpo_arm"})
    te = te.merge(b, on=["game_id", "play_id"]).merge(r, on=["game_id", "play_id"])
    te["p_blend"] = (te.p_rpo_arm + te.p_pass_mc) / 2
    rpo = te.call.isin([1, 2])
    print("\nP(pass) log loss / accuracy        all      RPO plays   non-RPO")
    for c, name in [("p_base", "binary, current features"), ("p_rpo_arm", "binary + RPO/PA habits"),
                    ("p_pass_mc", "call-type model (2022 only)"), ("p_blend", "blend of the two above"), ("xpass", "nflfastR xpass")]:
        cells = [f"{ll(g.y, g[c]).mean():.4f}/{((g[c] > .5) == g.y).mean():.3f}" for g in (te, te[rpo], te[~rpo])]
        print(f"  {name:32s} " + "  ".join(cells))

    # RPO-aware scoring: the play-caller chose "RPO", the QB chose pass/run after the snap
    print("\nRead rate (accuracy), scored three ways:")
    for c, name in [("p_base", "model"), ("xpass", "nflfastR xpass")]:
        hit = (te[c] > .5) == te.y
        call_hit = np.where(rpo, te.p_rpo > .5, hit) if c == "p_base" else hit   # RPO counts if we called RPO
        print(f"  {name:15s} as now {hit.mean():.3f} | RPOs excluded {hit[~rpo].mean():.3f} | "
              f"RPO scored as its own call {np.mean(call_hit):.3f}")
    neu = te.neutral.eq(1)
    hit = (te.p_base > .5) == te.y
    print(f"  neutral downs: as now {hit[neu].mean():.3f} | RPOs excluded {hit[neu & ~rpo].mean():.3f}; "
          f"RPOs are {rpo[neu].mean():.1%} of neutral plays vs {rpo.mean():.1%} overall")
    te[["game_id", "play_id", "call", "p_rpo", "p_pa", "p_pass_mc"]].to_parquet(f"{OUT}/2023_{model}_calltype.parquet", index=False)


if __name__ == "__main__":
    main(*sys.argv[1:])
