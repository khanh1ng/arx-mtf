"""Stage E: Hasbrouck VAR, MID identity tests, CLOSE b0 ladder, IRFs -> results/estimation.json"""
import os, sys, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from multiprocessing import Pool
from arxmtf import config as C, data, estimation as E


def _var(args):
    return E.var_one(*args)


def summarise_var(df):
    out = {}
    for mode in ("mid", "close"):
        for nm in [f"a{i}" for i in range(1, 6)] + [f"b{j}" for j in range(6)] + \
                  [f"c{i}" for i in range(1, 6)] + [f"d{j}" for j in range(1, 6)]:
            b, t = df[f"{mode}_{nm}"], df[f"{mode}_{nm}_t"]
            out[f"{mode}_{nm}"] = dict(mean=b.mean(), cs_t=b.mean() / (b.std(ddof=1) / np.sqrt(len(b))),
                                       med_t=t.median(), sig_pos=(t > 1.96).mean(), sig_neg=(t < -1.96).mean())
        out[f"{mode}_r2_price"] = df[f"{mode}_r2_price"].mean()
        out[f"{mode}_r2_flow"] = df[f"{mode}_r2_flow"].mean()
        out[f"{mode}_r2_flow_max"] = df[f"{mode}_r2_flow"].max()
        out[f"{mode}_n_mean"] = df[f"{mode}_n"].mean()
    out["share_x_pos"] = df.share_x_pos.mean(); out["share_x_zero"] = df.share_x_zero.mean()
    return out


if __name__ == "__main__":
    tics = data.universe()
    with Pool(C.N_WORKERS) as p:
        var = pd.DataFrame(p.map(_var, [(t, True) for t in tics] + [(t, False) for t in tics]))
        ident = pd.DataFrame(p.map(E.identity_one, tics))
        lad = pd.DataFrame(p.map(E.b0_ladder_one, tics))
    os.makedirs(C.RESULTS, exist_ok=True)
    var.to_csv(os.path.join(C.RESULTS, "est_var.csv"), index=False)
    ident.to_csv(os.path.join(C.RESULTS, "est_identity.csv"), index=False)
    lad.to_csv(os.path.join(C.RESULTS, "est_b0_ladder.csv"), index=False)

    vd = var[var.drop_overnight]; vi = var[~var.drop_overnight]
    res = {"n_stocks": len(tics), "var": summarise_var(vd), "var_overnight": summarise_var(vi)}
    irfs = {}
    for mode in ("mid", "close"):
        a = np.r_[0, [vd[f"{mode}_a{i}"].mean() for i in range(1, 6)]]
        b = np.array([vd[f"{mode}_b{j}"].mean() for j in range(6)])
        g = E.irf(a, b, 12); irfs[mode] = g.tolist(); irfs[mode + "_cum"] = float(g.sum())
    res["irf"] = irfs
    res["identity"] = dict(
        b1=ident.b1.mean(), pred_b1=ident.pred_b1.mean(), corr=float(np.corrcoef(ident.b1, ident.pred_b1)[0, 1]),
        b1_ctrl=ident.b1_ctrl.mean(), sig_before=(ident.b1_t.abs() > 1.96).mean(),
        sig_after=(ident.b1_ctrl_t.abs() > 1.96).mean(), sig_after_pos=(ident.b1_ctrl_t > 1.96).mean(),
        sig_after_neg=(ident.b1_ctrl_t < -1.96).mean(), r2=ident.r2.mean(), r2_ctrl=ident.r2_ctrl.mean(),
        b0=ident.b0.mean(), b0_ctrl=ident.b0_ctrl.mean(), corr_x_cl=ident.corr_x_cl.mean(),
        cls_b1=ident.cls_b1.mean())
    M = lad.mean(numeric_only=True)
    res["b0_ladder"] = dict(
        gauss=0.7978845608 * M.sd_oc, Eabs_oc=M.Eabs_oc, sd_oc=M.sd_oc, ratio=M.ratio, exkurt=M.exkurt,
        var_x=M.var_x, p_zero=M.p_zero, b_oc_uni=M.b_oc_uni, b_rc_uni=M.b_rc_uni, b_full=M.b_full,
        pred=M.Eabs_oc / M.var_x, corr=float(np.corrcoef(lad.b_full, lad.Eabs_oc / lad.var_x)[0, 1]),
        med_ratio=float(np.median(lad.b_full / (lad.Eabs_oc / lad.var_x))),
        corr_gap_x=M.corr_gap_x, sd_gap=M.sd_gap)
    json.dump(res, open(os.path.join(C.RESULTS, "estimation.json"), "w"), indent=1, default=float)
    v = res["var"]
    print("n", len(tics), "| MID b0 %.4f b1 %.4f a1 %.4f R2 %.3f | CLOSE b0 %.4f b1 %.4f R2 %.3f | flowR2 %.5f %.5f" % (
        v["mid_b0"]["mean"], v["mid_b1"]["mean"], v["mid_a1"]["mean"], v["mid_r2_price"],
        v["close_b0"]["mean"], v["close_b1"]["mean"], v["close_r2_price"], v["mid_r2_flow"], v["close_r2_flow"]))
    print("IRF cum", round(irfs["mid_cum"], 3), round(irfs["close_cum"], 3))
    print("identity", {k: round(float(x), 4) for k, x in res["identity"].items()})
    print("ladder", {k: round(float(x), 4) for k, x in res["b0_ladder"].items()})
