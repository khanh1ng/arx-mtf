"""Stage W: fill paper/template.tex with numbers and tables from results/ -> paper/main.tex.
No number in the paper is typed by hand: every @@key@@ is computed here from result files."""
import os, sys, json, glob, re
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np, pandas as pd
from arxmtf import config as C, stats as ST
sys.path.insert(0, os.path.join(C.ROOT, "tests"))
from test_regression import legacy_book

R = C.RESULTS
met = pd.read_csv(os.path.join(R, "metrics.csv"), dtype={"cost": str})
ps = pd.read_csv(os.path.join(R, "per_stock.csv"))
est = json.load(open(os.path.join(R, "estimation.json")))
sts = json.load(open(os.path.join(R, "stats.json")))
risk = json.load(open(os.path.join(R, "risk.json")))
anat = json.load(open(os.path.join(R, "anatomy.json")))
wts = json.load(open(os.path.join(R, "weights.json")))
metas = [json.load(open(f)) for f in sorted(glob.glob(os.path.join(C.PERTIC, "*.json")))]
book = np.load(os.path.join(R, "book.npz")); keys = book["keys"]
V = {}


def M(s, per="full", fill="B", cost="0.0", f="breakeven"):
    r = met[(met.strat == s) & (met.period == per) & (met.fill == fill) & (met.cost == cost)]
    assert len(r) == 1, (s, per, fill, cost); return float(r.iloc[0][f])


def n(x, d=2, sign=False):
    s = f"{x:+.{d}f}" if sign else f"{x:.{d}f}"
    return "$" + s.replace("-", "-") + "$"


def pct(x, d=1, sign=True):
    return "$" + (f"{100*x:+.{d}f}" if sign else f"{100*x:.{d}f}") + "\\%$"


def money(x):
    return "\\$" + f"{x:,.0f}".replace(",", "{,}")


def ci(key, stat, d=2, scale=1.0):
    a, b = sts["ci"][key][stat]
    return f"$[{a*scale:.{d}f},\\ {b*scale:.{d}f}]$"


def date(k):
    s = str(int(k // 10000)); return f"{s[:4]}-{s[4:6]}-{s[6:]}"


# ---------------- data and protocol
V["nstocks"] = str(est["n_stocks"])
V["start"], V["end"] = date(keys[0]), date(keys[-1])
V["years"] = f"{len(np.unique(keys // 10000)) / 252:.2f}"
V["hyears"] = f"{len(np.unique(keys[keys // 10000 >= C.LEARN_END] // 10000)) / 252:.2f}"
V["nbars"] = f"{len(keys):,}".replace(",", "{,}")
V["names"] = f"{book['n'].mean():.1f}"
V["share_up"] = pct(est["var"]["share_x_pos"], 1, False); V["share_flat"] = pct(est["var"]["share_x_zero"], 1, False)

# ---------------- accuracy (per-stock means)
def acc_mean(field_fn):
    vals = [field_fn(m) for m in metas]; vals = [h / c for h, c in vals if c]; return float(np.mean(vals))
A5 = lambda mdl, tg: acc_mean(lambda m: m["acc5"][f"{mdl}|{tg}"])
NAT = lambda key, tg: acc_mean(lambda m: m["nat"][key][tg])
V["acc_mid_own"] = n(A5("mid_5m", "mid"), 3); V["acc_mid_cls"] = n(A5("mid_5m", "cls"), 3)
V["acc_mid_oc"] = n(A5("mid_5m", "oc"), 3); V["acc_cls"] = n(A5("cls_5m", "cls"), 3)
V["acc_mid_4h"] = n(NAT("mid_4h", "own"), 3)
V["acc_mid_cls_pct"] = pct(A5("mid_5m", "cls"), 1, False)
V["acc_cls_pct"] = f"{100*A5('cls_5m','cls'):.1f}"
TAU = [nm for nm, _ in C.TAUS]
rows = ["MID, scored on next midpoint return & " + " & ".join(n(NAT(f"mid_{t}", "own"), 3) for t in TAU),
        "MID, scored on next close return & " + " & ".join(n(NAT(f"mid_{t}", "cls"), 3) for t in TAU),
        "CLOSE, scored on next close return & " + " & ".join(n(NAT(f"cls_{t}", "cls"), 3) for t in TAU)]
V["TAB_midacc"] = " \\\\\n".join(rows) + " \\\\"
V["mid_cls_lo"] = n(min(NAT(f"mid_{t}", "cls") for t in TAU), 3); V["mid_cls_hi"] = n(max(NAT(f"mid_{t}", "cls") for t in TAU), 3)

# ---------------- identity and Hasbrouck
I = est["identity"]; Lb = est["b0_ladder"]; VA = est["var"]
V.update(b1=n(I["b1"], 2), b1_ctrl=n(I["b1_ctrl"], 3), id_corr=n(I["corr"], 3), id_pred=n(I["pred_b1"], 2),
         sig_before=pct(I["sig_before"], 0, False), sig_after=pct(I["sig_after"], 0, False),
         r2_before=n(I["r2"], 3), r2_after=n(I["r2_ctrl"], 3), corr_x_cl=n(I["corr_x_cl"], 3),
         cls_b1=n(I["cls_b1"], 3), b0_mid=n(I["b0"], 2), b0_mid_ctrl=n(I["b0_ctrl"], 2),
         b1_removed=pct(1 - I["b1_ctrl"] / I["b1"], 0, False),
         b0_cls=n(VA["close_b0"]["mean"], 2), b0_pred=n(Lb["pred"], 2), b0_corr=n(Lb["corr"], 4),
         eabs=n(Lb["Eabs_oc"], 2), varx=n(Lb["var_x"], 3), pzero=pct(Lb["p_zero"], 1, False),
         flow_r2=n(VA["close_r2_flow"], 4), irf_mid=n(est["irf"]["mid_cum"], 2), irf_cls=n(est["irf"]["close_cum"], 2),
         a1_mid=n(VA["mid_a1"]["mean"], 2), a1_cls=n(VA["close_a1"]["mean"], 3))
def varrow(nm, lab):
    a, b = VA[f"mid_{nm}"], VA[f"close_{nm}"]
    f = lambda d: f"{100*(d['sig_pos']+d['sig_neg']):.1f}"
    return f"{lab} & {n(a['mean'],4)} & ${f(a)}$ & {n(b['mean'],4)} & ${f(b)}$"
V["TAB_var"] = " \\\\\n".join(varrow(k, l) for k, l in [("a1", "$a_1$"), ("a2", "$a_2$"), ("b0", "$b_0$"), ("b1", "$b_1$"),
                                                          ("b2", "$b_2$"), ("d1", "$d_1$")]) + " \\\\"
V["r2p_mid"], V["r2p_cls"] = n(VA["mid_r2_price"], 3), n(VA["close_r2_price"], 3)
V["r2f_mid"], V["r2f_cls"] = n(VA["mid_r2_flow"], 5), n(VA["close_r2_flow"], 5)
g = est["irf"]
V["IRF_mid"] = " ".join(f"({h},{v:.4f})" for h, v in enumerate(g["mid"][:9]))
V["IRF_cls"] = " ".join(f"({h},{v:.4f})" for h, v in enumerate(g["close"][:9]))

# ---------------- 5-minute variants
VAR5 = [("MID, target $\\Delta$MID", "mid|5m", "mid_5m"), ("CLOSE, target $\\Delta$CLOSE", "cls|5m", "cls_5m"),
        ("CLOSE, plain-sign proxy", "var|cls5_sign", "cls5_sign"), ("CLOSE, target OC", "var|oc5_tanh", "oc5_tanh"),
        ("CLOSE, ridge + decay", "var|cls5_rd", "cls5_rd")]
def share_pos(s, col="pnl_B"): q = ps[ps.strat == s]; return float((q[col] > 0).mean())
V["TAB_var5"] = " \\\\\n".join(
    f"{lab} & {n(A5(a,'mid'),3)} & {n(A5(a,'cls'),3)} & {n(A5(a,'oc'),3)} & {n(M(s,f='gross_bps'),3)} & "
    f"{n(M(s,f='sharpe'),2)} & {n(M(s,fill='A'),3)} & {n(M(s),3)} & ${100*share_pos(s):.1f}$"
    for lab, s, a in VAR5) + " \\\\"

# ---------------- main 5m numbers, anatomy, hit/size, fundamental law
for s, tag in (("cls|5m", "f"), ("cls|eq_sign", "es"), ("cls|eq_filt", "ef"), ("cls|4h", "h4"), ("mid|5m", "m5")):
    V[f"{tag}_beA"], V[f"{tag}_beB"], V[f"{tag}_beC"] = (n(M(s, fill=x), 3) for x in "ABC")
    V[f"{tag}_srA"], V[f"{tag}_srB"] = n(M(s, fill="A", f="sharpe"), 2), n(M(s, f="sharpe"), 2)
    V[f"{tag}_bpsB"] = n(M(s, f="gross_bps"), 3)
for s, tag in (("cls|lw_filt", "lf"), ("cls|eq_filt", "efh"), ("cls|eq_sign", "esh")):
    V[f"{tag}_beA"], V[f"{tag}_beB"], V[f"{tag}_beC"] = (n(M(s, "hold", x), 3) for x in "ABC")
    V[f"{tag}_srB"] = n(M(s, "hold", f="sharpe"), 2)
V["f_turn"] = n(M("cls|5m", f="turn"), 2); V["ef_turn"] = n(M("cls|eq_filt", f="turn"), 3)
V["es_turn"] = n(M("cls|eq_sign", f="turn"), 2)
V["es_keep"] = pct(M("cls|eq_sign", f="gross_bps") / M("cls|5m", f="gross_bps"), 0, False)
V["es_turnratio"] = pct(M("cls|eq_sign", f="turn") / M("cls|5m", f="turn"), 0, False)
co = anat["coefficients"]
V["beta_r0"] = n(co["r_t-0"]["mean"], 4); V["beta_r0_neg"] = pct(1 - co["r_t-0"]["pos"], 0, False)
V["gamma_x0_pos"] = pct(co["x_t-0"]["pos"], 0, False); V["n_refits"] = f"{anat['n_refits']:,}".replace(",", "{,}")
def hitrow(lab, s):
    q = ps[ps.strat == s]; p = q.hits.sum() / q.n.sum()
    mh, mm = q.sum_hit.sum() / q.hits.sum(), q.sum_miss.sum() / (q.n.sum() - q.hits.sum())
    a = q.hits / q.n
    return (p, mh, mm, f"{lab} & {n(p,4)} & {n(mh,2)} & {n(mm,2)} & ${100*(a<0.5).mean():.1f}\\%$ & "
            f"${100*((a<0.5)&(q.pnl_B>0)).mean():.1f}\\%$")
H = {s: hitrow(l, s) for l, s in [("CLOSE 5m", "cls|5m"), ("CLOSE MTF, equal, sign", "cls|eq_sign"),
                                   ("CLOSE MTF, equal, filtered", "cls|eq_filt"), ("MID MTF, equal, filtered", "mid|eq_filt")]}
V["TAB_hit"] = " \\\\\n".join(h[3] for h in H.values()) + " \\\\"
p, mh, mm, _ = H["cls|5m"]
V["hit_p"] = n(p, 4); V["hit_part"] = n((p - .5) * (mh + mm), 3); V["size_part"] = n(.5 * (mh - mm), 3)
V["hit_share"] = pct((p - .5) * (mh + mm) / ((p - .5) * (mh + mm) + .5 * (mh - mm)), 0, False)
pm, mhm, mmm, _ = H["mid|eq_filt"]; V["mid_filt_p"] = n(pm, 4)
qf = ps[ps.strat == "cls|eq_filt"]; af = qf.hits / qf.n
V["filt_below"] = pct((af < .5).mean(), 0, False); V["filt_below_pos"] = pct(((af < .5) & (qf.pnl_B > 0)).mean(), 0, False)
FL = sts["fundamental_law"]["cls|5m"]
V.update(ic=n(FL["ic_mean"], 4), rho=n(FL["rho"], 3), neff=f"${FL['n_eff']:.0f}$", sr_stock=n(FL["sr_stock_mean"], 2),
         sr_pred=n(FL["sr_port_pred"], 2), sr_act=n(FL["sr_port_daily"], 2),
         bets=f"{FL['bets_per_year_per_stock']:,.0f}".replace(",", "{,}"))
V["sr_indep"] = n(FL["sr_stock_mean"] * np.sqrt(FL["N"]), 0)

# ---------------- MTF
tot = lambda s, per, c: M(s, per, "B", c, "total")
def mtfrow(lab, s, per="full"):
    return (f"{lab} & {n(M(s,per,f='gross_bps'),3)} & {n(M(s,per,f='sharpe'),2)} & {n(M(s,per,f='turn'),3)} & "
            f"{n(M(s,per,'A'),3)} & {n(M(s,per),3)} & {pct(tot(s,per,'tick'),0)} & {pct(tot(s,per,'ar'),0)}")
rows = [mtfrow(f"{t} only", f"cls|{t}") for t in TAU] + ["\\midrule"]
rows += [mtfrow("MTF, equal, sign", "cls|eq_sign"), mtfrow("MTF, equal, size $|S|$", "cls|eq_size"),
         mtfrow("MTF, equal, filtered", "cls|eq_filt"), "\\midrule",
         "\\multicolumn{8}{@{}l}{\\textit{Out of sample, Jan 2022 -- Jan 2025}}",
         mtfrow("MTF, equal, filtered", "cls|eq_filt", "hold"), mtfrow("MTF, learned, sign", "cls|lw_sign", "hold"),
         mtfrow("MTF, learned, filtered", "cls|lw_filt", "hold")]
V["TAB_mtf"] = " \\\\\n".join(r if r.startswith("\\midrule") else r for r in rows).replace("\\midrule \\\\", "\\midrule") + " \\\\"
wc = wts["cls"]; tw = sum(wc.values())
V["w_cls"] = ", ".join(f"{wc[t]/tw:.2f}" for t in TAU)
V["w_mid_nonzero"] = ", ".join(t for t in TAU if wts["mid"][t] > 0)
Cm = np.array(anat["score_corr"]["cls"])
V["TAB_corr"] = " \\\\\n".join(f"{TAU[i]} & " + " & ".join(("$1$" if j == i else (n(Cm[i, j], 2) if j > i else ""))
                                                           for j in range(6)) for i in range(5)) + " \\\\"
V["corr_adj_lo"] = n(min(Cm[i, i + 1] for i in range(5)), 2); V["corr_adj_hi"] = n(max(Cm[i, i + 1] for i in range(5)), 2)
V["corr_5_4"] = n(Cm[0, 5], 2)
V["h4_active"] = pct(risk["risk"]["cls|4h"]["gross_exposure"], 0, False)

# ---------------- fills
def fillrow(lab, s, per="full"):
    return (f"{lab} & {n(M(s,per,'A'),3)} & {n(M(s,per,'B'),3)} & {n(M(s,per,'C'),3)} & "
            f"{n(M(s,per,'A',f='sharpe'),2)} & {n(M(s,per,'B',f='sharpe'),2)}")
V["TAB_fill"] = " \\\\\n".join([fillrow("CLOSE 5m", "cls|5m"), fillrow("CLOSE MTF, equal, sign", "cls|eq_sign"),
                                 fillrow("CLOSE MTF, equal, filtered", "cls|eq_filt"), fillrow("CLOSE 4h only", "cls|4h"),
                                 fillrow("MID 5m", "mid|5m"), "\\midrule \\multicolumn{6}{@{}l}{\\textit{Out of sample, Jan 2022 -- Jan 2025}}",
                                 fillrow("CLOSE MTF, learned, filtered", "cls|lw_filt", "hold")]).replace("\\textit{Out of sample, Jan 2022 -- Jan 2025}} \\\\", "\\textit{Out of sample, Jan 2022 -- Jan 2025}} \\\\") + " \\\\"
cut = [1 - M(s, per, "B") / M(s, per, "A") for s, per in [("cls|5m", "full"), ("cls|eq_sign", "full"), ("cls|eq_filt", "full"),
                                                          ("cls|lw_filt", "hold"), ("cls|eq_filt", "hold")]]
V["fill_cut_lo"], V["fill_cut_hi"] = pct(min(cut), 0, False), pct(max(cut), 0, False)

# ---------------- costs
q = ps[ps.strat == "cls|5m"]
V["tk_med"], V["ar_med"], V["cs_med"] = n(q.mean_tk.median(), 2), n(q.mean_ar.median(), 2), n(q.mean_cs.median(), 2)
V["tk_lo"], V["tk_hi"] = n(q.mean_tk.quantile(.1), 2), n(q.mean_tk.quantile(.9), 2)
V["ar_lo"], V["ar_hi"] = n(q.mean_ar.quantile(.1), 2), n(q.mean_ar.quantile(.9), 2)
V["cs_lo"], V["cs_hi"] = n(q.mean_cs.quantile(.1), 2), n(q.mean_cs.quantile(.9), 2)
V["cs_top"] = ", ".join(q.sort_values("mean_cs", ascending=False).tic.head(3))
be = q.pnl_B / q.turn; V["corr_be_ar"] = n(np.corrcoef(be, q.mean_ar)[0, 1], 2)
def bktrow(lab, s):
    cells = []
    for b in range(1, 6):
        cells.append(n(M(f"{s}@q{b}"), 3))
    return f"{lab} & " + " & ".join(cells)
def bkttot(lab, s, c):
    return f"{lab} & " + " & ".join(pct(M(f"{s}@q{b}", cost=c, f="total"), 0) for b in range(1, 6))
V["TAB_bkt"] = " \\\\\n".join([
    "\\multicolumn{6}{@{}l}{\\textit{Gross breakeven, next-open fill (bps)}}",
    bktrow("CLOSE 5m", "cls|5m"), bktrow("MTF, equal, filtered", "cls|eq_filt"), bktrow("MTF, learned, filtered", "cls|lw_filt"),
    "\\midrule \\multicolumn{6}{@{}l}{\\textit{Total return after the tick floor}}",
    bkttot("CLOSE 5m", "cls|5m", "tick"), bkttot("MTF, equal, filtered", "cls|eq_filt", "tick"),
    "\\midrule \\multicolumn{6}{@{}l}{\\textit{Total return after Abdi--Ranaldo costs (the ranking variable)}}",
    bkttot("CLOSE 5m", "cls|5m", "ar"), bkttot("MTF, equal, filtered", "cls|eq_filt", "ar")]) + " \\\\"
V["TAB_bkt"] = V["TAB_bkt"].replace("}} \\\\\n", "}} \\\\\n")
V["bkt_f_q1"], V["bkt_f_q5"] = n(M("cls|5m@q1"), 3), n(M("cls|5m@q5"), 3)
V["bkt_ef_q1"], V["bkt_ef_q5"] = n(M("cls|eq_filt@q1"), 3), n(M("cls|eq_filt@q5"), 3)
V["q1_ar_f"] = pct(M("cls|5m@q1", cost="ar", f="total"), 0); V["q1_tk_f"] = pct(M("cls|5m@q1", cost="tick", f="total"), 0)

# ---------------- $1M performance tables
def perfrows(s, per):
    out = []
    for lab, fill, c in [("close fill, gross", "A", "0.0"), ("next open, gross", "B", "0.0"), ("next open, tick floor", "B", "tick"),
                         ("next open, AR", "B", "ar"), ("next open, 0.87 bps", "B", "0.87")]:
        g = lambda f: M(s, per, fill, c, f)
        out.append(f" & {lab} & {money(g('end'))} & {pct(g('total'),1)} & {pct(g('cagr'),1)} & "
                   f"{pct(g('vol'),1,False)} & {n(g('sharpe'),2)} & {pct(g('mdd'),1)} & ${100*g('pos_months'):.0f}\\%$")
    return out
blocks = [("CLOSE 5m", "cls|5m", "full"), ("MTF, equal, sign", "cls|eq_sign", "full"),
          ("MTF, equal, filtered", "cls|eq_filt", "full"), ("MID 5m", "mid|5m", "full")]
rows = []
for lab, s, per in blocks:
    r = perfrows(s, per); r[0] = lab + r[0]; rows += r + ["\\midrule"]
V["TAB_perf"] = " \\\\\n".join(rows[:-1]).replace(" \\\\\n\\midrule \\\\\n", " \\\\\n\\midrule\n") + " \\\\"
V["TAB_perf"] = V["TAB_perf"].replace("\\midrule \\\\", "\\midrule")
r = perfrows("cls|lw_filt", "hold"); V["TAB_perfoos"] = " \\\\\n".join(x[3:] for x in r) + " \\\\"
for s, per, tag in [("cls|5m", "full", "p5"), ("cls|eq_filt", "full", "pef"), ("cls|lw_filt", "hold", "plf"),
                    ("cls|eq_sign", "full", "pes"), ("mid|5m", "full", "pm5"), ("cls|4h", "full", "p4h")]:
    for fill, c, t2 in [("A", "0.0", "ag"), ("B", "0.0", "bg"), ("B", "tick", "bt"), ("B", "ar", "ba"), ("B", "0.87", "b87")]:
        V[f"{tag}_{t2}_end"] = money(M(s, per, fill, c, "end")); V[f"{tag}_{t2}_tot"] = pct(M(s, per, fill, c, "total"), 0)
        V[f"{tag}_{t2}_cagr"] = pct(M(s, per, fill, c, "cagr"), 1); V[f"{tag}_{t2}_mdd"] = pct(M(s, per, fill, c, "mdd"), 1)
        V[f"{tag}_{t2}_sr"] = n(M(s, per, fill, c, "sharpe"), 2); V[f"{tag}_{t2}_mo"] = pct(M(s, per, fill, c, "pos_months"), 0, False)
    V[f"{tag}_vol"] = pct(M(s, per, "B", "0.0", "vol"), 1, False)

# ---------------- confidence intervals, DSR
CIROWS = [("CLOSE 5m", "cls|5m|B|full"), ("MTF, equal, sign", "cls|eq_sign|B|full"), ("MTF, equal, filtered", "cls|eq_filt|B|full"),
          ("CLOSE 4h only", "cls|4h|B|full"), ("MTF, learned, filtered, 2022+", "cls|lw_filt|B|hold")]
V["TAB_ci"] = " \\\\\n".join(f"{lab} & {ci(k,'gross_sharpe')} & {ci(k,'breakeven',3)} & {ci(k,'cagr_tick',1,100)} & "
                             f"{ci(k,'mdd_tick',1,100)}" for lab, k in CIROWS) + " \\\\"
D = sts["deflated"]
V.update(n_here=str(D["n_trials_here"]), n_total=str(D["n_trials_total"]), dsr_best=D["best"].replace("|", " ").replace("@q5", ", widest-spread quintile").replace("cls 5m", "CLOSE 5m"),
         dsr_sr=n(D["best_sharpe"], 2), dsr_85=n(D["by_N"][str(D["n_trials_total"])]["prob"], 3),
         dsr_200=n(D["by_N"]["200"]["prob"], 2), sr0_85=n(D["by_N"][str(D["n_trials_total"])]["expected_max_null"], 2),
         all_neg=pct(D["share_net_tick_negative"], 0, False))
bs = sts["block_sensitivity"]["cls|eq_filt"]
V["bs_lo"] = n(min(v["breakeven"][0] for v in bs.values()), 3); V["bs_hi"] = n(max(v["breakeven"][1] for v in bs.values()), 3)

# ---------------- risk, data quality, robustness
rr = risk["risk"]
V["TAB_risk"] = " \\\\\n".join(
    f"{lab} & {n(rr[s]['net_exposure'],3)} & {n(rr[s]['net_exposure_p99'],2)} & {n(rr[s]['gross_exposure'],2)} & "
    f"{n(rr[s]['beta_day'],3)} & {n(rr[s]['corr_day'],3)} & {pct(rr[s]['mdd_gross'],1)} & ${rr[s]['dd_days_gross']}$"
    for lab, s in [("CLOSE 5m", "cls|5m"), ("MTF, equal, sign", "cls|eq_sign"), ("MTF, equal, size", "cls|eq_size"),
                   ("MTF, equal, filtered", "cls|eq_filt"), ("MTF, learned, filtered", "cls|lw_filt"),
                   ("CLOSE 4h only", "cls|4h"), ("MID 5m", "mid|5m")]) + " \\\\"
dq = risk["data_quality"]; V["n_flag"] = str(dq["n_flagged"]); V["flag_ex"] = ", ".join(dq["flagged"][:6])
V["dq_ch"] = pct(dq["summary"]["share_c_eq_h"]["mean"], 1, False); V["dq_cl"] = pct(dq["summary"]["share_c_eq_l"]["mean"], 1, False)
rb = risk["robust_drop_flagged"]
V["TAB_robust"] = " \\\\\n".join(
    f"{lab} & {n(M(s,fill='A'),3)} & {n(rb[s]['breakeven_A'],3)} & {n(M(s),3)} & {n(rb[s]['breakeven_B'],3)} & "
    f"{pct(M(s,cost='tick',f='total'),0)} & {pct(rb[s]['total_tick_B'],0)}"
    for lab, s in [("CLOSE 5m", "cls|5m"), ("MTF, equal, sign", "cls|eq_sign"), ("MTF, equal, filtered", "cls|eq_filt"),
                   ("MTF, learned, filtered", "cls|lw_filt"), ("CLOSE 4h only", "cls|4h"), ("MID 5m", "mid|5m")]) + " \\\\"
V["rb_ef_B"] = n(rb["cls|eq_filt"]["breakeven_B"], 3); V["rb_ef_A"] = n(rb["cls|eq_filt"]["breakeven_A"], 3)

# ---------------- yearly (next-open fill, gross breakeven)
yr = keys // 100000000
def yearly(s):
    out = []
    for y in (2021, 2022, 2023, 2024):
        m = (yr == y) | ((y == 2021) & (yr == 2020))
        out.append(n(book[f"{s}|B"][m].mean() / book[f"{s}|turn"][m].mean(), 3))
    return out
V["TAB_year"] = " \\\\\n".join(f"{lab} & " + " & ".join(yearly(s)) + f" & {pct(M(s,f='pos_months'),0,False)}"
                               for lab, s in [("CLOSE 5m", "cls|5m"), ("CLOSE 4h only", "cls|4h"), ("MTF, equal, sign", "cls|eq_sign"),
                                              ("MTF, equal, filtered", "cls|eq_filt"),
                                              ("MTF, learned, filtered$^{\\dagger}$", "cls|lw_filt")]) + " \\\\"

# ---------------- time of day (exploratory)
mn = keys % 10000
edges = [(570, 600, "09:30--10:00"), (600, 660, "10:00--11:00"), (660, 780, "11:00--13:00"), (780, 900, "13:00--15:00"),
         (900, 930, "15:00--15:30"), (930, 961, "15:30--16:00")]
V["TAB_tod"] = " \\\\\n".join(f"{lab} & " + " & ".join(n(book[f'{s}|B'][(mn >= a) & (mn < b)].mean() /
                                                         book[f'{s}|turn'][(mn >= a) & (mn < b)].mean(), 3)
                                                         for s in ("cls|5m", "cls|eq_filt")) for a, b, lab in edges) + " \\\\"

# ---------------- replication of the earlier protocol
rep = {}
for price in ("cls", "mid"):
    k, nn, s = legacy_book(price)
    for c in (0.0, 0.25):
        rep[(price, c)] = ST.metrics(s["p"], s["u"], k, c)
def reprow(lab, f, d=2, kind="n"):
    fmt = (lambda x: n(x, d)) if kind == "n" else ((lambda x: pct(x, 1)) if kind == "p" else money)
    return f"{lab} & {fmt(rep[('mid',0.0)][f])} & {fmt(rep[('cls',0.0)][f])}"
V["TAB_rep"] = " \\\\\n".join([reprow("End value (from \\$1{,}000{,}000)", "end", kind="m"), reprow("Total return", "total", kind="p"),
                               reprow("CAGR", "cagr", kind="p"), reprow("Sharpe ratio", "sharpe"), reprow("Maximum drawdown", "mdd", kind="p"),
                               reprow("Mean return per bar (bps)", "gross_bps", 3), reprow("Breakeven half-spread (bps)", "breakeven", 3)]) + " \\\\"
V["rep_sr"] = n(rep[("cls", 0.0)]["sharpe"], 2); V["rep_be"] = n(rep[("cls", 0.0)]["breakeven"], 3)
V["rep_end"] = money(rep[("cls", 0.0)]["end"]); V["rep_mdd"] = pct(rep[("cls", 0.0)]["mdd"], 1)
V["rep_mid_end"] = money(rep[("mid", 0.0)]["end"]); V["rep_mid_tot"] = pct(rep[("mid", 0.0)]["total"], 0)

# ---------------- equity figure (weekly points, next-open fill unless stated)
def eq_curve(s, fill, costkey=None, per_mask=None):
    p = book[f"{s}|{fill}"] - (book[f"{s}|{costkey}"] if costkey else 0)
    eq = np.cumprod(1 + p / 1e4); wk = (keys // 10000)
    d = pd.to_datetime(wk.astype(str)); idx = pd.Series(np.arange(len(eq)), index=d).groupby(pd.Grouper(freq="W")).last().dropna()
    t0 = d[0]
    return " ".join(f"({(ix - t0).days / 365.25:.3f},{eq[int(v)]:.4f})" for ix, v in idx.items())
for s, tag in (("cls|5m", "f"), ("cls|eq_filt", "ef")):
    V[f"EQ_{tag}_A"] = eq_curve(s, "A"); V[f"EQ_{tag}_B"] = eq_curve(s, "B")
    V[f"EQ_{tag}_Bt"] = eq_curve(s, "B", "ctk"); V[f"EQ_{tag}_Ba"] = eq_curve(s, "B", "car")

tl = json.load(open(os.path.join(R, "test_lookahead.json"))); tr = json.load(open(os.path.join(R, "test_regression.json")))
V.update(t_ok=str(tl["truncation_ok"]), t_n=str(tl["truncation"]), t_stocks=str(tl["stocks"]), t_dates=str(tl["dates"]),
         reg_ok=str(tr["checks"] - tr["failures"]), reg_n=str(tr["checks"]),
         train_days=str(C.TRAIN_DAYS), reest_days=str(C.REEST_DAYS), train5=f"{C.TRAIN_5M:,}".replace(",", "{,}"),
         reest5=f"{C.TRAIN_5M//10:,}".replace(",", "{,}") if C.REEST_5M == C.TRAIN_5M // 10 else f"{C.REEST_5M:,}".replace(",", "{,}"),
         gate_q=f"{C.GATE_Q:.2f}", gate_pct=f"{100*C.GATE_Q:.0f}", gate_block=f"{C.GATE_BLOCK:,}".replace(",", "{,}"),
         sigma_days=str(C.SIGMA_DAYS), spread_days=str(C.SPREAD_LOOKBACK_DAYS), hac=str(C.HAC_LAGS), nlags=str(C.L),
         n_boot="1{,}000", boot_block="20")
V["turn_cut"] = pct(1 - M("cls|eq_filt", f="turn") / M("cls|5m", f="turn"), 0, False)
V["c5_lo"] = f"${M('cls|eq_sign') / M('cls|5m'):.1f}$"
V["c5_hi"] = f"${max(M('cls|eq_filt'), M('cls|lw_filt', 'hold') * M('cls|5m') / M('cls|5m', 'hold')) / M('cls|5m'):.1f}$"
if __name__ == "__main__":
    tpl = open(os.path.join(C.PAPER, "template.tex")).read()
    missing = sorted(set(re.findall(r"@@(\w+)@@", tpl)) - set(V))
    assert not missing, f"missing values: {missing}"
    out = re.sub(r"@@(\w+)@@", lambda m: V[m.group(1)], tpl)
    open(os.path.join(C.PAPER, "main.tex"), "w").write(out)
    json.dump(V, open(os.path.join(C.PAPER, "values.json"), "w"), indent=0)
    print("wrote paper/main.tex with", len(set(re.findall(r'@@(\w+)@@', tpl))), "values")
