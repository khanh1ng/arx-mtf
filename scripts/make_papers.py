"""Stage W2: fill paper/p1_template.tex, p2_template.tex, p3_template.tex -> paper/paper1.tex ... paper3.tex.
Reuses every value of make_paper.py and adds the values used only by the three-paper series."""
import os, sys, re, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, pandas as pd
from scipy import stats as sst
import make_paper as MP
from make_paper import V, M, n, pct, money, C, ST, est, sts, book, keys, ps, anat, wts, R

ex = json.load(open(os.path.join(R, "extra.json"))); sen = json.load(open(os.path.join(R, "sensitivity.json")))
ident = pd.read_csv(os.path.join(R, "est_identity.csv")); lad = est["b0_ladder"]; VA = est["var"]; VO = est["var_overnight"]

# ---------------- paper 1: full VAR table, overnight robustness, example, ladder, scatter, two-step
def fullrow(nm, lab, mode):
    d = VA[f"{mode}_{nm}"]
    return (f"{lab} & {n(d['mean'],5 if nm.startswith('c') else 4)} & {n(d['cs_t'],1)} & {n(d['med_t'],2)} & ${100*d['sig_pos']:.1f}$ & "
            f"${100*d['sig_neg']:.1f}$")
for mode, key in (("mid", "P1_VARMID"), ("close", "P1_VARCLS")):
    rows = [fullrow(f"a{i}", f"$a_{i}$", mode) for i in range(1, 6)] + [fullrow(f"b{j}", f"$b_{j}$", mode) for j in range(6)]
    V[key] = " \\\\\n".join(rows) + " \\\\"
    rows = [fullrow(f"c{i}", f"$c_{i}$", mode) for i in range(1, 6)] + [fullrow(f"d{j}", f"$d_{j}$", mode) for j in range(1, 6)]
    V[key + "F"] = " \\\\\n".join(rows) + " \\\\"
V["P1_ON"] = " \\\\\n".join(
    f"{lab} & {n(src[f'{m}_a1']['mean'],4)} & {n(src[f'{m}_b0']['mean'],3)} & {n(src[f'{m}_b1']['mean'],3)} & "
    f"{n(src[f'{m}_b2']['mean'],3)} & {n(src[f'{m}_r2_price'],3)}"
    for lab, src, m in [("MID, overnight removed", VA, "mid"), ("MID, overnight kept", VO, "mid"),
                        ("CLOSE, overnight removed", VA, "close"), ("CLOSE, overnight kept", VO, "close")]) + " \\\\"
V["r2_flow_max"] = n(max(VA["mid_r2_flow_max"], VA["close_r2_flow_max"]), 4)
V["nobs"] = f"{VA['mid_n_mean']:,.0f}".replace(",", "{,}")
E = ex["example"]
V.update(on_sd=n(ex["overnight"]["sd_on"], 0), on_5m=n(ex["overnight"]["sd_5m"], 1), on_ratio=f"${ex['overnight']['ratio']:.0f}$")
for k in ("o", "h", "l", "c", "mid", "o1", "h1", "l1", "c1", "mid1"):
    V[f"ex_{k}"] = f"{E[k]:.2f}"
for k in ("cl", "fwd", "rmid", "rcls"):
    V[f"ex_{k}"] = n(E[k], 2, True)
V["ex_day"], V["ex_time"] = E["day"], E["time"]
L_ = lad
V["P1_LADDER"] = " \\\\\n".join([
    f"(0) Normal approximation $\\sqrt{{2/\\pi}}\\,\\mathrm{{sd}}(oc)$ & {n(L_['gauss'],2)} & ---",
    f"(1) Actual $\\E|oc_t|$ & {n(L_['Eabs_oc'],2)} & {n(L_['Eabs_oc']-L_['gauss'],2,True)}",
    f"(2) Regression of $oc_t$ on $x_t$ alone & {n(L_['b_oc_uni'],2)} & {n(L_['b_oc_uni']-L_['Eabs_oc'],2,True)}",
    f"(3) Regression of $r_t$ on $x_t$ alone & {n(L_['b_rc_uni'],2)} & {n(L_['b_rc_uni']-L_['b_oc_uni'],2,True)}",
    f"(4) Full price equation (Table~\\ref{{tab:varmid}}) & {n(L_['b_full'],2)} & {n(L_['b_full']-L_['b_rc_uni'],2,True)}"]) + " \\\\"
V.update(lad_exkurt=n(L_["exkurt"], 1), lad_ratio=n(L_["ratio"], 3), lad_sd=n(L_["sd_oc"], 2),
         lad_gap_sd=n(L_["sd_gap"], 2), lad_gap_corr=n(L_["corr_gap_x"], 3), lad_med=n(L_["med_ratio"], 3))
I_ = est["identity"]
V.update(sig_after_pos=pct(I_["sig_after_pos"], 1, False), sig_after_neg=pct(I_["sig_after_neg"], 1, False),
         b1_ratio=f"${I_['b1']/I_['b1_ctrl']:.0f}$", b1_over_tk=f"${ps[ps.strat=='cls|5m'].mean_tk.median()/I_['b1_ctrl']:.1f}$")
V["SCATTER"] = " ".join(f"({a:.3f},{b:.3f})" for a, b in zip(ident.pred_b1, ident.b1))
V["sc_max"] = f"{max(ident.pred_b1.max(), ident.b1.max()) * 1.05:.1f}"
T = ex["twostep"]
V.update(ts_refits=f"{T['n_refits']:,}".replace(",", "{,}"), ts_cols=str(T["cols"]), ts_rank=str(T["rank_max"]),
         ts_maxdiff=f"${T['maxdiff']:.1e}".replace("e-", "\\times10^{-") + "}$",
         ts_upos=pct(T["u_pos"], 1, False), ts_ut2=pct(T["u_t2"], 0, False),
         ts_imean=n(T["i_mean"], 1), ts_ipos=pct(T["i_pos"], 0, False), ts_it2=pct(T["i_t2"], 0, False),
         ts_itmean=n(T["i_tmean"], 2), sd_ratio=n(np.sqrt(VA["close_r2_flow"]), 3))

# ---------------- paper 2: coefficients, weights, single timeframes, sensitivity
co = anat["coefficients"]
labs = [("const", "$\\beta_0$"), ("r_t-0", "$r_t$"), ("r_t-1", "$r_{t-1}$"), ("r_t-2", "$r_{t-2}$"), ("r_t-3", "$r_{t-3}$"),
        ("r_t-4", "$r_{t-4}$"), ("x_t-0", "$x_t$"), ("x_t-1", "$x_{t-1}$"), ("x_t-2", "$x_{t-2}$"), ("x_t-3", "$x_{t-3}$"),
        ("x_t-4", "$x_{t-4}$")]
V["P2_COEF"] = " \\\\\n".join(f"{lab} & {n(co[k]['mean'],4)} & {pct(co[k]['pos'],1,False)}" for k, lab in labs) + " \\\\"
TAU = [t for t, _ in C.TAUS]
V["P2_W"] = " \\\\\n".join(f"{p_} & " + " & ".join(n(wts[p_][t] / max(sum(wts[p_].values()), 1e-12), 2) for t in TAU)
                           for p_ in ("cls", "mid")).replace("cls &", "CLOSE &").replace("mid &", "MID &") + " \\\\"
def senrow(lab, key):
    v = sen[key]
    return f"{lab} & {n(v['beA'],3)} & {n(v['beB'],3)} & {n(v['srB'],2)} & {n(v['turn'],3)} & {pct(v['tot_tick'],0)}"
rows = []
for strat, sl in (("5m", "CLOSE 5m"), ("eq_sign", "MTF, equal, sign"), ("eq_filt_q075", "MTF, equal, filtered")):
    rows.append(f"\\multicolumn{{6}}{{@{{}}l}}{{\\textit{{{sl}}}}}")
    for vk, vl in (("base", "baseline ($L=5$, 256 sessions)"), ("L1", "$L=1$"), ("L3", "$L=3$"), ("L10", "$L=10$"),
                   ("W128", "window 128 sessions"), ("W512", "window 512 sessions$^{\\dagger}$")):
        rows.append(senrow(vl, f"{vk}|{strat}"))
    rows.append("\\midrule")
rows.append("\\multicolumn{6}{@{}l}{\\textit{MTF, equal, filter quantile}}")
for q in ("q050", "q075", "q090"):
    rows.append(senrow(f"$q={int(q[1:])/100:.2f}$", f"base|eq_filt_{q}"))
V["P2_SENS"] = " \\\\\n".join(rows).replace("\\midrule \\\\", "\\midrule") + " \\\\"
V["sen_start"] = MP.date(sen["base|5m"]["start"] * 10000); V["sen_start512"] = MP.date(sen["W512|5m"]["start"] * 10000)
f5 = [sen[f"{v}|5m"]["beB"] for v in ("base", "L1", "L3", "L10", "W128", "W512")]
ff = [sen[f"{v}|eq_filt_q075"]["beB"] for v in ("base", "L1", "L3", "L10", "W128", "W512")] + \
     [sen[f"base|eq_filt_{q}"]["beB"] for q in ("q050", "q090")]
tt = [v["tot_tick"] for v in sen.values()]
V.update(sen_f_lo=n(min(f5), 3), sen_f_hi=n(max(f5), 3), sen_ef_lo=n(min(ff), 3), sen_ef_hi=n(max(ff), 3),
         sen_tt_lo=pct(min(tt), 0), sen_tt_hi=pct(max(tt), 0), n_sens=str(len(sen) - 3))
def singlerow(t):
    s = f"cls|{t}"
    return (f"{t} & {n(M(s,f='gross_bps'),3)} & {n(M(s,f='sharpe'),2)} & {n(M(s,f='turn'),3)} & {n(M(s,fill='A'),3)} & "
            f"{n(M(s),3)} & {pct(M(s,f='pos_months'),0,False)}")
V["P2_SINGLE"] = " \\\\\n".join(singlerow(t) for t in TAU) + " \\\\"
def comborow(lab, s, per="full"):
    return (f"{lab} & {n(M(s,per,f='gross_bps'),3)} & {n(M(s,per,f='sharpe'),2)} & {n(M(s,per,f='turn'),3)} & "
            f"{n(M(s,per,'A'),3)} & {n(M(s,per),3)} & {pct(M(s,per,f='pos_months'),0,False)}")
V["P2_COMBO"] = " \\\\\n".join([comborow("5m only", "cls|5m"), comborow("Equal, sign", "cls|eq_sign"),
                                  comborow("Equal, size $|S|$", "cls|eq_size"), comborow("Equal, filtered", "cls|eq_filt"),
                                  "\\midrule \\multicolumn{7}{@{}l}{\\textit{Out of sample, Jan 2022 -- Jan 2025}}",
                                  comborow("5m only", "cls|5m", "hold"), comborow("Equal, filtered", "cls|eq_filt", "hold"),
                                  comborow("Learned, sign", "cls|lw_sign", "hold"),
                                  comborow("Learned, filtered", "cls|lw_filt", "hold")]) + " \\\\"
V["P2_COMBO"] = V["P2_COMBO"].replace("}} \\\\\n5m only", "}} \\\\\n5m only")
V["lw_beB"] = n(M("cls|lw_filt", "hold"), 3); V["ew_beB_h"] = n(M("cls|eq_filt", "hold"), 3)
V["lws_beB"] = n(M("cls|lw_sign", "hold"), 3); V["ews_beB_h"] = n(M("cls|eq_sign", "hold"), 3)

# score magnitudes, from the per-stock files
absS = {t: [] for t in TAU}; absC = []
for tic in MP.data.universe() if hasattr(MP, "data") else []:
    pass
from arxmtf import data as _data, signals as _S
for tic in _data.universe():
    z = np.load(os.path.join(C.PERTIC, tic + ".npz"))
    Sm = np.column_stack([z[f"s_cls_{t}"].astype(float) for t in TAU])
    for j, t in enumerate(TAU):
        absS[t].append(np.nanmean(np.abs(Sm[:, j])))
    absC.append(np.nanmean(np.abs(_S.combine(Sm, np.ones(6)))))
V["abs_s_lo"] = n(min(np.mean(v) for v in absS.values()), 3); V["abs_s_hi"] = n(max(np.mean(v) for v in absS.values()), 3)
V["abs_s5"] = n(np.mean(absS["5m"]), 3); V["abs_S"] = n(np.mean(absC), 3)
V["abs_s5_pct"] = pct(np.mean(absS["5m"]), 0, False)
wc_ = wts["cls"]; top2 = sorted(TAU, key=lambda t: -wc_[t])[:2]
V["w_top2"] = " and ".join(sorted(top2, key=TAU.index))
V["w_mid_pos"] = " and ".join(t for t in TAU if wts["mid"][t] > 0)

# ---------------- paper 3: decomposition, deflated Sharpe with sensitivity trials, cost stats
dc = ex["decomp"]; ga = ex["gap_after"]
V.update(dec_A=n(dc["cls|5m"]["A"], 3), dec_B=n(dc["cls|5m"]["B"], 3), dec_diff=n(dc["cls|5m"]["diff"], 3),
         dec_share=pct(dc["cls|5m"]["diff"] / dc["cls|5m"]["A"], 0, False),
         dec_ef_share=pct(dc["cls|eq_filt"]["diff"] / dc["cls|eq_filt"]["A"], 0, False),
         gap_up=n(ga["up"], 3, True), gap_dn=n(ga["down"], 3, True), gap_up_sh=pct(ga["share_up_neg"], 0, False),
         gap_dn_sh=pct(ga["share_dn_pos"], 0, False))
strats = sorted({k.rsplit("|", 1)[0] for k in book.files if k.endswith("|turn")})
srs = np.array([ST.sharpe_bar(book[f"{s}|B"]) for s in strats])
best = strats[int(np.nanargmax(srs))]; pb = book[f"{best}|B"]
N102 = len(strats) + 39 + (len(sen) - 3)
prob, sr0 = ST.deflated_sharpe(ST.sharpe_bar(pb), len(pb), float(sst.skew(pb)), float(sst.kurtosis(pb, fisher=False)),
                               np.resize(srs, N102))
V.update(n_all=str(N102), dsr_all=n(prob, 3), sr0_all=n(sr0, 2))
V["P3_COSTDIST"] = " \\\\\n".join(
    f"{lab} & {n(ps[ps.strat=='cls|5m'][col].quantile(.1),2)} & {n(ps[ps.strat=='cls|5m'][col].quantile(.25),2)} & "
    f"{n(ps[ps.strat=='cls|5m'][col].median(),2)} & {n(ps[ps.strat=='cls|5m'][col].quantile(.75),2)} & "
    f"{n(ps[ps.strat=='cls|5m'][col].quantile(.9),2)}"
    for lab, col in (("Tick floor (lower bound)", "mean_tk"), ("Abdi--Ranaldo (central)", "mean_ar"),
                     ("Corwin--Schultz (upper)", "mean_cs"))) + " \\\\"

for s_, tag in (("cls|5m", "f"), ("cls|eq_filt", "ef")):
    q_ = ps[ps.strat == s_]; be_ = q_.pnl_B / q_.turn
    V[f"beat_tk_{tag}"] = pct((be_ > q_.mean_tk).mean(), 0, False); V[f"beat_ar_{tag}"] = pct((be_ > q_.mean_ar).mean(), 0, False)
V["cost_turn_f"] = n(M("cls|5m", f="turn") * ps[ps.strat == "cls|5m"].mean_tk.median(), 3)

mn_ = keys % 10000
edges_ = [(570, 600, "09:30--10:00"), (600, 660, "10:00--11:00"), (660, 780, "11:00--13:00"), (780, 900, "13:00--15:00"),
          (900, 930, "15:00--15:30"), (930, 961, "15:30--16:00")]
tod_ = [(book["cls|eq_filt|B"][(mn_ >= a) & (mn_ < b)].mean() / book["cls|eq_filt|turn"][(mn_ >= a) & (mn_ < b)].mean(), l)
        for a, b, l in edges_]
V["tod_best"] = n(max(tod_)[0], 2); V["tod_best_win"] = max(tod_)[1]

# monthly equity points (small enough to paste)
def eq_month(s, fill, costkey=None, per_mask=None):
    m = np.ones(len(keys), bool) if per_mask is None else per_mask
    p = book[f"{s}|{fill}"][m] - (book[f"{s}|{costkey}"][m] if costkey else 0)
    eq = np.cumprod(1 + p / 1e4); kk = keys[m]
    mo = kk // 1_000_000; last = np.r_[np.flatnonzero(np.diff(mo)), len(mo) - 1]
    return "(0,1) " + " ".join(f"({i+1},{eq[j]:.4f})" for i, j in enumerate(last))
for s, tag in (("cls|5m", "f"), ("cls|eq_filt", "ef")):
    V[f"EQM_{tag}_A"] = eq_month(s, "A"); V[f"EQM_{tag}_B"] = eq_month(s, "B")
    V[f"EQM_{tag}_Bt"] = eq_month(s, "B", "ctk"); V[f"EQM_{tag}_Ba"] = eq_month(s, "B", "car")
hm = keys // 10000 >= C.LEARN_END
V["EQM_lf_B"] = eq_month("cls|lw_filt", "B", per_mask=hm); V["EQM_lf_Bt"] = eq_month("cls|lw_filt", "B", "ctk", hm)
V["EQM_lf_Ba"] = eq_month("cls|lw_filt", "B", "car", hm)

# passive execution follow-up (docs/spec/passive_execution.md), primary configuration, hold-out
PJ = json.load(open(os.path.join(R, "passive.json")))
PM = pd.read_csv(os.path.join(R, "passive_metrics.csv"))
pr = PM[(PM.cfg == PJ["primary"]["cfg"]) & (PM.period == "hold")].iloc[0]
V.update(ps_fill=pct(pr.fill_rate, 0, False), ps_mf=n(pr.move_filled_bps, 1, True), ps_mu=n(pr.move_unfilled_bps, 1, True),
         ps_sh=n(PJ["primary"]["sharpe_tick"], 1), ps_shB=n(M("cls|eq_filt", "hold", "B", "tick", "sharpe"), 1),
         ps_rebate=n(-PJ["primary"]["breakeven_fee_bps"], 2))

for i in (1, 2, 3):
    tp = os.path.join(C.PAPER, f"p{i}_template.tex")
    if not os.path.exists(tp):
        continue
    tpl = open(tp).read()
    keys_cited = set(k.strip() for grp in re.findall(r"\\cite[tp]?\{([^}]*)\}", tpl) for k in grp.split(","))
    items = re.findall(r"(\\bibitem\[[^\]]*\]\{(\w+)\}[^\n]*)", open(os.path.join(C.PAPER, "refs.tex")).read())
    unknown = keys_cited - {k for _, k in items}
    assert not unknown, f"paper {i} cites unknown keys: {unknown}"
    V["REFS"] = "\\begin{thebibliography}{99}\n" + "\n".join(t for t, k in items if k in keys_cited) + "\n\\end{thebibliography}"
    missing = sorted(set(re.findall(r"@@(\w+)@@", tpl)) - set(V))
    assert not missing, f"paper {i} missing values: {missing}"
    open(os.path.join(C.PAPER, f"paper{i}.tex"), "w").write(re.sub(r"@@(\w+)@@", lambda m: V[m.group(1)], tpl))
    nv = len(set(re.findall(r"@@(\w+)@@", tpl)))
    print(f"wrote paper{i}.tex with {nv} values")
