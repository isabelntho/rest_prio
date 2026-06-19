import pickle, numpy as np

with open("results_files/res_20260521_1613_upper_q75_all_seed103.pkl", "rb") as fh:
    r = pickle.load(fh)

print("top keys:", list(r.keys())[:20])
if "scenarios" in r:
    s = list(r["scenarios"].values())[0]
    print("scenario keys:", list(s.keys())[:20])
    print("objective_names:", s.get("objective_names"))
    print("decisions shape:", np.array(s.get("decisions", [])).shape)
    print("objectives shape:", np.array(s.get("objectives", [])).shape)
    nd = s.get("is_nondominated")
    print("is_nondominated:", None if nd is None else np.sum(nd), "of", len(nd) if nd is not None else 0)
    ic = s.get("initial_conditions", {})
    print("ic keys:", list(ic.keys())[:15])
    print("problem_info:", s.get("problem_info"))
else:
    print("objective_names:", r.get("objective_names"))
    print("decisions shape:", np.array(r.get("decisions", [])).shape)
    print("objectives shape:", np.array(r.get("objectives", [])).shape)
    ic = r.get("initial_conditions", {})
    print("ic keys:", list(ic.keys())[:15])
    print("problem_info:", r.get("problem_info"))

with open("results_files/dynamic_dynamic_fast_seed103_20260529_153952.pkl", "rb") as fh:
    d = pickle.load(fh)
print("\n--- dynamic ---")
print("dyn keys:", list(d.keys())[:20])
print("objective_names:", d.get("objective_names"))
print("decisions shape:", np.array(d.get("decisions", [])).shape)
print("objectives shape:", np.array(d.get("objectives", [])).shape)
nd = d.get("is_nondominated")
print("is_nondominated:", None if nd is None else np.sum(nd), "of", len(nd) if nd is not None else 0)
ic2 = d.get("initial_conditions", {})
print("ic keys:", list(ic2.keys())[:15])
print("problem_info:", d.get("problem_info"))
