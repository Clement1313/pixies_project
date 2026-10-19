import cv2
import tkinter as tk
from tkinter import ttk
import memo as det

IMG_PATH = det.FILE_PATH
full = cv2.imread(IMG_PATH, cv2.IMREAD_COLOR)

# (clé, min, max, pas, impair_obligatoire)
SPEC = {
    "Général / filtre carte": [
        ("scale", 0.05, 1.0, 0.05, False),
        ("min_vertices", 3, 10, 1, False), ("max_vertices", 3, 12, 1, False),
        ("min_area", 0, 20000, 100, False), ("max_area_ratio", 0.01, 1.0, 0.01, False),
        ("min_ratio", 0.1, 1.5, 0.05, False), ("max_ratio", 1.0, 5.0, 0.05, False),
    ],
    "Détection 1": [
        ("d1_blur", 1, 31, 2, True), ("d1_canny_lo", 0, 255, 1, False),
        ("d1_canny_hi", 0, 255, 1, False), ("d1_eps", 0.001, 0.1, 0.001, False),
    ],
    "Détection 2": [
        ("d2_blur", 1, 31, 2, True), ("d2_canny_lo", 0, 255, 1, False),
        ("d2_canny_hi", 0, 255, 1, False), ("d2_eps", 0.001, 0.1, 0.001, False),
        ("d2_close_k", 1, 15, 1, False), ("d2_close_iter", 0, 10, 1, False),
        ("d2_dilate_iter", 0, 10, 1, False),
    ],
    "Détection 3": [
        ("d3_bil_d", 1, 25, 1, False), ("d3_bil_color", 1, 250, 1, False),
        ("d3_bil_space", 1, 250, 1, False), ("d3_canny_lo", 0, 255, 1, False),
        ("d3_canny_hi", 0, 255, 1, False), ("d3_eps", 0.001, 0.1, 0.001, False),
    ],
    "Merge / Final / Crop": [
        ("merge_dist", 1, 200, 1, False), ("final_area_tol", 0, 50000, 100, False),
        ("crop_margin", 0, 100, 1, False),
    ],
}

root = tk.Tk()
root.title("Réglages")
nb = ttk.Notebook(root)
nb.pack(fill="both", expand=True)

dirty = True
vars_ = {}

def on_change(key, odd):
    def cb(_=None):
        global dirty
        v = vars_[key].get()
        if odd:
            v = int(v) | 1                      # force impair
        default = det.PARAMS[key]
        det.PARAMS[key] = v if isinstance(default, float) else int(round(v))
        dirty = True
    return cb

for tab, items in SPEC.items():
    frame = ttk.Frame(nb)
    nb.add(frame, text=tab)
    for r, (key, lo, hi, step, odd) in enumerate(items):
        ttk.Label(frame, text=key, width=18).grid(row=r, column=0, sticky="w", padx=5)
        var = tk.DoubleVar(value=det.PARAMS[key])
        vars_[key] = var
        tk.Scale(frame, from_=lo, to=hi, resolution=step, orient="horizontal",
                 length=320, variable=var, command=on_change(key, odd)
                 ).grid(row=r, column=1, padx=5)

def print_params():
    print("PARAMS =", det.PARAMS)

ttk.Button(root, text="Afficher PARAMS (console)", command=print_params).pack(pady=5)

def run():
    img = cv2.resize(full, None, fx=det.PARAMS["scale"], fy=det.PARAMS["scale"],
                     interpolation=cv2.INTER_AREA)
    c1, f1 = det.first_detection(img, True)
    c2, f2 = det.second_detection(img, True)
    c3, f3 = det.third_detection(img, True)
    allc = c1 + c2 + c3
    merged, merged_img = det.merge_cards(allc, img, True)
    if merged:
        _, checked_img = det.final_card_check(merged, img, True)
    else:
        checked_img = img.copy()
    for name, im in [("filtered", f1), ("filtered2", f2), ("filtered3", f3),
                     ("merged", merged_img), ("checked", checked_img)]:
        cv2.imshow(name, im)
    print(f"d1={len(c1)} d2={len(c2)} d3={len(c3)} merged={len(merged)}")

while True:
    try:
        root.update()
    except tk.TclError:
        break
    if dirty:
        dirty = False
        try:
            run()
        except Exception as e:
            print("Erreur :", e)
    if cv2.waitKey(30) == 27:   # Echap pour quitter
        break

cv2.destroyAllWindows()