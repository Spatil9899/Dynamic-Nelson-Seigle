# Yield-Curve Modeling using Nelson-Seigle
This project illustrates why the Nelson–Siegel model is an affine model.



This repo brings the **Dynamic Nelson–Siegel** yield-curve model to life in Python, so you can see how interest-rate smiles and term-structure twists evolve over time:

* **Three Simple Drivers** – Instead of dozens of parameters, DNS boils the entire curve down to **Level**, **Slope**, and **Curvature** factors, each with its own smooth loading curve.
* **Built as a State-Space Model** – We treat those three factors like hidden “states” that follow simple AR(1) rules, then map them to actual yields at different maturities.
* **Real-Time Updates via Kalman** – As new yield observations roll in, the Kalman filter and smoother quickly refine your factor estimates, cutting through the noise.
* **Two Ways to Calibrate** – Pick the classic Diebold–Li OLS two-step or go full-blown EM / maximum-likelihood on the Kalman innovations to fine-tune both factor dynamics and measurement error.
* **Arbitrage-Free Option** – Because DNS is an affine model, you can plug in the Pennacchi no-arbitrage tweaks (AFNS) so your curve never admits easy money-grab “static arbitrage.”
* **Try It on Toy or Real Data** – Comes with both simulated yield paths and sample U.S. Treasury data (Diebold & Rudebusch style), so you can see published examples in action.
* **Modular & Readable** – Data loading, factor estimation, plotting, diagnostics—all live in their own files, so you can swap out pieces or extend the engine without breaking everything else.

---

### Tools & Technology (the nuts & bolts)

* **Core Stack**: `numpy`, `pandas` for data; `matplotlib` for plots.
* **Econometrics**: `statsmodels` (for OLS and VAR diagnostics) and a custom AR(1) setup.
* **Kalman Engine**: Built from scratch with `scipy.optimize` for the EM / MLE steps, plus efficient filter/smoother loops.
* **Data I/O**: Clean functions to load the Liu & Wu Treasury yields, align maturities, fill gaps, and reindex time.
* **Plot Routines**: Simple scripts that recreate classic yield-curve snapshots and show how your Level, Slope, and Curvature factors drift over time.

Each module is designed to be as self-contained as possible—so whether you’re teaching term-structure econ or building a production risk dashboard, you can pick and choose what you need.
