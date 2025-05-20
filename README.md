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

### Tools & Technology

* **Core Stack**: `numpy`, `pandas` for data; `matplotlib` for plots.
* **Econometrics**: `statsmodels` (for OLS and VAR diagnostics) and a custom AR(1) setup.
* **Kalman Engine**: Built from scratch with `scipy.optimize` for the EM / MLE steps, plus efficient filter/smoother loops.
* **Data I/O**: Clean functions to load the Liu & Wu Treasury yields, align maturities, fill gaps, and reindex time.
* **Plot Routines**: Simple scripts that recreate classic yield-curve snapshots and show how your Level, Slope, and Curvature factors drift over time.

Each module is designed to be as self-contained as possible—so whether you’re teaching term-structure econ or building a production risk dashboard, you can pick and choose what you need.


* ** Python Code**
import numpy as np
import matplotlib.pyplot as plt
from data_loader import LoadLiu_and_Wu_Data, Preprocess
from estimation import (
    GridSearchBestLambda,
    FitNelsonSiegelOLS,
    EstimateKalmanSystemMatrix,
    DLM_Multivariate,
)

class DynamicNelsonSiegel:
    def __init__(
        self,
        frequency: str = "Monthly",
        low_lambda: float = 0.40,
        high_lambda: float = 0.71,
        grid_steps: int = 100,
        silent: bool = True,
    ):
        """
        frequency: data frequency label passed into loader/preprocessor
        low_lambda, high_lambda: search bounds for the spline decay parameter
        grid_steps: how many points between low_lambda and high_lambda
        silent: if True, suppress intermediate printouts
        """
        self.params = {
            "Frequency": frequency,
            "LowLambda": low_lambda,
            "HighLambda": high_lambda,
            "NumGrid": grid_steps,
            "Silent": silent,
        }

    def two_step(self):
        """Diebold–Li two-step OLS → Kalman initialization → filter run."""
        R = {"Frequency": self.params["Frequency"]}
        R = LoadLiu_and_Wu_Data(R)
        R = Preprocess(R)
        R.update(
            {
                "LambdaHistory": [],
                "DNSLLHistory": [],
                "RecordLLHistory": 0,
            }
        )

        # 1) Grid search for best λ
        R = GridSearchBestLambda(
            R,
            self.params["LowLambda"],
            self.params["HighLambda"],
            self.params["NumGrid"],
            self.params["Silent"],
        )
        R["Lambda"] = R["MaxLikelihoodLambda"]

        # 2) Cross-sectional OLS to back out Level/Slope/Curvature
        R = FitNelsonSiegelOLS(R, self.params["Silent"])

        # 3) Estimate AR(1) state matrix from those factors
        R = EstimateKalmanSystemMatrix(R, self.params["Silent"])

        # 4) Run the multivariate Kalman filter & smoother
        R["RecordLLHistory"] = 1
        R = DLM_Multivariate(R, self.params["Silent"])

        # 5) Diagnostics
        self._print_fit_stats(R)
        return R

    def em_fit(self, num_em_iters: int = 6, grid_steps: int = None):
        """
        Expectation-Maximization loop:
        Alternately search λ + OLS + system-matrix → Kalman fit → update covariances.
        """
        R = {"Frequency": self.params["Frequency"]}
        R = LoadLiu_and_Wu_Data(R)
        R = Preprocess(R)
        R.update(
            {
                "LambdaHistory": [],
                "DNSLLHistory": [],
                "RecordLLHistory": 0,
            }
        )

        steps = grid_steps or self.params["NumGrid"]
        for i in range(1, num_em_iters + 1):
            if not self.params["Silent"]:
                print(f"EM iteration {i}/{num_em_iters}")

            # same four-step core as two_step()
            R = GridSearchBestLambda(
                R,
                self.params["LowLambda"],
                self.params["HighLambda"],
                steps,
                self.params["Silent"],
            )
            R["Lambda"] = R["MaxLikelihoodLambda"]
            R = FitNelsonSiegelOLS(R, self.params["Silent"])
            R = EstimateKalmanSystemMatrix(R, self.params["Silent"])
            R["RecordLLHistory"] = 1
            R = DLM_Multivariate(R, self.params["Silent"])

            if not self.params["Silent"]:
                ll = np.round(R["DLMoutput"]["LL"], 2)
                print(f" → log-likelihood: {ll}")

        # Plot EM convergence
        history = R["DNSLLHistory"][:num_em_iters]
        plt.plot(range(1, len(history) + 1), history, marker="o")
        plt.title("DNS Log-Likelihood vs EM Iteration")
        plt.xlabel("EM Iteration")
        plt.ylabel("Log-Likelihood")
        plt.show()

        # Final diagnostics
        self._print_fit_stats(R)
        return R

    def _print_fit_stats(self, R):
        """Shared printout of overall & per-maturity correlations."""
        f = R["DLMoutput"]["f"]
        Y = R["DLMoutput"]["Y"]

        # overall Spearman
        sc = np.corrcoef(np.ravel(Y), np.ravel(f), method="spearman")[0, 1]
        print(f"Overall Spearman corr.: {sc:.4f}")

        # per-maturity Spearman / Pearson
        for idx, mat in enumerate(R["Maturity"]):
            y_row = Y[idx, :]
            f_row = f[idx, :]
            sp = np.corrcoef(y_row, f_row, method="spearman")[0, 1]
            pe = np.corrcoef(y_row, f_row, method="pearson")[0, 1]
            print(f" Maturity {mat}: Spearman={sp:.4f}, Pearson={pe:.4f}")

