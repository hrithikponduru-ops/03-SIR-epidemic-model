# SIR Epidemic Model with Hand-Written Runge–Kutta Solver

An outbreak in a closed population, modelled by three coupled differential equations, solved with a fourth-order Runge–Kutta integrator I wrote myself, and fitted to a real outbreak (influenza in an English boarding school, 1978) with a Nelder–Mead optimiser I also wrote myself. The only library code in the implementation is NumPy arrays. SciPy appears once, inside a test, as the reference solver.

## Key Results

| What | Result |
|---|---|
| Measured convergence order of my RK4 | 4.06, 4.03 (theory: 4) |
| Measured convergence order of Euler | 1.01, 1.01 (theory: 1) |
| Agreement with SciPy's RK45 on the SIR system | better than $10^{-6}$ |
| Fitted $R_0$ for the 1978 outbreak | 3.72 (literature: 3.6 to 3.8) |
| Predicted peak vs observed | 289 boys on day 6.4 vs 291 boys on day 6 |
| Tests | 12, all encoding a theorem or a known answer |

---

## 1. The Idea in Plain Words

Split a population of $N$ people into three groups:

- **S, susceptible**: healthy, could catch it;
- **I, infected**: sick and infectious;
- **R, recovered**: immune (or dead), out of the game.

Two things happen. Infected people meet susceptible people and pass the disease on, and infected people recover. Write $\beta$ for the number of infectious contacts one person makes per day, and $\gamma$ for the fraction of infected people who recover each day (so the average illness lasts $1/\gamma$ days).

$$
\begin{aligned}
\frac{dS}{dt} &= -\beta \frac{S I}{N} \\[6pt]
\frac{dI}{dt} &= \beta \frac{S I}{N} - \gamma I \\[6pt]
\frac{dR}{dt} &= \gamma I
\end{aligned}
$$

The infection term is proportional to $S \cdot I$ because an infection needs one person from each group to meet. Dividing by $N$ makes $\beta$ a per-person rate.

### When Does an Outbreak Grow?

Factor the middle equation:

$$
\frac{dI}{dt} = I \left( \beta \frac{S}{N} - \gamma \right).
$$

At the start almost everyone is susceptible, $S \approx N$, so the bracket is $\beta - \gamma$. The number of infected people grows if and only if $\beta > \gamma$, that is

$$
R_0 = \frac{\beta}{\gamma} > 1 .
$$

$R_0$ is the **basic reproduction number**: one sick person makes $\beta$ infectious contacts per day for $1/\gamma$ days, so infects $\beta / \gamma$ people in a fully susceptible population. Everything about the shape of an epidemic curve follows from this one number.

There is no formula for $S(t)$, $I(t)$, $R(t)$. The equations have to be solved numerically, which is what the rest of this project is about.

---

## 2. Worked Example You Can Check With a Calculator

Take $N = 1000$, one infected person, $\beta = 0.5$ and $\gamma = 0.25$ per day, so $R_0 = 2$. Take one step of one day with the RK4 method (Section 3.6). Write $y = [S, I, R]$ and $f(y)$ for the right-hand side above.

**Start:** $y_0 = [999,\ 1,\ 0]$.

### Slope 1, at the start:

$$
k_1 = f(y_0) = \left[ -\tfrac{0.5 \cdot 999 \cdot 1}{1000},\ \tfrac{0.5 \cdot 999 \cdot 1}{1000} - 0.25 \cdot 1,\ 0.25 \cdot 1 \right]
    = [-0.4995,\ 0.2495,\ 0.25].
$$

### Slope 2, at the midpoint reached by following $k_1$ for half a day:

$$
y_0 + \tfrac{h}{2} k_1 = [998.7503,\ 1.1247,\ 0.1250], \qquad
k_2 = f\!\left(y_0 + \tfrac{h}{2} k_1\right) = [-0.5617,\ 0.2805,\ 0.2812].
$$

### Slope 3, at the midpoint reached by following $k_2$ instead:

$$
y_0 + \tfrac{h}{2} k_2 = [998.7192,\ 1.1402,\ 0.1406], \qquad
k_3 = [-0.5694,\ 0.2843,\ 0.2851].
$$

### Slope 4, at the end of the step reached by following $k_3$ for a whole day:

$$
y_0 + h k_3 = [998.4306,\ 1.2843,\ 0.2851], \qquad
k_4 = [-0.6412,\ 0.3201,\ 0.3211].
$$

### Combine with weights $1, 2, 2, 1$ and divide by 6:

$$
y_1 = y_0 + \frac{h}{6}\,(k_1 + 2k_2 + 2k_3 + k_4)
    = [998.4329,\ 1.2832,\ 0.2839].
$$

Compare with Euler's method, which just follows $k_1$ for the whole day:
$y_1^{\text{Euler}} = y_0 + h k_1 = [998.5005,\ 1.2495,\ 0.25]$.

A reference solution with a step of $10^{-4}$ days gives $I(1) = 1.2832$. RK4's error in $I$ is $8 \times 10^{-6}$; Euler's is $3 \times 10^{-2}$. Four slope evaluations bought a factor of 4000 in accuracy.

### How big does this outbreak get?

Section 3.4 derives an equation for $S_\infty$, the number of people never infected:

$$
\ln \frac{999}{S_\infty} = 2 \left( 1 - \frac{S_\infty}{1000} \right).
$$

Solving by bisection gives $S_\infty = 202.85$, so 797 of 1000 people catch the disease even though $R_0$ is only 2. Running RK4 to day 400 gives $S = 202.85$ as well, and the peak of $I$ (154 people, day 27) happens when $S = 501$, which Section 3.3 says should be $N / R_0 = 500$.

`python worked_example.py` prints every one of these numbers.

---

## 3. Derivations

### 3.1 Conservation

Add the three right-hand sides:

$$
\frac{dS}{dt} + \frac{dI}{dt} + \frac{dR}{dt}
= -\beta \frac{SI}{N} + \left( \beta \frac{SI}{N} - \gamma I \right) + \gamma I = 0 .
$$

So $S + I + R$ is constant and equals $N$. People move between boxes; nobody is created or lost. Every numerical method that takes linear combinations of the right-hand side (Euler, RK4, all of them) inherits this exactly, which makes it a useful test but a weak one: see Section 7.

### 3.2 $R_0$ from the linearisation

Near the start, $S \approx N$, and the infected equation becomes linear:

$$
\frac{dI}{dt} \approx (\beta - \gamma)\, I
\quad\Longrightarrow\quad
I(t) \approx I_0\, e^{(\beta - \gamma) t} = I_0\, e^{\gamma (R_0 - 1) t}.
$$

Exponential growth when $R_0 > 1$, exponential decay when $R_0 < 1$. The growth rate $\gamma (R_0 - 1)$ is what an epidemiologist reads off the early part of a log-scale case plot.

### 3.3 The peak happens when $S = N / R_0$

$I$ is at its maximum when $dI/dt = 0$:

$$
I \left( \beta \frac{S}{N} - \gamma \right) = 0
\quad\Longrightarrow\quad
S = \frac{\gamma N}{\beta} = \frac{N}{R_0}.
$$

The peak is not when everyone has been infected. It is when the susceptible pool has shrunk enough that each infected person infects fewer than one other.

---

## 4. Fitting Data: Nelder–Mead Optimisation

### 4.1 The Sum of Squared Errors

Given observations $I_1^{\text{obs}}, I_2^{\text{obs}}, \ldots$ at times $t_1, t_2, \ldots$, minimise the sum of squared errors:

$$
\mathrm{SSE}{SSE}(\beta, \gamma) = \sum_j \left( I(t_j;\ \beta, \gamma) - I_j^{\text{obs}} \right)^2 .
$$

Each evaluation of SSE means running RK4 from $t = 0$ to the last observation. There is no formula for the gradient, so the minimiser must work from function values alone.

### 4.2 Nelder–Mead: a triangle that crawls downhill

Keep $d + 1$ points (a triangle in two dimensions) and their SSE values. Sort them best to worst, and let $\bar x$ be the centroid of all points except the worst, $x_w$. Then try, in order:

| Move | New point | Accept when |
|---|---|---|
| Reflect | $x_r = \bar x + (\bar x - x_w)$ | $f(x_r)$ is between the best and second-worst |
| Expand | $x_e = \bar x + 2(\bar x - x_w)$ | $f(x_r)$ is the new best; take $x_e$ if it beats $x_r$ |
| Contract outside | $x_c = \bar x + \tfrac{1}{2}(x_r - \bar x)$ | $f(x_r)$ beat the worst but nothing else |
| Contract inside | $x_c = \bar x + \tfrac{1}{2}(x_w - \bar x)$ | $f(x_r)$ was worse than the worst |
| Shrink | pull every vertex halfway toward the best | both contractions failed |

Stop when the triangle is smaller than $10^{-8}$ across and the SSE values at its corners agree to $10^{-10}$. On the boarding-school data this takes about 60 iterations from a starting guess of $(\beta, \gamma) = (2, 0.5)$.

---

## 5. Verification

Every test states a theorem and checks the code against it.

| Test | What it encodes | Tolerance |
|---|---|---|
| `test_measured_convergence_order` | global error is $O(h)$ for Euler, $O(h^4)$ for RK4; the measured exponent from halving $h$ twice | order within 0.1 |
| `test_rk4_on_cosine_matches_sine` | $y' = \cos t$ has exact solution $\sin t$ | $10^{-8}$ |
| `test_rk4_matches_scipy_on_sir` | agreement with `scipy.integrate.solve_ivp`, RK45 at rtol $10^{-10}$ | $10^{-6}$ |
| `test_population_is_conserved` | Section 3.1: $S + I + R = N$ at every step | $10^{-9}$ |
| `test_compartments_stay_non_negative` | counts of people cannot go negative | exact |
| `test_final_size_relation` | Section 3.4, after integrating to day 400; also matches the bisection root | $10^{-4}$ |
| `test_infection_peaks_when_s_equals_n_over_r0` | Section 3.3 | one step's change in $S$ |
| `test_no_outbreak_when_r0_below_one` | Section 3.2: $I$ is monotone decreasing when $R_0 < 1$ | exact |
| `test_nelder_mead_finds_minimum_of_quadratic` | the optimiser finds a known minimum | $10^{-6}$ |
| `test_nelder_mead_finds_minimum_of_rosenbrock` | the banana function, a standard hard case | $10^{-4}$ |
| `test_fit_recovers_known_parameters_from_noise_free_data` | fitting synthetic data returns the $\beta, \gamma$ that made it | $10^{-3}$ |

```
python -m pytest tests/ -v

12 passed
```

---

## 6. Results: Influenza in a Boarding School, 1978

### The Data

A letter to the *British Medical Journal* (4 March 1978, p. 587) reported an influenza outbreak among 763 boys at a boarding school in the north of England. One boy returned from holiday infected; over the next two weeks the number confined to bed each day was:

| Day | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | 11 | 12 | 13 | 14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| In bed | 3 | 8 | 28 | 75 | 221 | 291 | 255 | 235 | 190 | 125 | 70 | 28 | 12 | 5 |

A closed population, a single index case, a known $N$, and daily counts: this is as close to the SIR model's assumptions as real data gets, which is why it appears in every mathematical epidemiology textbook. "In bed" stands in for $I$; the letter also reports that 512 boys in total were confined to bed.

### The Fit

Starting at $t = 0$ with $S = 762$, $I = 1$, $R = 0$:

| Quantity | Fitted / predicted | Observed |
|---|---|---|
| $\beta$ | 1.665 per day | |
| $\gamma$ | 0.448 per day (illness lasts 2.2 days) | |
| $R_0 = \beta / \gamma$ | **3.72** | |
| Sum of squared errors | 4092 over 14 days (root-mean-square 17 boys) | |
| Peak | 289 boys on day 6.4 | 291 boys on day 6 |
| Susceptibles left at the peak | $N / R_0 = 205$ | |
| Final size $N - S_\infty$ | 743 boys (97 %) | 512 confined to bed (67 %) |

The peak is right to within two boys and half a day. The final size is not: the model says 97 % of the school should have been infected, the letter says 67 % were confined to bed. Those are not the same thing. Boys with mild cases were never sent to bed, so "in bed" undercounts $I$ on the way down, which is visible in the plot: the last four data points sit below the curve. A model with an asymptomatic fraction would fix this, at the cost of a parameter the data cannot identify.

### Counterfactual

Rerun the fitted model with $\beta$ reduced by 20 %, as if the school had cut infectious contacts by a fifth:

| | $R_0$ | Peak | Peak day | Final size |
|---|---|---|---|---|
| As happened | 3.72 | 289 | 6.4 | 743 (97 %) |
| Contacts down 20 % | 2.98 | 227 | 8.3 | 716 (94 %) |

The peak falls by a fifth and arrives two days later, but the final size barely moves. With $R_0$ near 3 almost everyone gets infected eventually; what intervention buys is a flatter, later peak. That is "flatten the curve" as a consequence of Section 3.4.

---

## 7. Numerical Lessons Learned

### The fitted $R_0$ depends on a number the data does not contain

The letter says one boy brought the flu back, so $I_0 = 1$. But the fit is sensitive to that assumption:

| Assumed $I_0$ | 0.5 | 1 | 2 | 3 | 5 |
|---|---|---|---|---|---|
| Fitted $R_0$ | 3.91 | 3.72 | 3.53 | 3.42 | 3.28 |
| SSE | 3774 | 4092 | 6145 | 8409 | 12697 |

A bigger seed needs a slower disease to reach 291 by day 6. The SSE prefers small $I_0$, but $I_0 = 0.5$ boys is not a thing. The honest statement is "$R_0 \approx 3.7$ given one index case", and the range 3.3 to 3.9 is a better summary of the uncertainty than any single number.

### Nelder–Mead is local

From starting guesses $(0.5, 0.1)$, $(5, 5)$ and $(0.1, 0.05)$ the simplex found the same minimum, $R_0 = 3.72$, in 73 to 115 iterations. From $(10, 0.2)$ it ran off to $\beta = 78$, $\gamma = 6.2$ with an SSE eighty times worse, and reported success. At that starting point the disease burns through the school in a day and the SSE surface is nearly flat, so the triangle finds a local dip and stops. Gradient-free does not mean global. `run.py` starts from $(2, 0.5)$, chosen by looking at the data: the peak is a week in, so $\gamma$ is around half.

### Conservation is a necessary check, not a sufficient one

With the fitted parameters and a step of $h = 2$ days, Euler's method produces $I = -1.6 \times 10^{7}$ boys, while $S + I + R$ still equals 763 to $10^{-9}$. Any method that updates $y$ by a linear combination of the right-hand side conserves the sum exactly, because the right-hand side sums to zero, no matter how wrong the individual components are. The test for non-negativity catches what the conservation test cannot. RK4 at the same step size stays positive but underestimates the peak by 6 %, and by 21 % at $h = 3$; at $h = 0.05$, the step `run.py` uses, it agrees with SciPy to $10^{-6}$.

### Step-size choice

The fit runs RK4 with $h = 0.05$ days, 20 steps per data point. Each SSE evaluation is 280 steps of a 3-vector, and Nelder–Mead needs about a hundred evaluations, so the whole fit takes a fraction of a second. Halving $h$ to 0.025 left the fitted $R_0 = 3.72002$ unchanged to five decimal places, which is how I know the step is small enough.

---

## 8. How to Run

```bash
pip install -r ../requirements.txt

python -m pytest tests/ -v     # 12 tests, about 3 seconds

python worked_example.py       # every number in Section 2

python run.py                  # fit the 1978 outbreak, write figures/

python make_figures.py         # the convergence-order and RK4-slopes diagrams
```

SciPy is used only in `tests/test_rk4.py` as the reference solver.

---

## 9. Code Map

| File | Contents |
|---|---|
| `sir/model.py` | `SIRParams`, the right-hand side `sir_rhs`, `r0`, the final-size relation and a hand-written `bisect` |
| `sir/rk4.py` | `euler_step`, `rk4_step`, and `integrate` (fixed step, lands on the end time exactly) |
| `sir/fit.py` | Nelder–Mead simplex (`nelder_mead`), the least-squares loss `sse`, and `fit_sir` |
| `data/boarding_school_1978.csv` | the 14 daily counts from the BMJ letter |
| `tests/test_rk4.py` | convergence orders, exact solutions, SciPy comparison |
| `tests/test_sir.py` | conservation, positivity, final size, peak location, $R_0 < 1$ |
| `tests/test_fit.py` | Nelder–Mead on a quadratic and on Rosenbrock; parameter recovery |
| `worked_example.py` | Section 2, one RK4 step by hand |
| `run.py` | Section 6, the fit and its figures |
| `make_figures.py` | Section 3 diagrams |
