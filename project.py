# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.18.1
#   kernelspec:
#     display_name: Python 3
#     name: python3
# ---

# %% [markdown] id="view-in-github" colab_type="text"
# <a href="https://colab.research.google.com/gist/DanaGoro66/0c00c72ae4fede812674be465e199f3c/project.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# %% [markdown] id="zOb5p-dB2Xj1"
# # Water Park Project

# %%
print("sync test from py")


# %% id="9thXDFNGeV8Y"
import queue
from re import S
import numpy as np
import copy
import random
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import probplot, norm, kstest, expon
from IPython.display import HTML, display
from google.colab import files
from datetime import time, datetime, timedelta
from collections import deque
import heapq
from __future__ import annotations
from typing import Optional, List, Any

# %% [markdown] id="DePf7NHLKTE9"
# # כתיבת קוד למציאת התפלגויות ומבחני טיב התאמה לזמני התגלשות במגלשות האבובים הקטנה והגדולה

# %% colab={"base_uri": "https://localhost:8080/", "height": 1000} id="65CP9yf8oNtc" outputId="4c40fca8-3dd8-443b-e110-ecd11efd0bee"
# =====================================
# 1. Loading data from file
# =====================================

uploaded = files.upload()
excel_file = list(uploaded.keys())[0]
xls = pd.ExcelFile(excel_file)
xls.sheet_names

# Converting the tables to a data structure
big_slide_df   = pd.read_excel(xls, xls.sheet_names[0])
small_slide_df = pd.read_excel(xls, xls.sheet_names[1])

# Taking the time column only
big_data   = big_slide_df['time_minutes'].values
small_data = small_slide_df['time_minutes'].values

alpha = 0.1  # Statistical significance (90% confidence level)

# =====================================
# 2. Graphs
# =====================================

def histogram_and_density_line(data, label, ax, candidate_dist=''):
    unified_blue = "#80CEED"
    unified_pink = "#FCB8D6"

    # Histogram
    ax.hist(
        data,
        bins=20,
        edgecolor='black',
        alpha=0.7,
        density=True,
        label='Histogram',
        color=unified_blue
    )

    # Density line
    pd.Series(data).plot(
        kind='kde',
        color=unified_pink,
        label='Density Line',
        bw_method=0.5,
        ax=ax
    )

    # Headline
    title = f"{label} – Histogram with Density"
    if candidate_dist:
        title += f" (candidate: {candidate_dist})"
    ax.set_title(title, color='dimgray')
    ax.set_xlabel("time_minutes", color='dimgray')
    ax.set_ylabel("Density", color='dimgray')
    ax.set_xlim(left=0)
    ax.legend()

# QQ
def QQ_plot_exponential(data, lambda_mle, ax):
    unified_blue = "#80CEED"
    # expon: (loc, scale) = (0, 1/lambda)
    probplot(data, dist="expon", sparams=(0, 1/lambda_mle), plot=ax)
    ax.get_lines()[1].set_color(unified_blue)
    ax.set_title("QQ Plot – Exponential", color='dimgray')

def QQ_plot_normal(data, mu_mle, sigma_mle, ax):
    unified_blue = "#80CEED"
    # norm: (loc, scale) = (mu, sigma)
    probplot(data, dist="norm", sparams=(mu_mle, sigma_mle), plot=ax)
    ax.get_lines()[1].set_color(unified_blue)
    ax.set_title("QQ Plot – Normal", color='dimgray')

## CDF
def CDF_plot_exponential(data, lambda_mle, ax):
    unified_pink = "#FCB8D6"
    unified_blue = "#80CEED"

    sorted_data = np.sort(data)
    n = len(data)
    empirical_cdf_vals = np.arange(1, n+1) / n
    cdf_fitted = expon.cdf(sorted_data, loc=0, scale=1/lambda_mle)

    ax.plot(
        sorted_data,
        empirical_cdf_vals,
        marker='o',
        linestyle='',
        label='Empirical CDF',
        color=unified_blue
    )
    ax.plot(
        sorted_data,
        cdf_fitted,
        '-',
        label='Fitted Exponential CDF',
        color=unified_pink
    )
    ax.set_title("CDF Comparison – Exponential", color='dimgray')
    ax.set_xlabel("time_minutes", color='dimgray')
    ax.set_ylabel("Cumulative Probability", color='dimgray')
    ax.legend(loc='lower right')

def CDF_plot_normal(data, mu_mle, sigma_mle, ax):
    unified_pink = "#FCB8D6"
    unified_blue = "#80CEED"

    sorted_data = np.sort(data)
    n = len(data)
    empirical_cdf_vals = np.arange(1, n+1) / n
    cdf_fitted = norm.cdf(sorted_data, loc=mu_mle, scale=sigma_mle)

    ax.plot(
        sorted_data,
        empirical_cdf_vals,
        marker='o',
        linestyle='',
        label='Empirical CDF',
        color=unified_blue
    )
    ax.plot(
        sorted_data,
        cdf_fitted,
        '-',
        label='Fitted Normal CDF',
        color=unified_pink
    )
    ax.set_title("CDF Comparison – Normal", color='dimgray')
    ax.set_xlabel("time_minutes", color='dimgray')
    ax.set_ylabel("Cumulative Probability", color='dimgray')
    ax.legend(loc='lower right')

# =====================================
# 3. Small Slide – Exponential (graphs & parameters)
# =====================================

# MLE for exponential
lambda_small_mle = 1 / np.mean(small_data)
theoretical_mean_small = 1 / lambda_small_mle

print("=== SMALL SLIDE – Exponential model ===")
print(f"λ_MLE (lambda)            = {lambda_small_mle:.6f}")
print(f"Theoretical mean (1/λ)    = {theoretical_mean_small:.6f}")
print()

display(HTML("<h2 style='text-align: center; color: #FCB8D6;'>Small Slide – Exponential Fit</h2>"))
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

histogram_and_density_line(small_data, "Small slide", axs[0], candidate_dist='Exponential')
QQ_plot_exponential(small_data, lambda_small_mle, axs[1])
CDF_plot_exponential(small_data, lambda_small_mle, axs[2])

plt.tight_layout()
plt.show()

# =====================================
# 4. Big Slide – Normal (graphs & parameters)
# =====================================

# MLE for normal
mu_big_mle    = np.mean(big_data)
sigma_big_mle = np.std(big_data, ddof=0)
var_big_mle   = sigma_big_mle**2

print("=== BIG SLIDE – Normal model ===")
print(f"μ_MLE (mean)           = {mu_big_mle:.6f}")
print(f"σ^2_MLE (variance)     = {var_big_mle:.6f}")
print(f"σ_MLE (std deviation)  = {sigma_big_mle:.6f}")
print()

display(HTML("<h2 style='text-align: center; color: #FCB8D6;'>Big Slide – Normal Fit</h2>"))
fig, axs = plt.subplots(1, 3, figsize=(18, 5))

histogram_and_density_line(big_data, "Big slide", axs[0], candidate_dist='Normal')
QQ_plot_normal(big_data, mu_big_mle, sigma_big_mle, axs[1])
CDF_plot_normal(big_data, mu_big_mle, sigma_big_mle, axs[2])

plt.tight_layout()
plt.show()


# %% [markdown] id="fO9gcBEox5W2"
# # ניתוח ההתפלגות עבור זמן התגלשות במגלשת האבובים הקטנה

# %% [markdown] id="9_uGrHb-bZNW"
# **היסטוגרמה:**
#
# כאשר ניתחנו את הנתונים מקובץ האקסל ושמנו אותם על הגרף, נכחנו לגלות כי ההיסטוגרמה מציגה דפוס הדומה להתפלגות אקספוננציאלית - ריכוז התצפיות הגדול ביותר מתכנס סביב הערכים 0-1 דקות וככל שמתקדמים לערכים גבוהים יותר בציר האיקס שמסמן דקות, השכיחות הולכת ונהיית נמוכה יותר, מאפיין התואם להתפלגות אקספוננציאלית.
#
#
#
# ---
#
# **QQ Plot:**
#
# בגרף זה, ניתן לראות התאמה בעיקר לערכים הנמוכים של ההתפלגות. כאשר נתקדם לערכים גבוהים יותר, יש סטייה של התצפיות מהקו התיאורטי של ההתפלגות, המעידה על אי התאמה חלקית להתפלגות באיזורים הנ"ל.
#
# ---
#
# **פונקצייה מצטברת:**
#
# גם בהשוואה הסופית שלנו - בין התצפיות לבין פונקצייה מצטברת של התפלגות אקספוננציאלית, יש התאמה טובה יחסית ונראה כי הרוב המוחלט של התצפיות נמצא על הגרף. נשים לב כי עד ערכי חצי דקה, התצפיות מדויקות יותר וכאשר נתבונן בתצפיות גבוהות יותר, הן הולכות ונהיות יותר מפוזרות.
#
# ---
#
# **סיכום:**
#
# המדגם כולל 100 תצפיות, שזהו מדגם קטן יחסית ולמרות זאת, בעזרת הכלים הגרפיים והסטטיסטיים המתאימים נראה כי קיימת סבירות גבוהה לכך שזמן ההתגלשות במגלשת האבובים הקטנה מתפלג אקספוננציאלית.
#

# %% [markdown] id="jzEZSo2FyWPb"
# # MLE for Exponential distribution

# %% [markdown] id="CINI4QbMqp3V"
#
# ##  **פונקציית צפיפות :**
#
# פונקציית הצפיפות של ההתפלגות המעריכית מוגדרת כך:
#
# $$
# f(x; \lambda) =
# \begin{cases}
# \lambda e^{-\lambda x}, & x \geq 0 \\
# 0, & x < 0
# \end{cases}
# $$
#
# כאשר הפרמטר של קצב ההתפלגות גדול מאפס.
#
# ---
#
# ##  **פונקציית הנראות:**
#
# בהינתן מדגם בגודל \( n \):
# $$
# x_1, x_2, \dots, x_n
# $$
# פונקציית הנראות היא:
#
# $$
# L(\lambda) = \prod_{i=1}^n \lambda e^{-\lambda x_i} = \lambda^n e^{-\lambda \sum_{i=1}^n x_i}
# $$
#
# ---
#
# ##  **פונקציית הלוג של פונקציית הנראות:**
#
# ניקח לוגריתם טבעי (ln) לפונקציית הנראות:
#
# $$
# \ell(\lambda) = \ln L(\lambda) = n \ln(\lambda) - \lambda \sum_{i=1}^n x_i
# $$
#
# ---
#
# ##  **מציאת אומדן λ (אומד נראות מקסימלי):**
#
# נגזור את פונקציית הלוג לפי $\lambda$, ונשווה לאפס:
#
# $$
# \frac{d\ell(\lambda)}{d\lambda} = \frac{n}{\lambda} - \sum_{i=1}^n x_i = 0
# $$
#
# נבודד את $\lambda$:
#
# $$
# \lambda = \frac{n}{\sum_{i=1}^n x_i} = \frac{1}{\bar{x}}
# $$
#
# כאשר $\bar{x}$ הוא הממוצע של המדגם.
#
# ---
#
# ##  **מסקנה:**
#
# אומד הנראות המקסימלי של  $\lambda$ הוא:
#
# $$
# \lambda_{MLE} = \frac{1}{\bar{x}}
# $$
#
# </div>
#

# %% [markdown] id="-1kP-1TRymbb"
# # ניתוח ההתפלגות עבור זמן התגלשות במגלשת האבובים הגדולה
#

# %% [markdown] id="Q4TCLCxjYoUB"
#
# **היסטוגרמה:**
#
# כאשר ניתחנו את הנתונים מקובץ האקסל ושמנו אותם על הגרף, נכחנו לגלות כי ההיסטוגרמה מציגה דפוס הדומה להתפלגות נורמלית - רוב התצפיות מתכנסות סביב הערכים 3-5 וככל שמתרחקים מערכים אלו, ריכוז התצפיות יורד, באופן יחסית סימטרי, מאפיין התואם להתפלגות נורמלית.
#
#
#
# ---
#
# **QQ Plot**
#
# בגרף זה ניתן לראות בעיקר התאמה לערכים סביב הממוצע של ההתפלגות. כאשר נתקדם ל-2 זנבות הגרף, יש סטייה של התצפיות מהקו התיאורטי של ההתפלגות, המעידה על אי התאמה חלקית להתפלגות באיזורי הזנבות.
#
# ---
#
# **פונקצייה מצטברת:**
#
# גם בהשוואה הסופית שלנו - בין התצפיות לבין פונקצייה מצטברת של התפלגות נורמלית, יש התאמה טובה יחסית ונראה כי הרוב המוחלט של התצפיות נמצא על הגרף או במיקום קרוב יחסית לגרף.
#
# ---
#
# **סיכום:**
#
# המדגם כולל 100 תצפיות, שזהו מדגם קטן יחסית ולמרות זאת, בעזרת הכלים הגרפיים והסטטיסטיים המתאימים נראה כי קיימת סבירות גבוהה לכך שזמן ההתגלשות במגלשת האבובים הגדולה מתפלג נורמלית.
#

# %% [markdown] id="CX4SNgBKy1Hr"
# # MLE for Normal distribution
#

# %% [markdown] id="tCbZycC-a3D2"
# ## **פונקציית צפיפות:**
#
# התפלגות נורמלית מאופיינת על ידי שני פרמטרים: הממוצע ($\mu$) וסטיית התקן ($\sigma$). פונקציית הצפיפות שלה מוגדרת כך:
#
# $$
# f(x; \mu, \sigma^2) = \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-\frac{(x - \mu)^2}{2 \sigma^2}}
# $$
#
# כאשר $\mu \in (-\infty, \infty)$ , $\sigma^2 > 0$.
#
# ---
#
# ## **פונקציית הנראות:**
#
# בהינתן מדגם של משתנים בלתי תלויים וזהים בהתפלגותם  בגודל $n$:
# $$
# \mathbf{x} = (x_1, x_2, \dots, x_n)
# $$
# פונקציית הנראות $L(\mu, \sigma^2)$ היא מכפלת פונקציות הצפיפות עבור כל תצפית:
#
# $$
# L(\mu, \sigma^2) = \prod_{i=1}^n f(x_i; \mu, \sigma^2)
# $$
#
# נציב את פונקציית הצפיפות הנורמלית:
#
# $$
# L(\mu, \sigma^2) = \prod_{i=1}^n \left( \frac{1}{\sqrt{2 \pi \sigma^2}} e^{-\frac{(x_i - \mu)^2}{2 \sigma^2}} \right)
# $$
#
# $$
# L(\mu, \sigma^2) = \left( \frac{1}{2 \pi \sigma^2} \right)^{n/2} e^{-\sum_{i=1}^n \frac{(x_i - \mu)^2}{2 \sigma^2}}
# $$
#
# ---
#
# ## **פונקציית הלוג של פונקציית הנראות:**
#
# ניקח לוגריתם טבעי ($\ln$) לפונקציית הנראות ($\ell(\mu, \sigma^2) = \ln L(\mu, \sigma^2)$). הדבר מפשט את המכפלה לסכום:
#
# $$
# \ell(\mu, \sigma^2) = \ln \left[ \left( 2 \pi \sigma^2 \right)^{-n/2} \cdot e^{-\frac{1}{2 \sigma^2} \sum_{i=1}^n (x_i - \mu)^2} \right]
# $$
#
# על פי חוקי הלוגריתמים: $\ln(a \cdot b) = \ln(a) + \ln(b)$ ו- $\ln(a^b) = b \ln(a)$:
#
# $$
# \ell(\mu, \sigma^2) = -\frac{n}{2} \ln(2 \pi \sigma^2) - \frac{1}{2 \sigma^2} \sum_{i=1}^n (x_i - \mu)^2
# $$
#
# ---
#
# ## **מציאת אומדן $\mu$ (אומד נראות מקסימלי):**
#
# נגזור את פונקציית הלוג של הנראות לפי $\mu$ ונשווה לאפס:
#
# $$
# \frac{\partial \ell}{\partial \mu} = \frac{\partial}{\partial \mu} \left[ -\frac{n}{2} \ln(2 \pi \sigma^2) - \frac{1}{2 \sigma^2} \sum_{i=1}^n (x_i - \mu)^2 \right] = 0
# $$
#
# האיבר הראשון אינו תלוי ב-$\mu$ ולכן נגזרתו אפס. נגזור את האיבר השני:
#
# $$
# \frac{\partial \ell}{\partial \mu} = - \frac{1}{2 \sigma^2} \sum_{i=1}^n 2 (x_i - \mu) (-1) = 0
# $$
#
# נצמצם ונסדר:
#
# $$
# \frac{1}{\sigma^2} \sum_{i=1}^n (x_i - \mu) = 0
# $$
#
# מכיוון ש-$\sigma^2 > 0$, נחלק ב-$\frac{1}{\sigma^2}$:
#
# $$
# \sum_{i=1}^n (x_i - \mu) = 0
# $$
#
# נחלק את הסכום לשני חלקים:
#
# $$
# \sum_{i=1}^n x_i - \sum_{i=1}^n \mu = 0
# $$
#
# ומכיוון ש-$\mu$ היא קבוע בסכימה:
#
# $$
# \sum_{i=1}^n x_i - n \mu = 0
# $$
#
# נבודד את $\mu$:
#
# $$
# \hat{\mu}_{MLE} = \frac{1}{n} \sum_{i=1}^n x_i
# $$
#
# ---
#
# ## **מציאת אומדן $\sigma^2$ (אומד נראות מקסימלי):**
#
# נגזור את פונקציית הלוג של הנראות לפי $\sigma^2$ (נחשוב על $\sigma^2$ כמשתנה יחיד, נסמנו $v = \sigma^2$) ונשווה לאפס.
#
# $$
# \frac{\partial \ell}{\partial v} = \frac{\partial}{\partial v} \left[ -\frac{n}{2} \ln(2 \pi v) - \frac{1}{2 v} \sum_{i=1}^n (x_i - \mu)^2 \right] = 0
# $$
#
# נגזור איבר איבר:
# * $\frac{\partial}{\partial v} \left[ -\frac{n}{2} \ln(2 \pi v) \right] = -\frac{n}{2} \cdot \frac{1}{2 \pi v} \cdot 2 \pi = -\frac{n}{2 v}$
# * $\frac{\partial}{\partial v} \left[ - \frac{1}{2 v} \sum_{i=1}^n (x_i - \mu)^2 \right] = - \sum_{i=1}^n (x_i - \mu)^2 \cdot \frac{\partial}{\partial v} \left( \frac{1}{2} v^{-1} \right) = - \sum_{i=1}^n (x_i - \mu)^2 \cdot \left( -\frac{1}{2} v^{-2} \right) = \frac{1}{2 v^2} \sum_{i=1}^n (x_i - \mu)^2$
#
# נשווה לאפס:
#
# $$
# -\frac{n}{2 v} + \frac{1}{2 v^2} \sum_{i=1}^n (x_i - \mu)^2 = 0
# $$
#
# נכפיל ב-$2v^2$ ונסדר:
#
# $$
# -n v + \sum_{i=1}^n (x_i - \mu)^2 = 0
# $$
#
# נבודד את $v$ (שהוא $\sigma^2$):
#
# $$
# \hat{v} = \hat{\sigma}^2_{MLE} = \frac{1}{n} \sum_{i=1}^n (x_i - \mu)^2
# $$
#
# נציב את אומדן ה-MLE של $\mu$ שמצאנו ($\hat{\mu}_{MLE} = \bar{x}$) במקום $\mu$:
#
# $$
# \hat{\sigma}^2_{MLE} = \frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})^2
# $$
#
# ---
#
# ## **מסקנה:**
#
# אומדי הנראות המקסימליים של הפרמטרים $\mu$ ו-$\sigma^2$ עבור התפלגות נורמלית הם:
#
# * **אומדן $\mu$ (הממוצע):**
# $$
# \hat{\mu}_{MLE} = \frac{1}{n} \sum_{i=1}^n x_i
# $$
# * **אומדן $\sigma^2$ (השונות):**
#     $$
#     \hat{\sigma}^2_{MLE} = \frac{1}{n} \sum_{i=1}^n (x_i - \bar{x})^2
#     $$
#
# ---

# %% [markdown] id="ehrMCHcDKsTO"
# # מבחני KS לצורך אישור בדיקת התאמת ההתפלגויות

# %% id="xypx_beZPz3o" colab={"base_uri": "https://localhost:8080/"} outputId="1d911f9c-4e75-4f2d-a6f1-015d5d394fbf"
# =====================================
# 5. KS Goodness-of-Fit Tests
# =====================================

# KS for exponential – Small Slide
ks_stat_small_exp, p_value_small_exp = kstest(small_data, 'expon', args=(0, 1/lambda_small_mle))

decision_small = "Reject H0" if p_value_small_exp < alpha else "Fail to reject H0"

print("Kolmogorov-Smirnov test – SMALL vs Exponential:")
print(f"KS - D statistic = {ks_stat_small_exp:.6f}")
print(f"p-value      = {p_value_small_exp:.6f}")
print(f"Decision at alpha={alpha}: {decision_small}")
print("\n" + "="*60 + "\n")

# KS for normal – Big Slide
ks_stat_big_norm, p_value_big_norm = kstest(big_data, 'norm', args=(mu_big_mle, sigma_big_mle))

decision_big = "Reject H0" if p_value_big_norm < alpha else "Fail to reject H0"

print("Kolmogorov-Smirnov test – BIG vs Normal:")
print(f"KS - D statistic = {ks_stat_big_norm:.6f}")
print(f"p-value      = {p_value_big_norm:.6f}")
print(f"Decision at alpha={alpha}: {decision_big}")


# %% [markdown] id="g4M5NQgfK2SO"
# ## מבחן KS עבור בדיקת התאמה של זמן ההתגלשות במגלשה הקטנה להתפלגות אקפוננציאלית
#
# הסטטיסטי שיצא במבחן:
#
# $D = 0.068455$
#
# מובהקות התוצאה:
#
# $P_{\text{value}} = 0.710573$
#
# רמת מובהקות המבחן:
#
# 10%
#
# מכיוון ש־$0.1 < P_{\text{value}}$, **אין לנו מספיק עדות לדחות את השערת האפס**, ולכן ניתן לומר שהנתונים מתאימים להתפלגות אקספוננציאלית עם הפרמטר שחושב.
#
#
# ---
#
#
#

# %% [markdown] id="mchCOv5jOdc6"
# ## מבחן KS עבור בדיקת התאמה של זמן ההתגלשות במגלשה הגדולה להתפלגות נורמלית
#
# הסטטיסטי שיצא במבחן:
#
# $D = 0.036953$
#
# מובהקות התוצאה:
#
# $P_{\text{value}} = 0.998451$
#
# רמת מובהקות המבחן:
#
# 10%
#
# מכיוון ש־$0.1 < P_{\text{value}}$, **אין לנו מספיק עדות לדחות את השערת האפס**, ולכן ניתן לומר שהנתונים מתאימים להתפלגות נורמלית עם הפרמטרים שחושבו.
#
#
# ---
#
#
#

# %% [markdown] id="lHnPXaPxkvQU"
# # Sampling Algorithms

# %% [markdown] id="DQ-C3L57E_E1"
# ### 1. Distribution of the Number of Children in a Family
# Let $X$ be the age, which follows a discrete uniform distribution
# for the integer range $[a, b]$, where: $a=1, b=5$
#
#
#
# **Total number of outcomes:**
# $$
# n = b - a + 1 = 5 - 1 + 1 = 5
# $$
#
# **Probability for each individual value:**
# $$
# P(X=k) = \frac{1}{n} = \frac{1}{5}
# $$
#
# **The Transformation:**
# To obtain discrete values from the continuous variable $U$, we divide the interval $[0,1]$ into $n$ equal parts and use the **floor function** to ensure rounding down (as the number of children must be an integer):
#
# $$
# x = a + \lfloor (b - a + 1) \cdot U \rfloor
# $$
#
# For the given data ($a=1, n=5$):
# $$
# x = 1 + \lfloor 5U \rfloor
# $$
#
# ## Sampling Algorithm:
# 1. Sample $U \sim U[0,1)$.
# 2. Return $x = 1 + \lfloor 5U \rfloor$.
#
#
# ##

# %% [markdown] id="W9OCYXdJBSa9"
# ### 2. Distribution Of Children's Age
# Let $X$ be the age, which follows a Continuous Uniform Distribution
# for the range $[a, b]$ where: $a=2, b=18$
#
#
# **Probability Density Function (PDF):**
# $$
# f(x) = \frac{1}{b - a} = \frac{1}{18-2}= \frac{1}{16}
# $$
#
# **Cumulative Distribution Function (CDF) for Continuous Uniform:**
# $$
# F(x) = \frac{x - a}{b - a} = u
# $$
#
# **Solving for $x$:**
# $$
# x - a = u(b - a) \\
# x = a + (b - a)u
# $$
#
# For the given values ($a=2, b=18$):
# $$
# x = 2 + 16u
# $$
#
# ## Sampling Algorithm:
# 1. Sample $U \sim [0,1]$.
# 2. Return $x = 2 + 16u$.
#
# ##

# %% [markdown] id="Rg7RHuGwISCZ"
# ##3. Family Arrival Rate
#
# ## Probability Density Function
# Let $X$ be the time until the family leaves, following an Exponential distribution with rate parameter $\lambda = \frac{3}{2}$.
# $$
# f(x) = \begin{cases}
# \frac{3}{2}e^{-\frac{3}{2}x} & x \ge 0 \\
# 0 & \text{else}
# \end{cases}
# $$
#
# ## Calculating the Cumulative Distribution Function:
# We calculate the CDF by integrating the PDF from the lower bound ($0$) to $x$:
#
#
# $$
# F(x) = \int_{0}^{x} \frac{3}{2}e^{-\frac{3}{2}t} \, dt = \left[ -e^{-\frac{3}{2}t} \right]_{0}^{x} = \left( -e^{-\frac{3}{2}x} \right) - \left( -e^{0} \right) = 1 - e^{-\frac{3}{2}x}
# $$
#
#
#
# ### Finding $x$ (Inverse Transform)
# To generate samples, we find the inverse function by setting $F(x) = u$ and solving for $x$:
#
# $$
# 1 - e^{-\frac{3}{2}x} = u
# $$
#
#
# $$
# e^{-\frac{3}{2}x} = 1 - u
# $$
#
# Take the natural logarithm ($\ln$) of both sides:
# $$
# -\frac{3}{2}x = \ln(1 - u)
# $$
#
#
# $$
# x = -\frac{2}{3}\ln(1 - u)
# $$
#
#
#
# ## Sampling Algorithm
# 1.Generate a random number $U$ from a uniform distribution: $U \sim [0,1]$.
#
# 2.Calculate $x$ using the derived formula: $$x = -\frac{2}{3}\ln(1 - U)$$
#    
#
# ##

# %% [markdown] id="TmmLGJ1T-tne"
# ## 4. Family Leaving Time
# ## Probability Density Function
# The PDF as given:
#
# $$
# f(x) = \begin{cases}
# \frac{2}{9}(x - 16) & 16 \le x \le 19 \\
# 0 & \text{else}
# \end{cases}
# $$
#
# ## Calculating the Cumulative Distribution Function
# We calculate the CDF by integrating the PDF from the lower bound (16) to $x$.
# The integration steps are condensed below:
#
# $$
# F(x) = \int_{16}^{x} \frac{2}{9}(t - 16) \, dt = \left[ \frac{t^2}{9} - \frac{32t}{9} \right]_{16}^{x} = \left( \frac{x^2}{9} - \frac{32x}{9} \right) - \left( \frac{256}{9} - \frac{512}{9} \right) = \frac{x^2}{9} - \frac{32x}{9} + \frac{256}{9}
# $$
#
# ---
#
# ### Finding $x$ (Inverse Transform)
# To find the inverse function, we set $F(x) = u$ and solve for $x$.
# $$
# \frac{x^2}{9} - \frac{32x}{9} + \frac{256}{9} = u \quad \Rightarrow \quad x^2 - 32x + (256 - 9u) = 0
# $$
#
# Using the quadratic formula (selecting the positive root since $x \ge 16$):
# $$
# x = \frac{32 + \sqrt{(-32)^2 - 4(1)(256 - 9u)}}{2} = \frac{32 + \sqrt{36u}}{2} = \frac{32 + 6\sqrt{u}}{2}
# $$
#
# $$
# x = 16 + 3\sqrt{u}
# $$
#
# ## Sampling Algorithm
# 1. Generate $U$ from a uniform distribution: $U \sim [0,1]$.
# 2. Calculate $x$ using the derived formula:  $$x = 16 + 3\sqrt{U}$$
#
# ###

# %% [markdown] id="qjNIx-oNMz6J"
# ###5. Number Of Teenagers In Group
# ## Sampling Algorithm
#
# 1. Generate $U$ from a uniform distribution: $U \sim [0,1]$.
#   
#
# 2. Determine the value of $X$ based on the interval in which $U$ falls:
#
#    * If $0 \le U \le 0.20$ $\Rightarrow$ Return $X = 2$
#    * Else If $0.20 < U \le 0.4$ $\Rightarrow$ Return $X = 3$
#    * Else If $0.40 < U \le 0.65$ $\Rightarrow$ Return $X = 4$
#    * Else If $0.65 < U \le 0.90$ $\Rightarrow$ Return $X = 5$
#    * Else  $\Rightarrow$ Return $X = 6$
#
#
#    ##

# %% [markdown] id="Zzw66pt6OZKc"
# ###6. Teenagers Arrival Rate
#
# ## Probability Density Function
# Let $X$ be the time teengares arrive, following an Exponential distribution with rate parameter $\lambda = \frac{18}{25}$.
# $$
# f(x) = \begin{cases}
# \frac{18}{25}e^{-\frac{18}{25}x} & x \ge 0 \\
# 0 & \text{else}
# \end{cases}
# $$
#
# ## Calculating the Cumulative Distribution Function:
# We calculate the CDF by integrating the PDF from the lower bound ($0$) to $x$:
#
# $$
# F(x) = \int_{0}^{x} \frac{18}{25}e^{-\frac{18}{25}t} \, dt = \left[ -e^{-\frac{18}{25}t} \right]_{0}^{x} = \left( -e^{-\frac{18}{25}x} \right) - \left( -e^{0} \right) = 1 - e^{-\frac{18}{25}x}
# $$
#
# ### Finding $x$ (Inverse Transform)
# To generate samples, we find the inverse function by setting $F(x) = u$ and solving for $x$:
#
# $$
# 1 - e^{-\frac{18}{25}x} = u
# $$
#
# $$
# e^{-\frac{18}{25}x} = 1 - u
# $$
#
# Take the natural logarithm ($\ln$) of both sides:
# $$
# -\frac{18}{25}x = \ln(1 - u)
# $$
#
# $$
# x = -\frac{25}{18}\ln(1 - u)
# $$
#
# ## Sampling Algorithm
# 1.Generate a random number $U$ from a uniform distribution: $U \sim [0,1]$.
#
# 2.Calculate $x$ using the derived formula: $$x = -\frac{25}{18}\ln(1 - U)$$
#
# ##

# %% [markdown] id="B0xymUPHQB4U"
# ###7. Solo Visitors Arrival Rate
#
#
# ## Probability Density Function
# Let $X$ be the time solo visitors arrive, following an Exponential distribution with rate parameter $\lambda = \frac{3}{2}$.
# $$
# f(x) = \begin{cases}
# \frac{3}{2}e^{-\frac{3}{2}x} & x \ge 0 \\
# 0 & \text{else}
# \end{cases}
# $$
#
# ## Calculating the Cumulative Distribution Function:
# We calculate the CDF by integrating the PDF from the lower bound ($0$) to $x$:
#
#
# $$
# F(x) = \int_{0}^{x} \frac{3}{2}e^{-\frac{3}{2}t} \, dt = \left[ -e^{-\frac{3}{2}t} \right]_{0}^{x} = \left( -e^{-\frac{3}{2}x} \right) - \left( -e^{0} \right) = 1 - e^{-\frac{3}{2}x}
# $$
#
#
#
# ### Finding $x$ (Inverse Transform)
# To generate samples, we find the inverse function by setting $F(x) = u$ and solving for $x$:
#
# $$
# 1 - e^{-\frac{3}{2}x} = u
# $$
#
#
# $$
# e^{-\frac{3}{2}x} = 1 - u
# $$
#
# Take the natural logarithm ($\ln$) of both sides:
# $$
# -\frac{3}{2}x = \ln(1 - u)
# $$
#
#
# $$
# x = -\frac{2}{3}\ln(1 - u)
# $$
#
#
#
# ## Sampling Algorithm
# 1.Generate a random number $U$ from a uniform distribution: $U \sim [0,1]$.
#
# 2.Calculate $x$ using the derived formula: $$x = -\frac{2}{3}\ln(1 - U)$$
#    
# ##

# %% [markdown] id="NaPyfUG5SlIC"
# ###8. Buying A Ticket
#
# Let $X$ be the time buying a ticket, which follows a Continuous Uniform Distribution for the range $[a, b]$ where: $a=5, b=10$
#
#
# Probability Density Function :
# $$
# f(x) = \frac{1}{b - a} = \frac{1}{10-5}= \frac{1}{5}
# $$
#
# **Cumulative Distribution Function (CDF) for Continuous Uniform:**
# $$
# F(x) = \frac{x - a}{b - a} = u
# $$
#
# **Solving for $x$:**
# $$
# x - a = u(b - a) \\
# x = a + (b - a)u
# $$
#
# For the given values ($a=5, b=10$):
# $$
# x = 5 + 10u
# $$
#
# ## Sampling Algorithm:
# 1. Sample $U \sim [0,1]$.
# 2. Return $x = 5 + 10u$.
# ##

# %% [markdown] id="vcbp5vhPTOT4"
# ###9. Bracelet Time
#
#
# ## Probability Density Function
# Let $X$ be the time until the family leaves, following an Exponential distribution with rate parameter $\lambda = \frac{1}{2}$.
# $$
# f(x) = \begin{cases}
# \frac{1}{2}e^{-\frac{1}{2}x} & x \ge 0 \\
# 0 & \text{else}
# \end{cases}
# $$
#
# ## Calculating the Cumulative Distribution Function:
# We calculate the CDF by integrating the PDF from the lower bound ($0$) to $x$:
#
# $$
# F(x) = \int_{0}^{x} \frac{1}{2}e^{-\frac{1}{2}t} \, dt = \left[ -e^{-\frac{1}{2}t} \right]_{0}^{x} = \left( -e^{-\frac{1}{2}x} \right) - \left( -e^{0} \right) = 1 - e^{-\frac{1}{2}x}
# $$
#
# ### Finding $x$ (Inverse Transform)
# To generate samples, we find the inverse function by setting $F(x) = u$ and solving for $x$:
#
# $$
# 1 - e^{-\frac{1}{2}x} = u
# $$
#
# $$
# e^{-\frac{1}{2}x} = 1 - u
# $$
#
# Take the natural logarithm ($\ln$) of both sides:
# $$
# -\frac{1}{2}x = \ln(1 - u)
# $$
#
# $$
# x = -2\ln(1 - u)
# $$
#
# ## Sampling Algorithm
# 1.Generate a random number $U$ from a uniform distribution: $U \sim [0,1]$.
#
# 2.Calculate $x$ using the derived formula: $$x = -2\ln(1 - U)$$
#
# ##

# %% [markdown] id="tD_lVN2TZdko"
# ## 12. Small Tube Watersilde
# Let $X$ be the ride's sliding time, following an Exponential distribution with rate parameter $\lambda = 2.107060$.
# $$
# f(x) = \begin{cases}
# 2.107060e^{-2.107060x} & x \ge 0 \\
# 0 & \text{else}
# \end{cases}
# $$
#
# ## Calculating the Cumulative Distribution Function:
# We calculate the CDF by integrating the PDF from the lower bound ($0$) to $x$:
#
#
# $$
# F(x) = \int_{0}^{x} 2.107060e^{-2.107060t} \, dt = \left[ -e^{-2.107060t} \right]_{0}^{x} = \left( -e^{-2.107060x} \right) - \left( -e^{0} \right) = 1 - e^{-2.107060x}
# $$
#
#
#
# ### Finding $x$ (Inverse Transform)
# To generate samples, we find the inverse function by setting $F(x) = u$ and solving for $x$:
#
# $$
# 1 - e^{-2.107060x} = u
# $$
#
#
# $$
# e^{-2.107060x} = 1 - u
# $$
#
# Take the natural logarithm ($\ln$) of both sides:
# $$
# -2.107060x = \ln(1 - u)
# $$
#
#
# $$
# x = -0.474595\ln(1 - u)
# $$
#
#
#
# ## Sampling Algorithm
# 1.Generate a random number $U$ from a uniform distribution: $U \sim [0,1]$.
#
# 2.Calculate $x$ using the derived formula: $$x = -0.474595\ln(1 - U)$$

# %% [markdown] id="GfXkFIkuCXjU"
# ## 10. LazyRiver
#
#
# Let $X$ be the time riding the Lazy River ride, which follows a Continuous Uniform Distribution
# for the range $[a, b]$ where: $a=2, b=18$
#
#
# **Probability Density Function (PDF):**
# $$
# f(x) = \frac{1}{b - a} = \frac{1}{30-20}= \frac{1}{10}
# $$
#
# **Cumulative Distribution Function (CDF) for Continuous Uniform:**
# $$
# F(x) = \frac{x - a}{b - a} = u
# $$
#
# **Solving for $x$:**
# $$
# x - a = u(b - a) \\
# x = a + (b - a)u
# $$
#
# For the given values ($a=20, b=30$):
# $$
# x = 20 + 10u
# $$
#
# ### Sampling Algorithm:
# 1. Sample $U \sim [0,1]$.
# 2. Return $x = 20 + 10u$.
#
#
#
#
#
#
#

# %% [markdown] id="izno8-yrY-N5"
# ### 11. Big Tube Watersilde
#
# Let $X$ be the time (minutes), which follows a **Normal Distribution** $N(\mu, \sigma^2)$ where: $\mu=4.800664, \sigma=1.823101$.
#
# **Probability Density Function (PDF):**
# $$
# f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{1}{2} \left(\frac{x - \mu}{\sigma}\right)^2}
# $$
# For the given values ($\mu=4.800664, \sigma=1.823101$):
# $$
# f(x) = \frac{1}{1.823101 \sqrt{2\pi}} e^{-\frac{1}{2} \left(\frac{x - 4.800664}{1.823101}\right)^2}
# $$
#
# **Cumulative Distribution Function (CDF) for Normal:**
# The CDF is given by the formula involving the standard normal CDF, $\Phi(z)$:
# $$
# F(x) = P(X \le x) = \Phi(z) \text{ where } z = \frac{x - \mu}{\sigma}
# $$
#
# **Transformation for Sampling:**
# To sample $X$ from $N(\mu, \sigma^2)$, we transform a sample $Z$ from the Standard Normal Distribution $N(0, 1)$ using the relationship:
# $$
# X = \mu + \sigma Z
# $$
#
# For the given values ($\mu=4.800664, \sigma=1.823101$):
# $$
# X = 4.800664 + 1.823101 Z
# $$
#
# **Box-Muller Transform for Standard Normal:**
# To generate $Z \sim N(0, 1)$, we use the Box-Muller transform with two independent uniform random variables $U_1, U_2 \sim U[0,1]$:
# $$
# Z = \sqrt{-2\ln(U_1)} \cdot \sin(2\pi U_2)
# $$
#
# ## Sampling Algorithm:
# 1. Generate $U_1, U_2 \sim U[0,1]$.
# 2. Calculate $Z = \sqrt{-2\ln(U_1)} \cdot \sin(2\pi U_2)$.
# 3. Return $x = 4.800664 + 1.823101 Z$.
#

# %% [markdown] id="Ao2nKc33VMTK"
# ###12. Making a Hamburger
#
# Let $X$ be the time to make a hamburger, which follows a Continuous Uniform Distribution for the range $[a, b]$ where: $a=3, b=4$
#
#
# Probability Density Function :
# $$
# f(x) = \frac{1}{b - a} = \frac{1}{4-3}= 1
# $$
#
# **Cumulative Distribution Function (CDF) for Continuous Uniform:**
# $$
# F(x) = \frac{x - a}{b - a} = u
# $$
#
# **Solving for $x$:**
# $$
# x - a = u(b - a) \\
# x = a + (b - a)u
# $$
#
# For the given values ($a=3, b=4$):
# $$
# x = 3 + (4 - 3)u \\
# x = 3 + u
# $$
#
# ## Sampling Algorithm:
# 1. Sample $U \sim [0,1]$.
# 2. Return $x = 3 + u$.
#
# ##

# %% [markdown] id="02DnLt3kaPGj"
# ## 13. WavePool
#
# ### Acceptance–Rejection Sampling
# Because the WavePool ride time follows a piecewise PDF that does not allow a closed-form inverse CDF, we use the **Acceptance–Rejection method** to generate valid samples.
#
# The target PDF \( f(x) \) is defined as:
# $$
# f(x) = \begin{cases}
# \frac{X}{2700} & 0 \le x \le 30 \\
# \frac{60-X}{2700}+ \frac{1}{30} & 30 \le x \le 50 \\
# \frac{60-X}{2700} & 50 \le x \le 60 \\
# 0 & \text{else}
# \end{cases}
# $$
#
# The maximum value of \( f(x) \) occurs at \( x = 30 \):
#
# $$
# t(x) =  f_{\max} = f(30^+) = \frac{2}{45}
# $$
#
# ### Bounding Constant \( c \)
#
# We require a constant \( c \) such that:
# $$
# f(x) \le c\, g(x)
# $$
#
# Since
# $$
# ( g(x) = \frac{1}{60} )$$ and $$( \max f(x) = \frac{2}{45} )$$, we get:
#
#
# $$
# c = 60 \cdot \frac{2}{45} = \frac{8}{3}
# $$
#
#
#
#
# ### Acceptance–Rejection Algorithm
# 1. Sample $$( U1 \sim [0, 1] ) $$
# and set $$X = 60 \cdot U_1$$
# 2. Sample $$( U2 \sim [0, 1] ) $$
# 3. Compute $$( f(X) )$$ using the piecewise definition.  
# 4. Accept $$( X ) $$ if:
#    $$[
#    U2 \le \frac{f(X)}{c\, g(X)} = \frac{f(X)}{2/45} = \frac{45}{2}\, f(X)
#    ]$$
# 5. If rejected, repeat from step 1.
#

# %% [markdown] id="lf-vBRVuaZEO"
# ## 14. Kids Pool
#
# ### Probability Density Function
# The probability density function (PDF) is defined as:
#
# $$
# f(x)=
# \begin{cases}
# \frac{16}{3}(x-1) & 1 \le x \le 1.25 \\[6pt]
# \frac{4}{3} & 1.25 < x \le 1.75 \\[6pt]
# \frac{16}{3}(2-x) & 1.75 < x \le 2 \\[6pt]
# 0 & \text{else}
# \end{cases}
# $$
#
#
#
# ### Calculating the Cumulative Distribution Function
#
# #### Case 1: \(1 \le x \le 1.25\)
#
# $$
# F(x)=\int_{1}^{x}\frac{16}{3}(t-1)\,dt
# =\frac{8}{3}(x-1)^2
# $$
#
#
#
# #### Case 2: \(1.25 < x \le 1.75\)
#
# $$
# F(1.25)=\frac{8}{3}(0.25)^2=\frac{1}{6}
# $$
#
# $$
# F(x)=\frac{1}{6}+\int_{1.25}^{x}\frac{4}{3}\,dt
# =\frac{1}{6}+\frac{4}{3}(x-1.25)
# $$
#
#
#
# #### Case 3: \(1.75 < x \le 2\)
#
# $$
# F(1.75)=\frac{1}{6}+\frac{4}{3}(0.5)=\frac{5}{6}
# $$
#
# $$
# F(x)=\frac{5}{6}+\int_{1.75}^{x}\frac{16}{3}(2-t)\,dt
# $$
#
#
#
# ### Finding \(x\) (Inverse Transform)
#
# The distribution is sampled using a mixture approach.
#
# $$
# p_1=\frac{1}{6}, \quad
# p_2=\frac{2}{3}, \quad
# p_3=\frac{1}{6}
# $$
#
#
#
# #### Segment 1: \(1 \le x \le 1.25\)
#
# $$
# F_1(x)=16(x-1)^2
# $$
#
# $$
# x=1+\frac{1}{4}\sqrt{u}
# $$
#
#
#
# #### Segment 2: \(1.25 < x \le 1.75\)
#
# $$
# F_2(x)=2(x-1.25)
# $$
#
# $$
# x=1.25+\frac{u}{2}
# $$
#
#
#
# #### Segment 3: \(1.75 < x \le 2\)
#
# $$
# F_3(x)=1-16(2-x)^2
# $$
#
# $$
# x=2-\frac{1}{4}\sqrt{1-u}
# $$
#
#
#
# ### Sampling Algorithm
#
# 1. Generate \(U \sim \text{Uniform}(0,1)\).
# 2. If \(U < \frac{1}{6}\):
#    - Set \(u = 6U\)
#    - Sample:
#      $$
#      x = 1 + \frac{1}{4}\sqrt{u}
#      $$
# 3. Else if \(U < \frac{5}{6}\):
#    - Set:
#      $$
#      u = \frac{3}{2}\left(U - \frac{1}{6}\right)
#      $$
#    - Sample:
#      $$
#      x = 1.25 + \frac{u}{2}
#      $$
# 4. Else:
#    - Set:
#      $$
#      u = 6\left(U - \frac{5}{6}\right)
#      $$
#    - Sample:
#      $$
#      x = 2 - \frac{1}{4}\sqrt{1-u}
#      $$
#

# %% [markdown] id="gHlq0fgIahbq"
# ## 15. Snorkeling Tour
#
# ### Distribution Of Tour's Time
#
# Let $X$ be the time (minutes) , which follows a **Normal Distribution** $N(\mu, \sigma^2)$ where: $\mu=30, \sigma=10$.
#
# **Probability Density Function (PDF):**
# $$
# f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{1}{2} \left(\frac{x - \mu}{\sigma}\right)^2}
# $$
# For the given values ($\mu=30, \sigma=10$):
# $$
# f(x) = \frac{1}{10 \sqrt{2\pi}} e^{-\frac{1}{2} \left(\frac{x - 30}{10}\right)^2}
# $$
#
# **Cumulative Distribution Function (CDF) for Normal:**
# The CDF is given by the formula involving the standard normal CDF, $\Phi(z)$:
# $$
# F(x) = P(X \le x) = \Phi(z) \text{ where } z = \frac{x - \mu}{\sigma}
# $$
#
# **Transformation for Sampling:**
# To sample $X$ from $N(\mu, \sigma^2)$, we transform a sample $Z$ from the Standard Normal Distribution $N(0, 1)$ using the relationship:
# $$
# X = \mu + \sigma Z
# $$
#
# For the given values ($\mu=30, \sigma=10$):
# $$
# X = 30 + 10 Z
# $$
#
# **Box-Muller Transform for Standard Normal:**
# To generate $Z \sim N(0, 1)$, we use the Box-Muller transform with two independent uniform random variables $U_1, U_2 \sim U[0,1]$:
# $$
# Z = \sqrt{-2\ln(U_1)} \cdot \sin(2\pi U_2)
# $$
#
# ## Sampling Algorithm:
# 1. Generate $U_1, U_2 \sim U[0,1]$.
# 2. Calculate $Z = \sqrt{-2\ln(U_1)} \cdot \sin(2\pi U_2)$.
# 3. Return $x = 30 + 10 Z$.

# %% [markdown] id="GwOSuroaVkAh"
# ###13. Making a Salad
#
# Let $X$ be the time to make a salad, which follows a Continuous Uniform Distribution for the range $[a, b]$ where: $a=3, b=7$
#
#
# Probability Density Function :
# $$
# f(x) = \frac{1}{b - a} = \frac{1}{7-3}= \frac{1}{4}
# $$
#
# **Cumulative Distribution Function (CDF) for Continuous Uniform:**
# $$
# F(x) = \frac{x - a}{b - a} = u
# $$
#
# **Solving for $x$:**
# $$
# x - a = u(b - a) \\
# x = a + (b - a)u
# $$
#
# For the given values ($a=3, b=7$):
# $$
# x = 3 + (7 - 3)u \\
# x = 3 + 4u
# $$
#
# ## Sampling Algorithm:
# 1. Sample $U \sim [0,1]$.
# 2. Return $x = 3 + 4u$.
#
# ##

# %% [markdown] id="fdvOjD03XrLF"
# ### 14. Service Time
#
# Let $X$ be the service time, which follows a **Normal Distribution** $N(\mu, \sigma^2)$ where: $\mu=5, \sigma=1.5$.
#
# **Probability Density Function (PDF):**
# $$
# f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{1}{2} \left(\frac{x - \mu}{\sigma}\right)^2}
# $$
# For the given values ($\mu=5, \sigma=1.5$):
# $$
# f(x) = \frac{1}{1.5 \sqrt{2\pi}} e^{-\frac{1}{2} \left(\frac{x - 5}{1.5}\right)^2}
# $$
#
# **Cumulative Distribution Function (CDF) for Normal:**
# The CDF is given by the formula involving the standard normal CDF, $\Phi(z)$:
# $$
# F(x) = P(X \le x) = \Phi(z) \text{ where } z = \frac{x - \mu}{\sigma}
# $$
#
# **Transformation for Sampling:**
# To sample $X$ from $N(\mu, \sigma^2)$, we transform a sample $Z$ from the Standard Normal Distribution $N(0, 1)$ using the relationship:
# $$
# X = \mu + \sigma Z
# $$
#
# For the given values ($\mu=5, \sigma=1.5$):
# $$
# X = 5 + 1.5 Z
# $$
#
# ## Sampling Algorithm:
# 1. Sample $Z$ from a Standard Normal Distribution $N(0, 1)$.
# 2. Return $x = 5 + 1.5 Z$.
#
# ##

# %% [markdown] id="dmEp-bZCW9L7"
# ##16. Eating Time
#
# Let $X$ be the time to eat, which follows a Continuous Uniform Distribution for the range $[a, b]$ where: $a=15, b=35$
#
# **Probability Density Function :
# $$
# f(x) = \frac{1}{b - a} = \frac{1}{35-15}= \frac{1}{20}
# $$
# ]
#
# Cumulative Distribution Function  for Continuous Uniform:
# $$
# F(x) = \frac{x - a}{b - a} = u
# $$
#
# Solving for $x$:
# $$
# x - a = u(b - a) \\
# x = a + (b - a)u
# $$
#
# For the given values ($a=15, b=35$):
# $$
# x = 15 + (35 - 15)u \\
# x = 15 + 20u
# $$
#
# ## Sampling Algorithm:
# 1. Sample $U \sim [0,1]$.
# 2. Return $x = 15 + 20u$.

# %% [markdown] id="bFZoArfjLdzn"
# ###17. Pizza Making Time
#
# Let $X$ be the time to make a pizza, which follows a Continuous Uniform Distribution for the range $[a, b]$ where: $a=4, b=6$
#
#
# Probability Density Function :
# $$
# f(x) = \frac{1}{b - a} = \frac{1}{6-4}= \frac{1}{2}
# $$
#
# **Cumulative Distribution Function (CDF) for Continuous Uniform:**
# $$
# F(x) = \frac{x - a}{b - a} = u
# $$
#
# **Solving for $x$:**
# $$
# x - a = u(b - a) \\
# x = a + (b - a)u
# $$
#
# For the given values ($a=4, b=6$):
# $$
# x = 4 + (6 - 4)u \\
# x = 4 + 2u
# $$
#
# ## Sampling Algorithm:
# 1. Sample $U \sim [0,1]$.
# 2. Return $x = 4 + 2u$.

# %% [markdown] id="QyBodmDxL6Uc"
# ###18. Making a Hamburger
#
# Let $X$ be the time to make a hamburger, which follows a Continuous Uniform Distribution for the range $[a, b]$ where: $a=3, b=4$
#
#
# Probability Density Function :
# $$
# f(x) = \frac{1}{b - a} = \frac{1}{4-3}= 1
# $$
#
# **Cumulative Distribution Function (CDF) for Continuous Uniform:**
# $$
# F(x) = \frac{x - a}{b - a} = u
# $$
#
# **Solving for $x$:**
# $$
# x - a = u(b - a) \\
# x = a + (b - a)u
# $$
#
# For the given values ($a=3, b=4$):
# $$
# x = 3 + (4 - 3)u \\
# x = 3 + u
# $$
#
# ## Sampling Algorithm:
# 1. Sample $U \sim [0,1]$.
# 2. Return $x = 3 + u$.

# %% [markdown] id="LQlE31KiMtO4"
# ###19. Making a Salad
#
# Let $X$ be the time to make a salad, which follows a Continuous Uniform Distribution for the range $[a, b]$ where: $a=3, b=7$
#
#
# Probability Density Function :
# $$
# f(x) = \frac{1}{b - a} = \frac{1}{7-3}= \frac{1}{4}
# $$
#
# **Cumulative Distribution Function (CDF) for Continuous Uniform:**
# $$
# F(x) = \frac{x - a}{b - a} = u
# $$
#
# **Solving for $x$:**
# $$
# x - a = u(b - a) \\
# x = a + (b - a)u
# $$
#
# For the given values ($a=3, b=7$):
# $$
# x = 3 + (7 - 3)u \\
# x = 3 + 4u
# $$
#
# ## Sampling Algorithm:
# 1. Sample $U \sim [0,1]$.
# 2. Return $x = 3 + 4u$.

# %% [markdown] id="iYXUGUoEM_FN"
# ### 20. Service Time
#
# Let $X$ be the age, which follows a **Normal Distribution** $N(\mu, \sigma^2)$ where: $\mu=5, \sigma=1.5$.
#
# **Probability Density Function (PDF):**
# $$
# f(x) = \frac{1}{\sigma \sqrt{2\pi}} e^{-\frac{1}{2} \left(\frac{x - \mu}{\sigma}\right)^2}
# $$
# For the given values ($\mu=5, \sigma=1.5$):
# $$
# f(x) = \frac{1}{1.5 \sqrt{2\pi}} e^{-\frac{1}{2} \left(\frac{x - 5}{1.5}\right)^2}
# $$
#
# **Cumulative Distribution Function (CDF) for Normal:**
# The CDF is given by the formula involving the standard normal CDF, $\Phi(z)$:
# $$
# F(x) = P(X \le x) = \Phi(z) \text{ where } z = \frac{x - \mu}{\sigma}
# $$
#
# **Transformation for Sampling:**
# To sample $X$ from $N(\mu, \sigma^2)$, we transform a sample $Z$ from the Standard Normal Distribution $N(0, 1)$ using the relationship:
# $$
# X = \mu + \sigma Z
# $$
#
# For the given values ($\mu=5, \sigma=1.5$):
# $$
# X = 5 + 1.5 Z
# $$
#
# **Box-Muller Transform for Standard Normal:**
# To generate $Z \sim N(0, 1)$, we use the Box-Muller transform with two independent uniform random variables $U_1, U_2 \sim U[0,1]$:
# $$
# Z = \sqrt{-2\ln(U_1)} \cdot \sin(2\pi U_2)
# $$
#
# ## Sampling Algorithm:
# 1. Generate $U_1, U_2 \sim U[0,1]$.
# 2. Calculate $Z = \sqrt{-2\ln(U_1)} \cdot \sin(2\pi U_2)$.
# 3. Return $x = 5 + 1.5 Z$.

# %% [markdown] id="R7-Ebd-gNLo4"
# ###21. Eating Time
#
# Let $X$ be the time to eat, which follows a Continuous Uniform Distribution for the range $[a, b]$ where: $a=15, b=35$
#
# **Probability Density Function :
# $$
# f(x) = \frac{1}{b - a} = \frac{1}{35-15}= \frac{1}{20}
# $$
# ]
#
# Cumulative Distribution Function  for Continuous Uniform:
# $$
# F(x) = \frac{x - a}{b - a} = u
# $$
#
# Solving for $x$:
# $$
# x - a = u(b - a) \\
# x = a + (b - a)u
# $$
#
# For the given values ($a=15, b=35$):
# $$
# x = 15 + (35 - 15)u \\
# x = 15 + 20u
# $$
#
# ## Sampling Algorithm:
# 1. Sample $U \sim [0,1]$.
# 2. Return $x = 15 + 20u$.

# %% [markdown] id="R1OMJTU5WHbr"
# #Algorithm Class

# %% id="KaIgNRaGX7tX"
# The algorithm class
class Algorithm :

  ### algorithm for discrete distribution
  @staticmethod
  def sample_number_of_children(a,b):
    U = random.random()  # Returns a scalar from [0, 1)
    x = a + math.floor((b - a + 1) * U) #Uses transformation
    return x


  ### algorithm for continuous distribution
  @staticmethod
  def sample_continuous_uniform(a, b):
    U = random.random() # Returns a scalar from [0, 1)
    x = a + (b - a) * U # Uses transformation
    return x


  ### algorithm for exponential distribution
  @staticmethod
  def sample_exponential(lam):
    U = random.random() # Returns a scalar from [0, 1)
    x = -(1 / lam) * math.log(1 - U)  # Uses transformation
    return x

  ### algorithm for family leaving time distribution
  @staticmethod
  def sample_family_leaving_time():
    U = random.random()
    x = 16 + 3 * math.sqrt(U)# Uses transformation
    return x

  ### algorithm for number of teenagers
  @staticmethod
  def sample_number_of_teenagers():
      U = random.random()

      # match to each probability as described in instructions
      if U <= 0.20:
          return 2
      elif U <= 0.40:
          return 3
      elif U <= 0.65:
          return 4
      elif U <= 0.90:
          return 5
      else:
          return 6


  ### algorithm for standard normal distribution
  @staticmethod
  def sample_standard_normal():

    U1 = random.random()
    U2 = random.random()

    Z = math.sqrt(-2 * math.log(U1)) * math.sin(2 * math.pi * U2)
    return Z

  @staticmethod
  ### algorithm for  normal distribution
  def sample_normal(mu, sigma):
    #Sample from N(mu, sigma^2)

    Z = sample_standard_normal()
    X = mu + sigma * Z
    return X


  @staticmethod
  ### algorithm for wavepool time
  def generate_wavepool_time():
    while True:
      # 1. Sample U1 ~ Uniform(0, 1) and set X = 60 * U1
      U1 = random.random()
      X = 60 * U1

      # 2. Sample U2 ~ Uniform(0, 1)
      U2 = random.random()

      # 3. Compute f(X)
      f_x = wavepool_pdf(X)

      # 4. Acceptance condition: U2 <= (45/2) * f(X)
      # Note: 45/2 is the simplified version of 1 / (c * g(x))
      if U2 <= (45 / 2) * f_x:
          return X

  ### Calculating the wavepool pdf
  @staticmethod
  def wavepool_pdf(x):
    # Calculates the probability density f(x) based on the piecewise definition.
    if 0 <= x <= 30:
        return x / 2700
    elif 30 < x <= 50:
        return (60 - x) / 2700 + (1 / 30)
    elif 50 < x <= 60:
        return (60 - x) / 2700
    else:
        return 0


  ### Algorithm for kids pool time
  @staticmethod
  def generate_kids_pool_time():
    # 1. Generate Global U ~ Uniform(0, 1)
    U = random.random()

    # 2. Check Segment 1 (Probability 1/6)
    if U < (1/6):
      # Scale U to local u [0, 1]
      u_local = 6 * U
      # Inverse Transform for Segment 1
      x = 1 + 0.25 * math.sqrt(u_local)
      return x

    # 3. Check Segment 2 (Probability 4/6)
    # Cumulative probability threshold is 1/6 + 4/6 = 5/6
    elif U < (5/6):
      # Shift and Scale U to local u [0, 1]
      # (U - 1/6) shifts start to 0. Multiplying by 3/2 scales range (4/6) to 1.
      u_local = 1.5 * (U - (1/6))
      # Inverse Transform for Segment 2
      x = 1.25 + (u_local / 2)
      return x

    # 4. Check Segment 3 (Probability 1/6)
    else:
      # Shift and Scale U to local u [0, 1]
      u_local = 6 * (U - (5/6))
      # Inverse Transform for Segment 3
      x = 2 - 0.25 * math.sqrt(1 - u_local)
      return x



# %% [markdown] id="ep8tlF4Gkm00"
# # Implementation details

# %% [markdown] id="bvZtwDkaI81n"
# ## **:הסבר על המידול**
#
# ## **:תרשים אירועים**
# ![ProjectEventStateDiagram9.drawio.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAzIAAAIzCAYAAADbHiw8AAByR3RFWHRteGZpbGUAJTNDbXhmaWxlJTIwaG9zdCUzRCUyMmFwcC5kaWFncmFtcy5uZXQlMjIlMjBhZ2VudCUzRCUyMk1vemlsbGElMkY1LjAlMjAoV2luZG93cyUyME5UJTIwMTAuMCUzQiUyMFdpbjY0JTNCJTIweDY0KSUyMEFwcGxlV2ViS2l0JTJGNTM3LjM2JTIwKEtIVE1MJTJDJTIwbGlrZSUyMEdlY2tvKSUyMENocm9tZSUyRjE0My4wLjAuMCUyMFNhZmFyaSUyRjUzNy4zNiUyMiUyMHZlcnNpb24lM0QlMjIyOS4yLjklMjIlMjBzY2FsZSUzRCUyMjElMjIlMjBib3JkZXIlM0QlMjIwJTIyJTNFJTBBJTIwJTIwJTNDZGlhZ3JhbSUyMG5hbWUlM0QlMjJQYWdlLTElMjIlMjBpZCUzRCUyMkctbnBDVXpvZ2ZFeEJDeUNxUGItJTIyJTNFJTBBJTIwJTIwJTIwJTIwJTNDbXhHcmFwaE1vZGVsJTIwZHglM0QlMjI5ODMlMjIlMjBkeSUzRCUyMjE2MTglMjIlMjBncmlkJTNEJTIyMSUyMiUyMGdyaWRTaXplJTNEJTIyMTAlMjIlMjBndWlkZXMlM0QlMjIxJTIyJTIwdG9vbHRpcHMlM0QlMjIxJTIyJTIwY29ubmVjdCUzRCUyMjElMjIlMjBhcnJvd3MlM0QlMjIxJTIyJTIwZm9sZCUzRCUyMjElMjIlMjBwYWdlJTNEJTIyMSUyMiUyMHBhZ2VTY2FsZSUzRCUyMjElMjIlMjBwYWdlV2lkdGglM0QlMjI4NTAlMjIlMjBwYWdlSGVpZ2h0JTNEJTIyMTEwMCUyMiUyMG1hdGglM0QlMjIwJTIyJTIwc2hhZG93JTNEJTIyMCUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUzQ3Jvb3QlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteENlbGwlMjBpZCUzRCUyMjAlMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteENlbGwlMjBpZCUzRCUyMjElMjIlMjBwYXJlbnQlM0QlMjIwJTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhDZWxsJTIwaWQlM0QlMjJ4TGs3V0ZSUXNLQ2xFd0RjdThtWC0xNSUyMiUyMHBhcmVudCUzRCUyMjElMjIlMjBzdHlsZSUzRCUyMmVsbGlwc2UlM0J3aGl0ZVNwYWNlJTNEd3JhcCUzQmh0bWwlM0QxJTNCYXNwZWN0JTNEZml4ZWQlM0IlMjIlMjB2YWx1ZSUzRCUyMkVuZCUyMGx1bmNoJTIwZXZlbnQlMjIlMjB2ZXJ0ZXglM0QlMjIxJTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhHZW9tZXRyeSUyMGhlaWdodCUzRCUyMjgwJTIyJTIwd2lkdGglM0QlMjI4MCUyMiUyMHglM0QlMjI2NzAlMjIlMjB5JTNEJTIyNzQlMjIlMjBhcyUzRCUyMmdlb21ldHJ5JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGbXhDZWxsJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhDZWxsJTIwaWQlM0QlMjJ4TGs3V0ZSUXNLQ2xFd0RjdThtWC0xNyUyMiUyMHBhcmVudCUzRCUyMjElMjIlMjBzdHlsZSUzRCUyMmVsbGlwc2UlM0J3aGl0ZVNwYWNlJTNEd3JhcCUzQmh0bWwlM0QxJTNCYXNwZWN0JTNEZml4ZWQlM0IlMjIlMjB2YWx1ZSUzRCUyMkVuZCUyMG9mJTIwc2ltdWxhdGlvbiUyMiUyMHZlcnRleCUzRCUyMjElMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteEdlb21ldHJ5JTIwaGVpZ2h0JTNEJTIyODAlMjIlMjB3aWR0aCUzRCUyMjgwJTIyJTIweCUzRCUyMjQ0MCUyMiUyMHklM0QlMjI0NDAlMjIlMjBhcyUzRCUyMmdlb21ldHJ5JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGbXhDZWxsJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhDZWxsJTIwaWQlM0QlMjJ4TGs3V0ZSUXNLQ2xFd0RjdThtWC0yMCUyMiUyMGVkZ2UlM0QlMjIxJTIyJTIwcGFyZW50JTNEJTIyMSUyMiUyMHN0eWxlJTNEJTIyZW5kQXJyb3clM0RjbGFzc2ljJTNCaHRtbCUzRDElM0Jyb3VuZGVkJTNEMCUzQmVudHJ5WCUzRDAlM0JlbnRyeVklM0QwLjUlM0JlbnRyeUR4JTNEMCUzQmVudHJ5RHklM0QwJTNCJTIyJTIwdGFyZ2V0JTNEJTIyeExrN1dGUlFzS0NsRXdEY3U4bVgtMTclMjIlMjB2YWx1ZSUzRCUyMiUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214R2VvbWV0cnklMjBoZWlnaHQlM0QlMjI1MCUyMiUyMHJlbGF0aXZlJTNEJTIyMSUyMiUyMHdpZHRoJTNEJTIyNTAlMjIlMjBhcyUzRCUyMmdlb21ldHJ5JTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDQXJyYXklMjBhcyUzRCUyMnBvaW50cyUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyMzQwJTIyJTIweSUzRCUyMjUyMCUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyMzYwJTIyJTIweSUzRCUyMjQ4MCUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyMzgwJTIyJTIweSUzRCUyMjUyMCUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyNDAwJTIyJTIweSUzRCUyMjQ4MCUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRkFycmF5JTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhQb2ludCUyMHglM0QlMjIzMjAlMjIlMjB5JTNEJTIyNDgwJTIyJTIwYXMlM0QlMjJzb3VyY2VQb2ludCUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyNDQwJTIyJTIweSUzRCUyMjQ4MCUyMiUyMGFzJTNEJTIydGFyZ2V0UG9pbnQlMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZteEdlb21ldHJ5JTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGbXhDZWxsJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhDZWxsJTIwaWQlM0QlMjJ4TGs3V0ZSUXNLQ2xFd0RjdThtWC0yMSUyMiUyMHBhcmVudCUzRCUyMjElMjIlMjBzdHlsZSUzRCUyMmVsbGlwc2UlM0J3aGl0ZVNwYWNlJTNEd3JhcCUzQmh0bWwlM0QxJTNCYXNwZWN0JTNEZml4ZWQlM0IlMjIlMjB2YWx1ZSUzRCUyMkZhbWlseSUyMGFycml2YWwlMjB0byUyMHBhcmslMjBldmVudCUyMiUyMHZlcnRleCUzRCUyMjElMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteEdlb21ldHJ5JTIwaGVpZ2h0JTNEJTIyODAlMjIlMjB3aWR0aCUzRCUyMjgwJTIyJTIweCUzRCUyMjE3MCUyMiUyMHklM0QlMjI2NCUyMiUyMGFzJTNEJTIyZ2VvbWV0cnklMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZteENlbGwlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteENlbGwlMjBpZCUzRCUyMnhMazdXRlJRc0tDbEV3RGN1OG1YLTIyJTIyJTIwZWRnZSUzRCUyMjElMjIlMjBwYXJlbnQlM0QlMjIxJTIyJTIwc3R5bGUlM0QlMjJlbmRBcnJvdyUzRGNsYXNzaWMlM0JodG1sJTNEMSUzQnJvdW5kZWQlM0QwJTNCZW50cnlYJTNEMCUzQmVudHJ5WSUzRDAuNSUzQmVudHJ5RHglM0QwJTNCZW50cnlEeSUzRDAlM0IlMjIlMjB0YXJnZXQlM0QlMjJ4TGs3V0ZSUXNLQ2xFd0RjdThtWC0yMSUyMiUyMHZhbHVlJTNEJTIyJTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhHZW9tZXRyeSUyMGhlaWdodCUzRCUyMjUwJTIyJTIwcmVsYXRpdmUlM0QlMjIxJTIyJTIwd2lkdGglM0QlMjI1MCUyMiUyMGFzJTNEJTIyZ2VvbWV0cnklMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NBcnJheSUyMGFzJTNEJTIycG9pbnRzJTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhQb2ludCUyMHglM0QlMjI3MCUyMiUyMHklM0QlMjIxNDQlMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteFBvaW50JTIweCUzRCUyMjkwJTIyJTIweSUzRCUyMjEwNCUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyMTEwJTIyJTIweSUzRCUyMjE0NCUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyMTMwJTIyJTIweSUzRCUyMjEwNCUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRkFycmF5JTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhQb2ludCUyMHglM0QlMjI1MCUyMiUyMHklM0QlMjIxMDQlMjIlMjBhcyUzRCUyMnNvdXJjZVBvaW50JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhQb2ludCUyMHglM0QlMjIxNzAlMjIlMjB5JTNEJTIyMTA0JTIyJTIwYXMlM0QlMjJ0YXJnZXRQb2ludCUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14R2VvbWV0cnklM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZteENlbGwlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteENlbGwlMjBpZCUzRCUyMnhMazdXRlJRc0tDbEV3RGN1OG1YLTIzJTIyJTIwcGFyZW50JTNEJTIyMSUyMiUyMHN0eWxlJTNEJTIyZWxsaXBzZSUzQndoaXRlU3BhY2UlM0R3cmFwJTNCaHRtbCUzRDElM0Jhc3BlY3QlM0RmaXhlZCUzQiUyMiUyMHZhbHVlJTNEJTIyUXVldWUlMjBhYmFuZG9ubWVudCUyMGV2ZW50JTIyJTIwdmVydGV4JTNEJTIyMSUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214R2VvbWV0cnklMjBoZWlnaHQlM0QlMjI4MCUyMiUyMHdpZHRoJTNEJTIyODAlMjIlMjB4JTNEJTIyNDcwJTIyJTIweSUzRCUyMjY0JTIyJTIwYXMlM0QlMjJnZW9tZXRyeSUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14Q2VsbCUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214Q2VsbCUyMGlkJTNEJTIyeExrN1dGUlFzS0NsRXdEY3U4bVgtMjQlMjIlMjBlZGdlJTNEJTIyMSUyMiUyMHBhcmVudCUzRCUyMjElMjIlMjBzdHlsZSUzRCUyMmVuZEFycm93JTNEY2xhc3NpYyUzQmh0bWwlM0QxJTNCcm91bmRlZCUzRDAlM0JjdXJ2ZWQlM0QxJTNCJTIyJTIwdmFsdWUlM0QlMjIlMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteEdlb21ldHJ5JTIwaGVpZ2h0JTNEJTIyNTAlMjIlMjByZWxhdGl2ZSUzRCUyMjElMjIlMjB3aWR0aCUzRCUyMjUwJTIyJTIwYXMlM0QlMjJnZW9tZXRyeSUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ0FycmF5JTIwYXMlM0QlMjJwb2ludHMlMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteFBvaW50JTIweCUzRCUyMjUwNy41JTIyJTIweSUzRCUyMjQlMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZBcnJheSUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyNDg3LjUlMjIlMjB5JTNEJTIyNjQlMjIlMjBhcyUzRCUyMnNvdXJjZVBvaW50JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhQb2ludCUyMHglM0QlMjI1MzIuNSUyMiUyMHklM0QlMjI2NCUyMiUyMGFzJTNEJTIydGFyZ2V0UG9pbnQlMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZteEdlb21ldHJ5JTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGbXhDZWxsJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhDZWxsJTIwaWQlM0QlMjJ4TGs3V0ZSUXNLQ2xFd0RjdThtWC0yNSUyMiUyMHBhcmVudCUzRCUyMjElMjIlMjBzdHlsZSUzRCUyMmVsbGlwc2UlM0J3aGl0ZVNwYWNlJTNEd3JhcCUzQmh0bWwlM0QxJTNCYXNwZWN0JTNEZml4ZWQlM0IlMjIlMjB2YWx1ZSUzRCUyMkxlYXZpbmclMjBmcm9tJTIwcGFyayUyMGV2ZW50JTIyJTIwdmVydGV4JTNEJTIyMSUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214R2VvbWV0cnklMjBoZWlnaHQlM0QlMjI4MCUyMiUyMHdpZHRoJTNEJTIyODAlMjIlMjB4JTNEJTIyNzgwJTIyJTIweSUzRCUyMjIxNCUyMiUyMGFzJTNEJTIyZ2VvbWV0cnklMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZteENlbGwlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteENlbGwlMjBpZCUzRCUyMnhMazdXRlJRc0tDbEV3RGN1OG1YLTI2JTIyJTIwZWRnZSUzRCUyMjElMjIlMjBwYXJlbnQlM0QlMjIxJTIyJTIwc3R5bGUlM0QlMjJlbmRBcnJvdyUzRGNsYXNzaWMlM0JodG1sJTNEMSUzQnJvdW5kZWQlM0QwJTNCY3VydmVkJTNEMSUzQiUyMiUyMHZhbHVlJTNEJTIyJTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhHZW9tZXRyeSUyMGhlaWdodCUzRCUyMjUwJTIyJTIwcmVsYXRpdmUlM0QlMjIxJTIyJTIwd2lkdGglM0QlMjI1MCUyMiUyMGFzJTNEJTIyZ2VvbWV0cnklMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NBcnJheSUyMGFzJTNEJTIycG9pbnRzJTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhQb2ludCUyMHglM0QlMjI4MTcuNSUyMiUyMHklM0QlMjIxNTQlMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZBcnJheSUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyNzk3LjUlMjIlMjB5JTNEJTIyMjE0JTIyJTIwYXMlM0QlMjJzb3VyY2VQb2ludCUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyODQyLjUlMjIlMjB5JTNEJTIyMjE0JTIyJTIwYXMlM0QlMjJ0YXJnZXRQb2ludCUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14R2VvbWV0cnklM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZteENlbGwlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteENlbGwlMjBpZCUzRCUyMnhMazdXRlJRc0tDbEV3RGN1OG1YLTI3JTIyJTIwcGFyZW50JTNEJTIyMSUyMiUyMHN0eWxlJTNEJTIyZWxsaXBzZSUzQndoaXRlU3BhY2UlM0R3cmFwJTNCaHRtbCUzRDElM0Jhc3BlY3QlM0RmaXhlZCUzQiUyMiUyMHZhbHVlJTNEJTIyRW5kJTIwYXR0cmFjdGlvbiUyMGV2ZW50JTIyJTIwdmVydGV4JTNEJTIyMSUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214R2VvbWV0cnklMjBoZWlnaHQlM0QlMjI4MCUyMiUyMHdpZHRoJTNEJTIyODAlMjIlMjB4JTNEJTIyNDkwJTIyJTIweSUzRCUyMjIxNCUyMiUyMGFzJTNEJTIyZ2VvbWV0cnklMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZteENlbGwlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteENlbGwlMjBpZCUzRCUyMnhMazdXRlJRc0tDbEV3RGN1OG1YLTI4JTIyJTIwZWRnZSUzRCUyMjElMjIlMjBwYXJlbnQlM0QlMjIxJTIyJTIwc3R5bGUlM0QlMjJlbmRBcnJvdyUzRGNsYXNzaWMlM0JodG1sJTNEMSUzQnJvdW5kZWQlM0QwJTNCY3VydmVkJTNEMSUzQiUyMiUyMHZhbHVlJTNEJTIyJTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhHZW9tZXRyeSUyMGhlaWdodCUzRCUyMjUwJTIyJTIwcmVsYXRpdmUlM0QlMjIxJTIyJTIwd2lkdGglM0QlMjI1MCUyMiUyMGFzJTNEJTIyZ2VvbWV0cnklMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NBcnJheSUyMGFzJTNEJTIycG9pbnRzJTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhQb2ludCUyMHglM0QlMjI1MjcuNSUyMiUyMHklM0QlMjIxNTQlMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZBcnJheSUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyNTA3LjUlMjIlMjB5JTNEJTIyMjE0JTIyJTIwYXMlM0QlMjJzb3VyY2VQb2ludCUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyNTUyLjUlMjIlMjB5JTNEJTIyMjE0JTIyJTIwYXMlM0QlMjJ0YXJnZXRQb2ludCUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14R2VvbWV0cnklM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZteENlbGwlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteENlbGwlMjBpZCUzRCUyMkJtd0ZkR180TXAybmxPcWFTVDZwLTYlMjIlMjBlZGdlJTNEJTIyMSUyMiUyMHBhcmVudCUzRCUyMjElMjIlMjBzb3VyY2UlM0QlMjJ4TGs3V0ZSUXNLQ2xFd0RjdThtWC0yNyUyMiUyMHN0eWxlJTNEJTIyZW5kQXJyb3clM0RjbGFzc2ljJTNCaHRtbCUzRDElM0Jyb3VuZGVkJTNEMCUzQmV4aXRYJTNEMCUzQmV4aXRZJTNEMCUzQmV4aXREeCUzRDAlM0JleGl0RHklM0QwJTNCZW50cnlYJTNEMC41JTNCZW50cnlZJTNEMSUzQmVudHJ5RHglM0QwJTNCZW50cnlEeSUzRDAlM0JjdXJ2ZWQlM0QxJTNCJTIyJTIwdGFyZ2V0JTNEJTIyeExrN1dGUlFzS0NsRXdEY3U4bVgtMjMlMjIlMjB2YWx1ZSUzRCUyMiUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214R2VvbWV0cnklMjBoZWlnaHQlM0QlMjI1MCUyMiUyMHJlbGF0aXZlJTNEJTIyMSUyMiUyMHdpZHRoJTNEJTIyNTAlMjIlMjBhcyUzRCUyMmdlb21ldHJ5JTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDQXJyYXklMjBhcyUzRCUyMnBvaW50cyUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyNTAwJTIyJTIweSUzRCUyMjE3NCUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRkFycmF5JTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhQb2ludCUyMHglM0QlMjI2MjAlMjIlMjB5JTNEJTIyNDE0JTIyJTIwYXMlM0QlMjJzb3VyY2VQb2ludCUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyNjcwJTIyJTIweSUzRCUyMjM2NCUyMiUyMGFzJTNEJTIydGFyZ2V0UG9pbnQlMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZteEdlb21ldHJ5JTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGbXhDZWxsJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhDZWxsJTIwaWQlM0QlMjJCbXdGZEdfNE1wMm5sT3FhU1Q2cC03JTIyJTIwZWRnZSUzRCUyMjElMjIlMjBwYXJlbnQlM0QlMjIxJTIyJTIwc291cmNlJTNEJTIyeExrN1dGUlFzS0NsRXdEY3U4bVgtMjclMjIlMjBzdHlsZSUzRCUyMmVuZEFycm93JTNEY2xhc3NpYyUzQmh0bWwlM0QxJTNCcm91bmRlZCUzRDAlM0JleGl0WCUzRDElM0JleGl0WSUzRDAuNSUzQmV4aXREeCUzRDAlM0JleGl0RHklM0QwJTNCZW50cnlYJTNEMCUzQmVudHJ5WSUzRDAuNSUzQmVudHJ5RHglM0QwJTNCZW50cnlEeSUzRDAlM0JjdXJ2ZWQlM0QxJTNCJTIyJTIwdGFyZ2V0JTNEJTIyeExrN1dGUlFzS0NsRXdEY3U4bVgtMjUlMjIlMjB2YWx1ZSUzRCUyMiUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214R2VvbWV0cnklMjBoZWlnaHQlM0QlMjI1MCUyMiUyMHJlbGF0aXZlJTNEJTIyMSUyMiUyMHdpZHRoJTNEJTIyNTAlMjIlMjBhcyUzRCUyMmdlb21ldHJ5JTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhQb2ludCUyMHglM0QlMjI2MzAlMjIlMjB5JTNEJTIyNDA0JTIyJTIwYXMlM0QlMjJzb3VyY2VQb2ludCUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyNjgwJTIyJTIweSUzRCUyMjM1NCUyMiUyMGFzJTNEJTIydGFyZ2V0UG9pbnQlMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZteEdlb21ldHJ5JTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGbXhDZWxsJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhDZWxsJTIwaWQlM0QlMjJCbXdGZEdfNE1wMm5sT3FhU1Q2cC04JTIyJTIwZWRnZSUzRCUyMjElMjIlMjBwYXJlbnQlM0QlMjIxJTIyJTIwc291cmNlJTNEJTIyeExrN1dGUlFzS0NsRXdEY3U4bVgtMjclMjIlMjBzdHlsZSUzRCUyMmVuZEFycm93JTNEY2xhc3NpYyUzQmh0bWwlM0QxJTNCcm91bmRlZCUzRDAlM0JleGl0WCUzRDAuOTUyJTNCZXhpdFklM0QwLjI3JTNCZXhpdER4JTNEMCUzQmV4aXREeSUzRDAlM0JlbnRyeVglM0QwJTNCZW50cnlZJTNEMSUzQmVudHJ5RHglM0QwJTNCZW50cnlEeSUzRDAlM0JjdXJ2ZWQlM0QxJTNCZXhpdFBlcmltZXRlciUzRDAlM0IlMjIlMjB0YXJnZXQlM0QlMjJ4TGs3V0ZSUXNLQ2xFd0RjdThtWC0xNSUyMiUyMHZhbHVlJTNEJTIyJTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhHZW9tZXRyeSUyMGhlaWdodCUzRCUyMjUwJTIyJTIwcmVsYXRpdmUlM0QlMjIxJTIyJTIwd2lkdGglM0QlMjI1MCUyMiUyMGFzJTNEJTIyZ2VvbWV0cnklMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteFBvaW50JTIweCUzRCUyMjcwMCUyMiUyMHklM0QlMjIzMTQlMjIlMjBhcyUzRCUyMnNvdXJjZVBvaW50JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhQb2ludCUyMHglM0QlMjI3NTAlMjIlMjB5JTNEJTIyMjY0JTIyJTIwYXMlM0QlMjJ0YXJnZXRQb2ludCUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14R2VvbWV0cnklM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZteENlbGwlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteENlbGwlMjBpZCUzRCUyMkJtd0ZkR180TXAybmxPcWFTVDZwLTEyJTIyJTIwZWRnZSUzRCUyMjElMjIlMjBwYXJlbnQlM0QlMjIxJTIyJTIwc291cmNlJTNEJTIyeExrN1dGUlFzS0NsRXdEY3U4bVgtMTUlMjIlMjBzdHlsZSUzRCUyMmVuZEFycm93JTNEY2xhc3NpYyUzQmh0bWwlM0QxJTNCcm91bmRlZCUzRDAlM0JleGl0WCUzRDAuMDM1JTNCZXhpdFklM0QwLjY5OCUzQmV4aXREeCUzRDAlM0JleGl0RHklM0QwJTNCZW50cnlYJTNEMSUzQmVudHJ5WSUzRDAlM0JlbnRyeUR4JTNEMCUzQmVudHJ5RHklM0QwJTNCY3VydmVkJTNEMSUzQmV4aXRQZXJpbWV0ZXIlM0QwJTNCJTIyJTIwdGFyZ2V0JTNEJTIyeExrN1dGUlFzS0NsRXdEY3U4bVgtMjclMjIlMjB2YWx1ZSUzRCUyMiUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214R2VvbWV0cnklMjBoZWlnaHQlM0QlMjI1MCUyMiUyMHJlbGF0aXZlJTNEJTIyMSUyMiUyMHdpZHRoJTNEJTIyNTAlMjIlMjBhcyUzRCUyMmdlb21ldHJ5JTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhQb2ludCUyMHglM0QlMjI2MTAlMjIlMjB5JTNEJTIyMTU0JTIyJTIwYXMlM0QlMjJzb3VyY2VQb2ludCUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyNjYwJTIyJTIweSUzRCUyMjEwNCUyMiUyMGFzJTNEJTIydGFyZ2V0UG9pbnQlMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZteEdlb21ldHJ5JTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGbXhDZWxsJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhDZWxsJTIwaWQlM0QlMjJCbXdGZEdfNE1wMm5sT3FhU1Q2cC0xMyUyMiUyMGVkZ2UlM0QlMjIxJTIyJTIwcGFyZW50JTNEJTIyMSUyMiUyMHN0eWxlJTNEJTIyZW5kQXJyb3clM0RjbGFzc2ljJTNCaHRtbCUzRDElM0Jyb3VuZGVkJTNEMCUzQmN1cnZlZCUzRDElM0IlMjIlMjB2YWx1ZSUzRCUyMiUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214R2VvbWV0cnklMjBoZWlnaHQlM0QlMjI1MCUyMiUyMHJlbGF0aXZlJTNEJTIyMSUyMiUyMHdpZHRoJTNEJTIyNTAlMjIlMjBhcyUzRCUyMmdlb21ldHJ5JTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDQXJyYXklMjBhcyUzRCUyMnBvaW50cyUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyMjA3LjUlMjIlMjB5JTNEJTIyNCUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRkFycmF5JTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhQb2ludCUyMHglM0QlMjIxODcuNSUyMiUyMHklM0QlMjI2NCUyMiUyMGFzJTNEJTIyc291cmNlUG9pbnQlMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteFBvaW50JTIweCUzRCUyMjIzMi41JTIyJTIweSUzRCUyMjY0JTIyJTIwYXMlM0QlMjJ0YXJnZXRQb2ludCUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14R2VvbWV0cnklM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZteENlbGwlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteENlbGwlMjBpZCUzRCUyMkJtd0ZkR180TXAybmxPcWFTVDZwLTE1JTIyJTIwZWRnZSUzRCUyMjElMjIlMjBwYXJlbnQlM0QlMjIxJTIyJTIwc291cmNlJTNEJTIyeExrN1dGUlFzS0NsRXdEY3U4bVgtMjMlMjIlMjBzdHlsZSUzRCUyMmVuZEFycm93JTNEY2xhc3NpYyUzQmh0bWwlM0QxJTNCcm91bmRlZCUzRDAlM0JleGl0WCUzRDAlM0JleGl0WSUzRDElM0JleGl0RHglM0QwJTNCZXhpdER5JTNEMCUzQmVudHJ5WCUzRDAuMDY2JTNCZW50cnlZJTNEMC4yMzIlM0JlbnRyeUR4JTNEMCUzQmVudHJ5RHklM0QwJTNCZW50cnlQZXJpbWV0ZXIlM0QwJTNCY3VydmVkJTNEMSUzQiUyMiUyMHRhcmdldCUzRCUyMnhMazdXRlJRc0tDbEV3RGN1OG1YLTI3JTIyJTIwdmFsdWUlM0QlMjIlMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteEdlb21ldHJ5JTIwaGVpZ2h0JTNEJTIyNTAlMjIlMjByZWxhdGl2ZSUzRCUyMjElMjIlMjB3aWR0aCUzRCUyMjUwJTIyJTIwYXMlM0QlMjJnZW9tZXRyeSUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ0FycmF5JTIwYXMlM0QlMjJwb2ludHMlMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteFBvaW50JTIweCUzRCUyMjQ0MCUyMiUyMHklM0QlMjIxOTQlMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZBcnJheSUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyNzYwJTIyJTIweSUzRCUyMjIxNCUyMiUyMGFzJTNEJTIyc291cmNlUG9pbnQlMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteFBvaW50JTIweCUzRCUyMjgxMCUyMiUyMHklM0QlMjIxNjQlMjIlMjBhcyUzRCUyMnRhcmdldFBvaW50JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGbXhHZW9tZXRyeSUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14Q2VsbCUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214Q2VsbCUyMGlkJTNEJTIyOTlkcHlSRFVDRU40TS1ueHp1eGQtMSUyMiUyMGVkZ2UlM0QlMjIxJTIyJTIwcGFyZW50JTNEJTIyMSUyMiUyMHNvdXJjZSUzRCUyMnhMazdXRlJRc0tDbEV3RGN1OG1YLTE1JTIyJTIwc3R5bGUlM0QlMjJlbmRBcnJvdyUzRGNsYXNzaWMlM0JodG1sJTNEMSUzQnJvdW5kZWQlM0QwJTNCZXhpdFglM0QwLjc3NSUzQmV4aXRZJTNEMC45MTclM0JleGl0RHglM0QwJTNCZXhpdER5JTNEMCUzQmVudHJ5WCUzRDAuMDIzJTNCZW50cnlZJTNEMC4zMzMlM0JlbnRyeUR4JTNEMCUzQmVudHJ5RHklM0QwJTNCZW50cnlQZXJpbWV0ZXIlM0QwJTNCZXhpdFBlcmltZXRlciUzRDAlM0IlMjIlMjB0YXJnZXQlM0QlMjJ4TGs3V0ZSUXNLQ2xFd0RjdThtWC0yNSUyMiUyMHZhbHVlJTNEJTIyJTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhHZW9tZXRyeSUyMGhlaWdodCUzRCUyMjUwJTIyJTIwcmVsYXRpdmUlM0QlMjIxJTIyJTIwd2lkdGglM0QlMjI1MCUyMiUyMGFzJTNEJTIyZ2VvbWV0cnklMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteFBvaW50JTIweCUzRCUyMjg0MCUyMiUyMHklM0QlMjIxNDQlMjIlMjBhcyUzRCUyMnNvdXJjZVBvaW50JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhQb2ludCUyMHglM0QlMjI4OTAlMjIlMjB5JTNEJTIyOTQlMjIlMjBhcyUzRCUyMnRhcmdldFBvaW50JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGbXhHZW9tZXRyeSUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14Q2VsbCUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214Q2VsbCUyMGlkJTNEJTIyOTlkcHlSRFVDRU40TS1ueHp1eGQtMiUyMiUyMHBhcmVudCUzRCUyMjElMjIlMjBzdHlsZSUzRCUyMmVsbGlwc2UlM0J3aGl0ZVNwYWNlJTNEd3JhcCUzQmh0bWwlM0QxJTNCYXNwZWN0JTNEZml4ZWQlM0IlMjIlMjB2YWx1ZSUzRCUyMkVuZCUyMGdldHRpbmclMjB0aWNrZXQlMjBldmVudCUyNmFtcCUzQm5ic3AlM0IlMjIlMjB2ZXJ0ZXglM0QlMjIxJTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhHZW9tZXRyeSUyMGhlaWdodCUzRCUyMjgwJTIyJTIwd2lkdGglM0QlMjI4MCUyMiUyMHglM0QlMjIzMjAlMjIlMjB5JTNEJTIyNjQlMjIlMjBhcyUzRCUyMmdlb21ldHJ5JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGbXhDZWxsJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhDZWxsJTIwaWQlM0QlMjI5OWRweVJEVUNFTjRNLW54enV4ZC00JTIyJTIwZWRnZSUzRCUyMjElMjIlMjBwYXJlbnQlM0QlMjIxJTIyJTIwc3R5bGUlM0QlMjJlbmRBcnJvdyUzRGNsYXNzaWMlM0JodG1sJTNEMSUzQnJvdW5kZWQlM0QwJTNCY3VydmVkJTNEMSUzQiUyMiUyMHZhbHVlJTNEJTIyJTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhHZW9tZXRyeSUyMGhlaWdodCUzRCUyMjUwJTIyJTIwcmVsYXRpdmUlM0QlMjIxJTIyJTIwd2lkdGglM0QlMjI1MCUyMiUyMGFzJTNEJTIyZ2VvbWV0cnklMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NBcnJheSUyMGFzJTNEJTIycG9pbnRzJTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhQb2ludCUyMHglM0QlMjIzNTcuNSUyMiUyMHklM0QlMjI0JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGQXJyYXklM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteFBvaW50JTIweCUzRCUyMjMzNy41JTIyJTIweSUzRCUyMjY0JTIyJTIwYXMlM0QlMjJzb3VyY2VQb2ludCUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyMzgyLjUlMjIlMjB5JTNEJTIyNjQlMjIlMjBhcyUzRCUyMnRhcmdldFBvaW50JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGbXhHZW9tZXRyeSUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14Q2VsbCUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214Q2VsbCUyMGlkJTNEJTIyOTlkcHlSRFVDRU40TS1ueHp1eGQtNSUyMiUyMGVkZ2UlM0QlMjIxJTIyJTIwcGFyZW50JTNEJTIyMSUyMiUyMHNvdXJjZSUzRCUyMnhMazdXRlJRc0tDbEV3RGN1OG1YLTIxJTIyJTIwc3R5bGUlM0QlMjJlbmRBcnJvdyUzRGNsYXNzaWMlM0JodG1sJTNEMSUzQnJvdW5kZWQlM0QwJTNCZXhpdFglM0QxJTNCZXhpdFklM0QwLjUlM0JleGl0RHglM0QwJTNCZXhpdER5JTNEMCUzQmVudHJ5WCUzRDAlM0JlbnRyeVklM0QwLjUlM0JlbnRyeUR4JTNEMCUzQmVudHJ5RHklM0QwJTNCJTIyJTIwdGFyZ2V0JTNEJTIyOTlkcHlSRFVDRU40TS1ueHp1eGQtMiUyMiUyMHZhbHVlJTNEJTIyJTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhHZW9tZXRyeSUyMGhlaWdodCUzRCUyMjUwJTIyJTIwcmVsYXRpdmUlM0QlMjIxJTIyJTIwd2lkdGglM0QlMjI1MCUyMiUyMGFzJTNEJTIyZ2VvbWV0cnklMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteFBvaW50JTIweCUzRCUyMjIwMCUyMiUyMHklM0QlMjIyMzQlMjIlMjBhcyUzRCUyMnNvdXJjZVBvaW50JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhQb2ludCUyMHglM0QlMjIyNTAlMjIlMjB5JTNEJTIyMTg0JTIyJTIwYXMlM0QlMjJ0YXJnZXRQb2ludCUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14R2VvbWV0cnklM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZteENlbGwlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteENlbGwlMjBpZCUzRCUyMjk5ZHB5UkRVQ0VONE0tbnh6dXhkLTclMjIlMjBlZGdlJTNEJTIyMSUyMiUyMHBhcmVudCUzRCUyMjElMjIlMjBzb3VyY2UlM0QlMjI5OWRweVJEVUNFTjRNLW54enV4ZC0yJTIyJTIwc3R5bGUlM0QlMjJlbmRBcnJvdyUzRGNsYXNzaWMlM0JodG1sJTNEMSUzQnJvdW5kZWQlM0QwJTNCZXhpdFglM0QxJTNCZXhpdFklM0QwLjUlM0JleGl0RHglM0QwJTNCZXhpdER5JTNEMCUzQmVudHJ5WCUzRDAlM0JlbnRyeVklM0QwLjUlM0JlbnRyeUR4JTNEMCUzQmVudHJ5RHklM0QwJTNCJTIyJTIwdGFyZ2V0JTNEJTIyeExrN1dGUlFzS0NsRXdEY3U4bVgtMjMlMjIlMjB2YWx1ZSUzRCUyMiUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214R2VvbWV0cnklMjBoZWlnaHQlM0QlMjI1MCUyMiUyMHJlbGF0aXZlJTNEJTIyMSUyMiUyMHdpZHRoJTNEJTIyNTAlMjIlMjBhcyUzRCUyMmdlb21ldHJ5JTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhQb2ludCUyMHglM0QlMjIzNDAlMjIlMjB5JTNEJTIyMjU0JTIyJTIwYXMlM0QlMjJzb3VyY2VQb2ludCUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyMzkwJTIyJTIweSUzRCUyMjIwNCUyMiUyMGFzJTNEJTIydGFyZ2V0UG9pbnQlMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZteEdlb21ldHJ5JTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGbXhDZWxsJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhDZWxsJTIwaWQlM0QlMjI5OWRweVJEVUNFTjRNLW54enV4ZC04JTIyJTIwZWRnZSUzRCUyMjElMjIlMjBwYXJlbnQlM0QlMjIxJTIyJTIwc291cmNlJTNEJTIyOTlkcHlSRFVDRU40TS1ueHp1eGQtMiUyMiUyMHN0eWxlJTNEJTIyZW5kQXJyb3clM0RjbGFzc2ljJTNCaHRtbCUzRDElM0Jyb3VuZGVkJTNEMCUzQmV4aXRYJTNEMC41JTNCZXhpdFklM0QxJTNCZXhpdER4JTNEMCUzQmV4aXREeSUzRDAlM0JlbnRyeVglM0QwJTNCZW50cnlZJTNEMC41JTNCZW50cnlEeCUzRDAlM0JlbnRyeUR5JTNEMCUzQmN1cnZlZCUzRDElM0IlMjIlMjB0YXJnZXQlM0QlMjJ4TGs3V0ZSUXNLQ2xFd0RjdThtWC0yNyUyMiUyMHZhbHVlJTNEJTIyJTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhHZW9tZXRyeSUyMGhlaWdodCUzRCUyMjUwJTIyJTIwcmVsYXRpdmUlM0QlMjIxJTIyJTIwd2lkdGglM0QlMjI1MCUyMiUyMGFzJTNEJTIyZ2VvbWV0cnklMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NBcnJheSUyMGFzJTNEJTIycG9pbnRzJTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhQb2ludCUyMHglM0QlMjIzNjAlMjIlMjB5JTNEJTIyMjI0JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGQXJyYXklM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteFBvaW50JTIweCUzRCUyMjM2MCUyMiUyMHklM0QlMjIyMjQlMjIlMjBhcyUzRCUyMnNvdXJjZVBvaW50JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhQb2ludCUyMHglM0QlMjI0MTAlMjIlMjB5JTNEJTIyMTc0JTIyJTIwYXMlM0QlMjJ0YXJnZXRQb2ludCUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14R2VvbWV0cnklM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZteENlbGwlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteENlbGwlMjBpZCUzRCUyMjk5ZHB5UkRVQ0VONE0tbnh6dXhkLTklMjIlMjBlZGdlJTNEJTIyMSUyMiUyMHBhcmVudCUzRCUyMjElMjIlMjBzb3VyY2UlM0QlMjJ4TGs3V0ZSUXNLQ2xFd0RjdThtWC0yMSUyMiUyMHN0eWxlJTNEJTIyZW5kQXJyb3clM0RjbGFzc2ljJTNCaHRtbCUzRDElM0Jyb3VuZGVkJTNEMCUzQmV4aXRYJTNEMSUzQmV4aXRZJTNEMCUzQmV4aXREeCUzRDAlM0JleGl0RHklM0QwJTNCZW50cnlYJTNEMCUzQmVudHJ5WSUzRDAlM0JlbnRyeUR4JTNEMCUzQmVudHJ5RHklM0QwJTNCY3VydmVkJTNEMSUzQiUyMiUyMHRhcmdldCUzRCUyMnhMazdXRlJRc0tDbEV3RGN1OG1YLTI1JTIyJTIwdmFsdWUlM0QlMjIlMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteEdlb21ldHJ5JTIwaGVpZ2h0JTNEJTIyNTAlMjIlMjByZWxhdGl2ZSUzRCUyMjElMjIlMjB3aWR0aCUzRCUyMjUwJTIyJTIwYXMlM0QlMjJnZW9tZXRyeSUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ0FycmF5JTIwYXMlM0QlMjJwb2ludHMlMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteFBvaW50JTIweCUzRCUyMjg2MCUyMiUyMHklM0QlMjItMjAwJTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGQXJyYXklM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteFBvaW50JTIweCUzRCUyMjMwMCUyMiUyMHklM0QlMjIyODQlMjIlMjBhcyUzRCUyMnNvdXJjZVBvaW50JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhQb2ludCUyMHglM0QlMjIzNTAlMjIlMjB5JTNEJTIyMjM0JTIyJTIwYXMlM0QlMjJ0YXJnZXRQb2ludCUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14R2VvbWV0cnklM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZteENlbGwlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteENlbGwlMjBpZCUzRCUyMmcySFF3c3ZUZ19uTlNRTk5yNDlBLTElMjIlMjBwYXJlbnQlM0QlMjIxJTIyJTIwc3R5bGUlM0QlMjJlbGxpcHNlJTNCd2hpdGVTcGFjZSUzRHdyYXAlM0JodG1sJTNEMSUzQmFzcGVjdCUzRGZpeGVkJTNCJTIyJTIwdmFsdWUlM0QlMjJTaW5nbGUlMjB2aXNpdG9yJTIwYXJyaXZhbCUyMHRvJTIwcGFyayUyMGV2ZW50JTIyJTIwdmVydGV4JTNEJTIyMSUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214R2VvbWV0cnklMjBoZWlnaHQlM0QlMjI4MCUyMiUyMHdpZHRoJTNEJTIyODAlMjIlMjB4JTNEJTIyMTkwJTIyJTIweSUzRCUyMjIwNCUyMiUyMGFzJTNEJTIyZ2VvbWV0cnklMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZteENlbGwlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteENlbGwlMjBpZCUzRCUyMmcySFF3c3ZUZ19uTlNRTk5yNDlBLTIlMjIlMjBlZGdlJTNEJTIyMSUyMiUyMHBhcmVudCUzRCUyMjElMjIlMjBzdHlsZSUzRCUyMmVuZEFycm93JTNEY2xhc3NpYyUzQmh0bWwlM0QxJTNCcm91bmRlZCUzRDAlM0JlbnRyeVglM0QwJTNCZW50cnlZJTNEMC41JTNCZW50cnlEeCUzRDAlM0JlbnRyeUR5JTNEMCUzQiUyMiUyMHRhcmdldCUzRCUyMmcySFF3c3ZUZ19uTlNRTk5yNDlBLTElMjIlMjB2YWx1ZSUzRCUyMiUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214R2VvbWV0cnklMjBoZWlnaHQlM0QlMjI1MCUyMiUyMHJlbGF0aXZlJTNEJTIyMSUyMiUyMHdpZHRoJTNEJTIyNTAlMjIlMjBhcyUzRCUyMmdlb21ldHJ5JTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDQXJyYXklMjBhcyUzRCUyMnBvaW50cyUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyOTAlMjIlMjB5JTNEJTIyMjg0JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhQb2ludCUyMHglM0QlMjIxMTAlMjIlMjB5JTNEJTIyMjQ0JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhQb2ludCUyMHglM0QlMjIxMzAlMjIlMjB5JTNEJTIyMjg0JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhQb2ludCUyMHglM0QlMjIxNTAlMjIlMjB5JTNEJTIyMjQ0JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGQXJyYXklM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteFBvaW50JTIweCUzRCUyMjcwJTIyJTIweSUzRCUyMjI0NCUyMiUyMGFzJTNEJTIyc291cmNlUG9pbnQlMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteFBvaW50JTIweCUzRCUyMjE5MCUyMiUyMHklM0QlMjIyNDQlMjIlMjBhcyUzRCUyMnRhcmdldFBvaW50JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGbXhHZW9tZXRyeSUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14Q2VsbCUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214Q2VsbCUyMGlkJTNEJTIyZzJIUXdzdlRnX25OU1FOTnI0OUEtMyUyMiUyMGVkZ2UlM0QlMjIxJTIyJTIwcGFyZW50JTNEJTIyMSUyMiUyMHN0eWxlJTNEJTIyZW5kQXJyb3clM0RjbGFzc2ljJTNCaHRtbCUzRDElM0Jyb3VuZGVkJTNEMCUzQmN1cnZlZCUzRDElM0IlMjIlMjB2YWx1ZSUzRCUyMiUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214R2VvbWV0cnklMjBoZWlnaHQlM0QlMjI1MCUyMiUyMHJlbGF0aXZlJTNEJTIyMSUyMiUyMHdpZHRoJTNEJTIyNTAlMjIlMjBhcyUzRCUyMmdlb21ldHJ5JTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDQXJyYXklMjBhcyUzRCUyMnBvaW50cyUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyMjI3LjUlMjIlMjB5JTNEJTIyMTQ0JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGQXJyYXklM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteFBvaW50JTIweCUzRCUyMjIwNy41JTIyJTIweSUzRCUyMjIwNCUyMiUyMGFzJTNEJTIyc291cmNlUG9pbnQlMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteFBvaW50JTIweCUzRCUyMjI1Mi41JTIyJTIweSUzRCUyMjIwNCUyMiUyMGFzJTNEJTIydGFyZ2V0UG9pbnQlMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZteEdlb21ldHJ5JTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGbXhDZWxsJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhDZWxsJTIwaWQlM0QlMjJnMkhRd3N2VGdfbk5TUU5OcjQ5QS02JTIyJTIwcGFyZW50JTNEJTIyMSUyMiUyMHN0eWxlJTNEJTIyZWxsaXBzZSUzQndoaXRlU3BhY2UlM0R3cmFwJTNCaHRtbCUzRDElM0Jhc3BlY3QlM0RmaXhlZCUzQiUyMiUyMHZhbHVlJTNEJTIyVGVlbmFnZXJzJTIwYXJyaXZhbCUyMHRvJTIwcGFyayUyMGV2ZW50JTIyJTIwdmVydGV4JTNEJTIyMSUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214R2VvbWV0cnklMjBoZWlnaHQlM0QlMjI4MCUyMiUyMHdpZHRoJTNEJTIyODAlMjIlMjB4JTNEJTIyMzEwJTIyJTIweSUzRCUyMjI5MSUyMiUyMGFzJTNEJTIyZ2VvbWV0cnklMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZteENlbGwlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteENlbGwlMjBpZCUzRCUyMmcySFF3c3ZUZ19uTlNRTk5yNDlBLTclMjIlMjBlZGdlJTNEJTIyMSUyMiUyMHBhcmVudCUzRCUyMjElMjIlMjBzdHlsZSUzRCUyMmVuZEFycm93JTNEY2xhc3NpYyUzQmh0bWwlM0QxJTNCcm91bmRlZCUzRDAlM0JlbnRyeVglM0QwJTNCZW50cnlZJTNEMC41JTNCZW50cnlEeCUzRDAlM0JlbnRyeUR5JTNEMCUzQiUyMiUyMHRhcmdldCUzRCUyMmcySFF3c3ZUZ19uTlNRTk5yNDlBLTYlMjIlMjB2YWx1ZSUzRCUyMiUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214R2VvbWV0cnklMjBoZWlnaHQlM0QlMjI1MCUyMiUyMHJlbGF0aXZlJTNEJTIyMSUyMiUyMHdpZHRoJTNEJTIyNTAlMjIlMjBhcyUzRCUyMmdlb21ldHJ5JTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDQXJyYXklMjBhcyUzRCUyMnBvaW50cyUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyMjEwJTIyJTIweSUzRCUyMjM3MSUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyMjMwJTIyJTIweSUzRCUyMjMzMSUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyMjUwJTIyJTIweSUzRCUyMjM3MSUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyMjcwJTIyJTIweSUzRCUyMjMzMSUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRkFycmF5JTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhQb2ludCUyMHglM0QlMjIxOTAlMjIlMjB5JTNEJTIyMzMxJTIyJTIwYXMlM0QlMjJzb3VyY2VQb2ludCUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyMzEwJTIyJTIweSUzRCUyMjMzMSUyMiUyMGFzJTNEJTIydGFyZ2V0UG9pbnQlMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZteEdlb21ldHJ5JTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGbXhDZWxsJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhDZWxsJTIwaWQlM0QlMjJnMkhRd3N2VGdfbk5TUU5OcjQ5QS04JTIyJTIwZWRnZSUzRCUyMjElMjIlMjBwYXJlbnQlM0QlMjIxJTIyJTIwc3R5bGUlM0QlMjJlbmRBcnJvdyUzRGNsYXNzaWMlM0JodG1sJTNEMSUzQnJvdW5kZWQlM0QwJTNCY3VydmVkJTNEMSUzQiUyMiUyMHZhbHVlJTNEJTIyJTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhHZW9tZXRyeSUyMGhlaWdodCUzRCUyMjUwJTIyJTIwcmVsYXRpdmUlM0QlMjIxJTIyJTIwd2lkdGglM0QlMjI1MCUyMiUyMGFzJTNEJTIyZ2VvbWV0cnklMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NBcnJheSUyMGFzJTNEJTIycG9pbnRzJTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhQb2ludCUyMHglM0QlMjIzNDcuNSUyMiUyMHklM0QlMjIyMzAlMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZBcnJheSUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyMzI3LjUlMjIlMjB5JTNEJTIyMjkwJTIyJTIwYXMlM0QlMjJzb3VyY2VQb2ludCUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyMzcyLjUlMjIlMjB5JTNEJTIyMjkwJTIyJTIwYXMlM0QlMjJ0YXJnZXRQb2ludCUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14R2VvbWV0cnklM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZteENlbGwlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteENlbGwlMjBpZCUzRCUyMmcySFF3c3ZUZ19uTlNRTk5yNDlBLTklMjIlMjBlZGdlJTNEJTIyMSUyMiUyMHBhcmVudCUzRCUyMjElMjIlMjBzb3VyY2UlM0QlMjJnMkhRd3N2VGdfbk5TUU5OcjQ5QS0xJTIyJTIwc3R5bGUlM0QlMjJlbmRBcnJvdyUzRGNsYXNzaWMlM0JodG1sJTNEMSUzQnJvdW5kZWQlM0QwJTNCZXhpdFglM0QxJTNCZXhpdFklM0QwJTNCZXhpdER4JTNEMCUzQmV4aXREeSUzRDAlM0JlbnRyeVglM0QwJTNCZW50cnlZJTNEMSUzQmVudHJ5RHglM0QwJTNCZW50cnlEeSUzRDAlM0IlMjIlMjB0YXJnZXQlM0QlMjI5OWRweVJEVUNFTjRNLW54enV4ZC0yJTIyJTIwdmFsdWUlM0QlMjIlMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteEdlb21ldHJ5JTIwaGVpZ2h0JTNEJTIyNTAlMjIlMjByZWxhdGl2ZSUzRCUyMjElMjIlMjB3aWR0aCUzRCUyMjUwJTIyJTIwYXMlM0QlMjJnZW9tZXRyeSUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyMzYwJTIyJTIweSUzRCUyMjI3MCUyMiUyMGFzJTNEJTIyc291cmNlUG9pbnQlMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteFBvaW50JTIweCUzRCUyMjQxMCUyMiUyMHklM0QlMjIyMjAlMjIlMjBhcyUzRCUyMnRhcmdldFBvaW50JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGbXhHZW9tZXRyeSUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14Q2VsbCUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214Q2VsbCUyMGlkJTNEJTIyZzJIUXdzdlRnX25OU1FOTnI0OUEtMTAlMjIlMjBlZGdlJTNEJTIyMSUyMiUyMHBhcmVudCUzRCUyMjElMjIlMjBzb3VyY2UlM0QlMjJnMkhRd3N2VGdfbk5TUU5OcjQ5QS02JTIyJTIwc3R5bGUlM0QlMjJlbmRBcnJvdyUzRGNsYXNzaWMlM0JodG1sJTNEMSUzQnJvdW5kZWQlM0QwJTNCZXhpdFglM0QwJTNCZXhpdFklM0QwJTNCZXhpdER4JTNEMCUzQmV4aXREeSUzRDAlM0JlbnRyeVglM0QwLjM0MiUzQmVudHJ5WSUzRDAuOTc1JTNCZW50cnlEeCUzRDAlM0JlbnRyeUR5JTNEMCUzQmVudHJ5UGVyaW1ldGVyJTNEMCUzQmN1cnZlZCUzRDElM0IlMjIlMjB0YXJnZXQlM0QlMjI5OWRweVJEVUNFTjRNLW54enV4ZC0yJTIyJTIwdmFsdWUlM0QlMjIlMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteEdlb21ldHJ5JTIwaGVpZ2h0JTNEJTIyNTAlMjIlMjByZWxhdGl2ZSUzRCUyMjElMjIlMjB3aWR0aCUzRCUyMjUwJTIyJTIwYXMlM0QlMjJnZW9tZXRyeSUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ0FycmF5JTIwYXMlM0QlMjJwb2ludHMlMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteFBvaW50JTIweCUzRCUyMjI2MCUyMiUyMHklM0QlMjIyNTAlMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZBcnJheSUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyMzMwJTIyJTIweSUzRCUyMjM1MCUyMiUyMGFzJTNEJTIyc291cmNlUG9pbnQlMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteFBvaW50JTIweCUzRCUyMjM4MCUyMiUyMHklM0QlMjIzMDAlMjIlMjBhcyUzRCUyMnRhcmdldFBvaW50JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGbXhHZW9tZXRyeSUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14Q2VsbCUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214Q2VsbCUyMGlkJTNEJTIyZzJIUXdzdlRnX25OU1FOTnI0OUEtMTElMjIlMjBlZGdlJTNEJTIyMSUyMiUyMHBhcmVudCUzRCUyMjElMjIlMjBzb3VyY2UlM0QlMjJnMkhRd3N2VGdfbk5TUU5OcjQ5QS02JTIyJTIwc3R5bGUlM0QlMjJlbmRBcnJvdyUzRGNsYXNzaWMlM0JodG1sJTNEMSUzQnJvdW5kZWQlM0QwJTNCZXhpdFglM0QxJTNCZXhpdFklM0QwLjUlM0JleGl0RHglM0QwJTNCZXhpdER5JTNEMCUzQmVudHJ5WCUzRDAlM0JlbnRyeVklM0QxJTNCZW50cnlEeCUzRDAlM0JlbnRyeUR5JTNEMCUzQiUyMiUyMHRhcmdldCUzRCUyMnhMazdXRlJRc0tDbEV3RGN1OG1YLTI1JTIyJTIwdmFsdWUlM0QlMjIlMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteEdlb21ldHJ5JTIwaGVpZ2h0JTNEJTIyNTAlMjIlMjByZWxhdGl2ZSUzRCUyMjElMjIlMjB3aWR0aCUzRCUyMjUwJTIyJTIwYXMlM0QlMjJnZW9tZXRyeSUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyMzMwJTIyJTIweSUzRCUyMjM2MCUyMiUyMGFzJTNEJTIyc291cmNlUG9pbnQlMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteFBvaW50JTIweCUzRCUyMjM4MCUyMiUyMHklM0QlMjIzMTAlMjIlMjBhcyUzRCUyMnRhcmdldFBvaW50JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGbXhHZW9tZXRyeSUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14Q2VsbCUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214Q2VsbCUyMGlkJTNEJTIyZzJIUXdzdlRnX25OU1FOTnI0OUEtMTIlMjIlMjBlZGdlJTNEJTIyMSUyMiUyMHBhcmVudCUzRCUyMjElMjIlMjBzb3VyY2UlM0QlMjJnMkhRd3N2VGdfbk5TUU5OcjQ5QS0xJTIyJTIwc3R5bGUlM0QlMjJlbmRBcnJvdyUzRGNsYXNzaWMlM0JodG1sJTNEMSUzQnJvdW5kZWQlM0QwJTNCZXhpdFglM0QwJTNCZXhpdFklM0QxJTNCZXhpdER4JTNEMCUzQmV4aXREeSUzRDAlM0JlbnRyeVglM0QwLjUlM0JlbnRyeVklM0QxJTNCZW50cnlEeCUzRDAlM0JlbnRyeUR5JTNEMCUzQmN1cnZlZCUzRDElM0IlMjIlMjB0YXJnZXQlM0QlMjJ4TGs3V0ZSUXNLQ2xFd0RjdThtWC0yNSUyMiUyMHZhbHVlJTNEJTIyJTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhHZW9tZXRyeSUyMGhlaWdodCUzRCUyMjUwJTIyJTIwcmVsYXRpdmUlM0QlMjIxJTIyJTIwd2lkdGglM0QlMjI1MCUyMiUyMGFzJTNEJTIyZ2VvbWV0cnklMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NBcnJheSUyMGFzJTNEJTIycG9pbnRzJTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhQb2ludCUyMHglM0QlMjItMzAlMjIlMjB5JTNEJTIyNTgwJTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGQXJyYXklM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteFBvaW50JTIweCUzRCUyMjUxMCUyMiUyMHklM0QlMjI0NDAlMjIlMjBhcyUzRCUyMnNvdXJjZVBvaW50JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhQb2ludCUyMHglM0QlMjI1NjAlMjIlMjB5JTNEJTIyMzkwJTIyJTIwYXMlM0QlMjJ0YXJnZXRQb2ludCUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14R2VvbWV0cnklM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZteENlbGwlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZyb290JTNFJTBBJTIwJTIwJTIwJTIwJTNDJTJGbXhHcmFwaE1vZGVsJTNFJTBBJTIwJTIwJTNDJTJGZGlhZ3JhbSUzRSUwQSUzQyUyRm14ZmlsZSUzRSUwQbT+lF0AABAASURBVHgB7N0L8GVFfSfwPsimMEocRCqYhTD4wEhVNJogKlH+WJSbLCWrhnURjA6usEgMJSJJdEFnNEtiEM2qEEqMAhGkEhONKUq2ltIZEFfE+EqiCT4ANYGU4iMmkbIUdj73T8/cuXPv/d/HOef2ufc3Nf0/5/Tj9/h296/7193n3H3uj3+BQCAQCAQCgUAgEAh0EIHbb7/9/tsfCB/96Efvf8973jNR2Lp16/3ThC1btvTyv6eP/kc/+tH7P/pAuP3223tydBDCEDkQ6DQC+6T4FwgEAlMiENkDgUAgEAgE6kTgjjvuSHfsDNu3b0/bd4Yrrrgibdu2bVc47bTT0nHHHdcLhx9+eKqqqhdynKv8O3bsSJOEaWU/9thje0X6aW/rkw9/oarW5SJjDuIFOuSgLB2379RV6BGPP4FAIDA1AuHITA1ZFAgEAoFAIBCYGoEosHIIcEwEE3UTd8FE3qReMNGvqvWJv2dBHoHD0A8YR+L1r399EnbugqSdS8i9cPvtt6ccxO/cMUmTBHSmCTt3ZHq8+2njl0OWIcuV410zHzrkQDc60hUmVVUleMDAs5AdHRjKHyEQCAT2RiAcmb0xiZhAIBAIBAKBQCAQ2AABE2zBhNuEXDARF6qq6u2euBefSZnI54m9SX6e+GdHQJzAYcj5XDkSa2traW1n2Lx5cyZX7JWMOZBZoEMOdKIjXekOB/fiYSRwdGAHw6ra09ERD3dOojooFogQLBBoGIFwZBoGOMgHAoFAIBAIBAJdRMAE2UTZhNnE2S6BSbWdg6ra7aiYcGf9TMQFE3MTdCFP0MWbyJvUCyb6uVxcU4IHXGAkjHN04AV39ZLrxFUdqS/1Jk+EQGDZERjiyCy7yqFfIBAIBAKBQCAQCEAgOysmxILJcFXtPu4lzoRZXhNtzgjHpN9RMeEWL5iIC/JHqBcB+MOWkwNruKuLfmcx7+RwaDic6lMdhmNTb10EtXIQCEemnLoISbqMQMgeCAQCgUDBCIxyWPJEN4tugsxJEfIE2YRZvGAibUKd88e1DATUibrh5KivXHcveclLegJyZqpq/XgaJyd2bXqwxJ8lQCAcmSWoxFAhEAgEAoEuIhAy149Av8NiwspRqar1Y2Ams5kjp6TfWfEsmAznPHHtNgKcG46NerVzo75d866N9mDXRtBWwrnpdn2vqvThyKxqzYfegUAgEAgEAp1FYBKHxUTWJNYENq/QexbCYels1c8luDbBubFrw6nJITs3HJrs2MRxtLmgjsItIRCOTEtAB5tAIBAIBAKBQGBaBMJhmRaxyD8NAhwbITs32eEVl3dsODd2a6ahG3kDgbYQmMyRaUua4BMIBAKBQCAQCKwoAoNOS1XteSTM5NJuSuywrGgDaUnt3M76d2uuvPLK3u/chFPTUiUEm4kRCEdmYqgiYyAwHQKROxAIBAKBcQhwXKx6C95lEUwUlTGZDIcFEhEWiYB2aLem36nZsWNHz6nRXrdt25biCNoiayh4hyMTbSAQCAQCgUCgFASWVo5Bp6Wq1ndbssJ2WhzrEdwLOS2ugUAJCGSnJr9f44to2jWHxns14dSUUEurJ0M4MqtX56FxIBAIBAKBQIMImNx5p8DEziSvqtadFvHYclLybot7YW1tTVKEQGAGBNov0u/UaMt2bEihzWenxnOEQKBpBMKRaRrhoB8IBAKBQCCwtAhwThytMYHrd1ocv6E0J8VEz06LlWzP4bRAJsIyIcCx0bY5NIJ+kR0a98uka+hSFgIzOzJlqRHSBAKBQCAQCAQCzSNgUjbotHjG2UQunBZIRFhlBDg1nHYODRw4NN794vB7jhAI1IlAODJ1ohm0AoHxCERqIBAIdAyBYY4LFeyqZKfFhI0TI05ahEAgEEiJQ6Nf2I30OzUcfk6Na+ATCNSFQDgydSEZdAKBQCAQCAQaQKBdkqMcFxOy7Li4F9qVLLgFAt1EgEPT/+UzfSw7NO67qVVIXQoC4ciUUhMhRyAQCAQCgUDrCJhIWSHuf7+FEByVfscldlugEqEzCBQqKKem/9iZfif4OEahIodYhSMQjkzhFRTiBQKBQCAQCNSDAKfFOf1wXOrBM6gEArMiwKGxWOBYps845x/c1DdnpRnlVhOBOh2Z1UQwtA4EAoFAIBAoEoFBx8VxFi8dE9YkKnZcIBEhEFgcAhyaYcfOYodmcXXSNc7hyHStxkLeJUMg1AkEAoG6EBh0XBxZySu84bjUhXLQCQSaQYBT49iZoN9aeNCnm+EWVJcFgXBklqUmQ49AIBAIBFYFgQf0NMnpPyo26Lj4WpKjK5yYeMflAdDiEggUjoC+qt/aqbGDyqnR1wsXO8RbEALhyCwI+GAbCAQCgUAgMD0CJjQmNpwWK7YmOqhwVsJxgUSEQGA4Al2KtTujT9udIbf+rt+7jxAI9CMQjkw/GnEfCAQCgUAgUBQCHBfn5U1iqqpKJjQENMmJd1wgESEQWF4EskNjh4aWFi/YAvcRAgEINOzIYBEhEAgEAoFAIBCYHAHOi8kKp0XwRSOl7bgInBjHT8RFCAQCgeVHoN+hcZyUQ+O6/JqHhhshEI7MRghFeiDQNgLBLxBYMQSy48J5qao9d104LlZjOS8mMysGTagbCAQCfQiwAdkeOFYqsB99WeJ2xRAIR2bFKjzUDQQCgUCgBARMPjguVlbtungm16zHxZSNEAgEAquBgA8BcGg4NuwHW7IamoeWgwiEIzOISDwHAoFAIBAI1I4AR8Vkw6SjqnbvuniZ166Lq12X2hkHwUAgEBiFQKfjOTFsBoeGIhZF2Bn3EVYHgXBkVqeuQ9NAIBAIBFpDwITCGXbOS1Xtdlxe8pKXpNh1aa0aglEgsPQIZIfGLo2FEjZn6ZUOBXch0L4js4t13AQCgUAgEAgsEwKcF5MIkwmro+7pZ8dFsHpqsiEuQiAQCAQCdSLAvtid8ZVDNog9qpN+0CoTgXBkyqyXkCoQ2AOBeAgESkXAZIHDUlW7d11MKOy6mFS4t2JaqvwhVyAQCCwPAmwNu+OrhpwZtml5tAtNhiEQjswwVCIuEAgEAoFAYCgCHBcrniYIVbXbebHjInBcTCKGFm43MrgFAoHACiLAmWGHODTZVq0gDCujcjgyK1PVoWggEAgEArMhwHnhuFjhdGQs/66LXZfsvJg8zEY9SgUCgUA5CCyPJGwSZ4ZG7BY75j7CciEQjsxy1WdoEwgEAoFALQgY9LPzwoFB1E4L58XkwIqnuAiBQCAQCJSKAGeGrfJuHjvGppUqa8g1GwJFODKziR6lAoFAIBAIBOpEIDsvVi8N+mibBORdF/fiIgQCgUAg0CUE2C4LMHHUrEu1Npms4chMhlPkCgRKQyDkCQRqQWDYJ5IN+Nl5sQtTC6MgEggEAoHAAhGwO8O2EcFijYUb9xG6jUA4Mt2uv5A+EAgEAoGpEDB4O15x2mmnpaqqkisCjoxl58WAL275QmgUCAQCq4wA2+a3rPJRM/ZwlfFYBt3DkVmGWgwdAoFAIBAYg4DBmvNy3HHHJcEujAG933kZUzySAoFAYJURWDLd2T5HzcKZWY6KDUdmOeoxtAgEAoFAYA8EsvPiCAXnRaLB266L4xXuxUUIBAKBQGAVEWADw5npfs2X6sh0H9nQIBAIBAKBlhHIzktV7f59l/e85z2J82LQjvddWq6QYBcIBAJFI8AuChZ72M+ihQ3hhiIQjsxQWCIyEOgiAiHzKiJg8HVsrKp2Oy8cF8EAHc7LKraK0DkQCAQmRcCuDFvJmXHsdtJyka8MBMKRKaMeQopAIBAIBCZGYCPnxRnwiYmtesbQPxAIBFYeAc6MI7c+fmJhaOUB6RAA4ch0qLJC1EAgEFhdBLLz0v/Oi10XwWpiOC+r2zZC80CgbQSWkR8bypm54oorUjgz3anhcGS6U1chaSAQCKwYAsOcFwNtOC8r1hBC3UAgEGgFgezMbN++PZyZVhCfn0mHHJn5lQ0KgUAgEAiUjkA4L6XXUMgXCAQCy4wAZ8ZHUmJnphu1HI5MN+oppAwEZkMgSnUCgWHOi4E0dl46UX0hZCAQCCwZApwZu9/hzJRfseHIlF9HIWEgEAgsIQIbOS/xtbHFVXpwDgQCgUAgnJlutIFwZLpRTyFlIBAILAEC2XnxmU+BSv07L+G8QCRCIBAIdBCBpRQ5nJnyqzUcmfLrKCQMBAKBDiMwzHnxlbF8bCyclw5XbogeCAQCS49AvzPjIwBLr3DHFOy2I9MxsEPcQCAQWA0EwnlZjXoOLQOBQGA1EODM2D33OzPs+2po3Q0tw5HpRj2FlIFAbQgEoWYQMLj57QFHxgRcXvKSl6TYeYFEhEAgEAgEuo2A3XM/nJnte7e1WR7pw5FZnroMTQKBQGABCDhqYJXOD1VyZvqdF4PeAkQKlvUjEBQDgUAgEEiOBbPrbH7AUQYC4ciUUQ8hRSAQCHQIAQ6L3ZeqqpIBbfPmzen+++9Pjh4Y5DqkSogaCAQCgUBDCCwnWYtVeQxYTg27pVU4Mt2qr5A2EAgEFoRAHrjsvOSjBY6NCVbpFiRWsA0EAoFAIBBoEQELVxat4jdmWgR9DKulc2TG6BpJgUAgEAhMhUB2XjgugsIGsOy8GNDERQgEAoFAIBBYHQTY/vyDmY4Xr47m5Wkajkx5dRISBQJtIxD8BhDIDozdF4OUowTZefHC50D2eAwEAoFAIBBYMQQ4M3bjHS82ZqyY+sWoG45MMVURggQCgcAiETAQ5fde8u6L916susV7L4usmVJ5h1yBQCCw6ggYG4Q8Zqw6HovQPxyZRaAePAOBQKAIBLLzYuclD0R2XgQrbUUIGUIEAoFAILAsCCyhHsYKzoydmSVUr3iVwpEpvopCwEAgEKgTgey8cFwEtOO9FyhECAQCgUAgEJgFAceP89gyS/koMzsCq+DIzI5OlAwEAoGlQSAPMnZffG3Guy5558X90igaigQCgUAgEAi0ioD3ZSyIGVu8V9kq8xVnFo7MijeAUD8QGI7AcsRm56WqqpR3X7z3kh2Y5dAytAgEAoFAIBBYNAKcGcfMvGu5aFlWiX84MqtU26FrILACCGTnxc5Ldl44LoJBZgUgCBUXhUDwDQQCgZVGIO/u25lZaSBaVD4cmRbBDlaBQCDQHAJWwbxsmZ0X2/zZebFS1hznoBwIBAKBQCAwKwLLVM5YY+wxHllUWybdStUlHJlSaybkCgQCgQ0RMFAYMKqqSlbADCLZeckrYxsSiQyBQCAQCAQCgUBNCBiHtmzZkiys1UQyyIxBYEUdmTGIRFIgEAgUjUB2XvqPjsV7L0VXWQgXCAQCgcBKIeArZhS2wOYaoTkEwpFpDtugPAMCJqk6vlV21/j6xwwgNlVkwXS1De2CA0MU2/d598XzuKCs9qS8a7SrcWhF2qQIRLuaFKnIFwisFgJ2ZYxRxpzV0rx9bcORaR/z4DgEARNLE1TvN+zYsaOXw3Xbtm1JvKubJiVlAAAQAElEQVQ8vYT4szIImCiqe21A26C43Rcv7a+trXkcG7SZXFZ7ktk103SVR3yEQGBSBLSZSdvVpDQjXyAQCCwXApyZLVu2xBGzhqs1HJmGAQ7y4xEwUXWOVLB6YYXd1UTV9aMf/WgSUDHpFNxHWG4EcrvIzos2oG1oF5NonstHu5oErcgzKQLRriZFKvIFAnMhsDSFHTGz8CEsjVKFKRKOTGEVskri6NhWNa1amKSura0NVV+6CSzHxrGgcGaGwtT5SJNEdVtV67/5ot61C3XvflIFo11NilTkmwaBaFfToBV5A4FAAALGLnMXi2qeI9SPQDgyGdO4toqASYGObaXdRHUS5gyC/PJygEx83UfoNgKcFzsvAk04L8Kk7UKZHKJdZSTiWicC0a7qRDNoBQKrhYBFWsGcZ7U0b0fbcGTawTm49CFgUmDSapVC5+5L2vCWM2OCu2XLlt4vtYczsyFkjWaYlbh648BU1fpnk22/Z+dFHc9CN9rVLKhFmY0QiHa1EUKRHggEAhshYN5i3GNPNsob6dMhEI7MdHhF7jkR0JE5MXZW1tbWZqbGKGRnZmYiUbBVBNQ958VumjaAuRf3OTDq0vOsAW00o13NimCUG4ZAQ+1qGKuICwQCgSVGwAKdBbvYlam/ksORqR/ToDgGAZ14lp2YYSQ5M5whNIelR1wZCJgMZgfGvXrjvLjWJaE2EO2qLjSDTkYg2lVGIq6BwKIR6D5/C3YcGu/6dl+bcjQIR6aculh6SayYczx05rqUNRm2VRuGoS5E66HDYcnOi3pH1e4LZ6PO+kcX/WhXkIhQJwLRrupEM2gFAoEABMxZjI3uI9SDQDgyY3CMpPoQ4GyY3OrE9VFNyeqG40QMAx510g5a0yOgjq1imwQqrW7q3n1BNwd1jme0q4xIXOtAINpVHSgGjUAgEBhEwKKbeUssvg4iM/tzODKzYxclp0CAo2E1fooiE2dlFExkTaAnLhQZa0OAI6F+q6r32eSec5mdF3VTG6MhhPCNdjUEmIiaC4FoV3PBF4UDgUBgDALmK2zMmCyRNAUC4chMAVZknQ2BvPJgJWI2ChuXclzJpDmMw8ZY1ZVDvXIe8+4L50VgpOviMY4O/tKjXUEhQl0ILKZd1SV90AkEAoHSETBmma9kW1O6vKXLF45M6TW0BPKZ7Da1at4PDx4Mgx2C/vi4rw8B2HIWq6pKroxxdl7c18dpY0rRrjbGKHJMj0C0q+kxixKBwEIQ6DBTC37G0A6rUIzo4cgUUxXLKYjV+q1bt/aOGzWtoYm0nZkwDvUjfccddyQTPPWJOudFYIw9tx3IEe2qbdSXn1+0q+Wv49AwECgBgbwr4328EuTpsgzhyExXe5F7CgR0UBPgNie7vtOOrzCFqJF1CALqjlOYf/eFo5idF/dDirQSpW7JFu2qFbhXhkm0q5Wp6lA0ECgCAWOYBcIihOmwEOHIdLjyShfdJFhHbVNOE2w88W6T7zLx4iTAjwNDr/m/PIZKfYFs6rg+ihtTina1MUZdzxHtqus1GPIHAt1CIHZl6qmvcGTqwTGoDCBgMmyF01GvgaTGHzNP78s0zmxJGKgvEznOi+M11PK7LxwGk3jPJQRyRrsqoSaWS4Yi29VyQRzaBAKBwBAEjLGxKzMEmCmiwpGZAqzIOjkCJsXeYZi8RL05GQcy1Et1+aiZwMGJA+Mebvn4WInakjXaVYk1022Zol11u/5C+kAgI9C1a+zKzF9j4cjMj2FQGIKA3RDvqwxJaiUqGwdytMKwY0xM3Oy8CES3++Krb3k3S1yJQX1GuyqxZrotU7SrbtdfSB8IdBkBC4ixKzN7DYYjMzt2D5SMyyACJskmxIs+ksQ4kGVQvlV9tuMCj6qqUp64lbz7MlhPZN+yZUsrX8Ab5N3/HO2qH43u30e76n4dhgaBQJcRWFtb641rjk13WY9FyR6OzKKQX2K+Jskme4tWMYzDeg1wYKz25N0XzouwZcuW9QyL+DsDz2hXM4AWRTZEINrVhhBFhkAgEGgYAXMm43TDbJaSfDgyS1mti1PKpGDtgdWFxUmxmzPjYMV1d8xq3HFe6O3dFw6M3THOCzzcdw2FaFddq7FuyNu1dtUNVEPKQCAQmBaBPG+KXZlpkUspHJnpMYsSYxAweV7kOwyDouVJ+6BxGHweLNfV534Hhg6lfTqZTLOEaFezoBZlNkIg2tVGCEV6INB5BDqjgLkTm9QZgQsRNByZQipiGcSwukkPKwuuJQSOzKBxIOcll1xSgni1ycD45d0XRL2839XdF/L3B/XlOdoVFCLUhUC0q7qQDDqBQCBQBwLGOIuRddBaJRrhyDRR2ytKc8eOHcnkuTT1s3GwCyM4h/rQhz60NDGnlofB27ZtW6qqKtEL9vn42NTECi4Q7argyumwaNGuOlx5IXogsIQIWHgV8iLLEqrYiErhyDQC62oS1fnW1taKUN4kPweGwSTfpN/7IgQU59rFAGd6COS3++IIWdde3id7fxh1T99oV6PQifhZEYh2NStyUS4QCASaQsBc5corr2yK/FLSDUdmKau1faVMCkykS3IQTPQdt6qqKtmFsWsBGbsxhx12mNvOBE4ZR6yqquTquNwy7r4MVki0q0FE4rkOBJaoXdUBR9AIBAKBQhAwhzJXEQoRqXgxwpEpvoq6IeCOHTvSscceW4ywjIFdipNOOintt99+xcg1rSAcGE4Yp0xZzouwZcsWj0sfol0tfRUvRMFoVwuBPZgGAoUgUK4Y5i7GdzaqXCnLkiwcmbLqo7PS5BXOkhRgEC666KJ05plnpn333XeXaO6l7Yoo8Gbbtm3JbhIHhqycF1vO7gsUtzGRol01Bu1KE452tdLVH8oHAkUj4MQFG1W0kAUJF45MS5WxzGx0OCsIJepo4v/Wt741nX/++cmRMjLee++9Sbz7koLdFw5MVVUJphyX7MCUJGdbssBgy5YtbbGbio/205V2NZViK5A52tUKVHKoGAh0GIG1tbXeHCWOl01WieHITIZT5BqDgBfTrCCMybLwJE7B29/+9rRp06bEkVm4QH0CZAfG7otozouwZcsWjysbrrzyyhTtamWrvzHFw141Bm0QDgQCgZoQWNvpzLBVNZFbajLhyCx19TavnEm4VQOdrm5uvsZVJ02OwWc+85l0yCGH1El2Zlp2X/LxMUQ4Lxwuq/2eVzlEu1rl2m9O99VoV83hF5QDgUCgHQQs4plbtcOt21zCkel2/S1cehOD97znPVPLwUm59dZb07ve9a507rnnpuc+97npl37pl9Khhx6a9t9///SgBz1oV/AsXrp88iunPDrTMOck3HTTTcl1mnJ15YUXB6aq4vjYOEzhFO1qHEKRNgsC0a5mQS3KBAIrgEBhKpqjCOHMbFwx+2ycJXIEAqMRsBOzZcuW0Rn6Uj796U+nCy+8MB1//PG9L4mdccYZ6eabb04HH3xw7wjRZZddlj7+8Y+nu+66K/3whz9M9913X+/qWbx0qxTyK6f8fvvt16OHLvp97EbeMg4jExtKMIHiwNiBwcLuizApdsqsUlhbW0uTYqPe1f8qtqtVahN16Brtqg4Ug0YgEAi0gYD5jnlDG7y6zCMcmcXV3kpwvu2229LWrVvTkUcemU455ZT0zW9+M51zzjnpnnvuSY55WXU/77zz0vOe97xdOzJeyrcjAyBXz3lHRj75lVMeHfTQRR8f/PBVftHBasppp52WjjvuuJ4odpDi+FgPirn+qF/1rL7Vu/rXDrQH7UL70E60Fzt52o92pD1h7OpZvHT55FdOeXTQQxd9fPDDV/kIy4mA+lXP6lu9q3/tQHvQLrQP7UR70W60H+1Ie4KIq2fx0uWTXznl0UEPXfTxwQ9f5SMEAoFAIJARsPBiETQ/x3U4AuHIDMclYudE4Prrr+8dFzvmmGPS9773vWQg//u///vkS08nnHDCri+IzcmmRwc9dNHHBz98HUMjx7w8pi1/xx139H600u4LJ2bz5s3J7gsHZlpakX9PBK6//vqVbVd7IhFPdSLATrAX7Ab7wY6wJ+wK+8I5qYMfOuihiz4++OGLPznq4BM0AoFAoPsImDsIvrTYfW2a0yAcmeawXUnK1113XW/34TWveU0yYFt5NGgfffTRreCBD3744k8OuyHkalqAO/ocGLz8IGc4MJCYP6g/9ag+1av6Vc/qe37qG1PABz988ScHeci1cenIUSoC6k89qk/1qn7Vs/puQ2Z88MMXf3KQh1y18g9igUAg0EkELIBeeeWVnZS9LaHDkWkL6SXn87nPfa63Um4gftnLXtY7Nnb66acvVGv8HecgD7mseJKzbqE4MHZeTEDQjuNjUKgnqC/1pv7Uo/pUr/VQn40K/uQgD7nIR87ZqEWpRSCgvtSb+lOP6lO9LkKWzBN/cpCHXOQjZ06PayAQCLSLQAnc7MiYY2zfvr0EcYqUIRyZIqulW0Jt3bo1Pf3pT0/PfOYz0+c///l06qmnFqUAechFPnKStw4BvYTn+BgHhrGJ3Zc6UN1NQz2pL/Wm/tTj7tTF35GHXOQjJ3kXL1VIsBEC6kl9qTf1px43KtNmOnnIRT5ykrdN/sErEAgEykHA3ELYsWNHOUIVJkk4MkVVSLeE8bUoA62z3sKrXvWqohUgHzkFcpN/WoGtjGzbti1VVZWskDjjHg7MtCiOz69e1I96EtTb+BKLTSUfOQVyk3+xEgX3YQioF/WjngT1NixfKXHkI6dAbvKXIlvIEQgEAu0h4HiZ+UZ7HLvFKRyZbtVXMdL6HRdf5Xnxi1+crr322t7vvxQj3BhBfE2IvOQmPz3GZN+VxIHpPz7GefEOzNra2q48cTM/AupDvagf9aS+NqRaQAZykpfc5KdHAWKFCA8goD7Ui/pRT+rrgaSiL+QkL7nJT4+iBQ7hAoFAoHYE7MiYg9ROeEkIhiOzJBXZphq/9Vu/ld72trelT37yk+nMM89sk3VtvMhNfnrQZxRhuy/Djo8xLKPKRPxsCKgH9aFe1M9sVBZbitzkpwd9FitNcIeAelAf6kX9iOtaIDf56UGfOuQPGoFAmwh4d7RNfsvEy3xDiF2Z4bUajsxwXCJ2BAK//uu/nhx18AOVVghHZOtENPnpQR96ZaGtfHBgqqpKPntoW9cOjGvOE9d6EYC/elAf6qVe6u1SIz896EOvdrkHt34E4K8e1Id66U/r2j356UEfenVN/pB3eRHgpNx6663JjuG5557b+/CP9mpHcf/9909+XykHz+Kl+6CF/Mopj07hKC1MPKc/4j2Z4fCHIzMcl4gdggCjs++++6YPfehDvd9vGZKlc1F+14E+9Hr2s5/d+/0XL+9ThPMibNmyxWOEhhBY9nZFv4agC7JjEIC7fq1/6+djsnYmiR70oRf9OiN4CLp0CHhn68ILL0zHH3982m+//dIZZ5yRbr755nTwwQenl7zkJemyyy5LHO+77ror/fCHP0z33XdfIspppgAAEABJREFU7+pZvHT55FdO+f120kEPXfSXDrQ5FDr22GNT7MgMBzAcmeG4lBNbiCQGTQbHy+2FiFSrGPQ66KCDeu/7cF7svtjKrZVJENsLgVVoV/oNPfdSPiIaQwDecNevG2OyQML0oh89FyhGsF4xBG677bbkK3pHHnlkOuWUU5LfPzrnnHPSPffc0/vJBe3yvPPOS8973vOSHRc7L5xvuzGgcvUsXrp88ivn0+PooIcu+vjgh6/yqxzMR5wWWWUMRukejswoZCJ+FwKOMRxwwAG9FZZdkUt4c/XVV6enPvWpib5LqF5xKsG5qXZVkrJWHulJ35LkWlZZ4AxvuC+rjvSiHz3p6zlCINAUAtdff33vuNgxxxyTvve97yWOhyOOb33rW9MJJ5xQ2wkNTg566KKPD374ctrJ0ZSOpdPlyAixK7N3TYUjszcmEdOHgBdLGRIGpS96aW/pSV96L62SBSgGXzjDuwBxGheBnvSld+PMVpgBfOEM71WAgZ70pXcN+gaJQGAPBK677rrkqPVrXvOansNip4STcfTRR++Rr6kHfPDDl4NDDvKQqymeJdNdW1tL8Z7M3jUUjszemETMAwh4Ac8KyDXXXPNAzGpc6Etv+q+Gxu1qCVf4wrldzovlRl9603+xkiwnd7jCF87LqeFwrehLb/oPzxGxgcB0CHzuc5/r7cBwHF72spf1jo2dfvrp0xGpOTf+jp+Rh1x2aMhZM5sZyLVXJN6TGY51ODLDcVn5WC/aefnu3e9+d23bxl0B1fY2vekPh67I3QU54QlX+MK5CzLXJSN96U1/ONRFN+ikBE+4whfOq4QJfelNfzisku6ha/0IeCfFD7A+85nPTJ///OfTqaeeWj+TOSiSh1zkIyd55yDXqaKOlt1xxx2dkrkNYcORaQPlmnm0Qe4Vr3hFuvTSS3sv7LXBrzQeXkSkPxxKk63L8sATrvDtsh6zyk5v+sNhVhpRbm8E4AlX+O6duvwx9KY/HJZf29CwCQQ4wRwD76YIr3rVq5pgUxtN8pFTIDf5ayNeKCGOjBDvyexZQeHI7IlHPO1EwArHz/7sz3b2xy53qlDLfz9CBwd41EJwxYnAEZ5wXQAUxbCkPxzgUYxQHRYEjvCEa4fVmFt0+sMBHnMTCwIrhYBjiZzhF7/4xb0vd/qqWBcAIOe1116byE1+enRB7nlkjPdk9kYvHJm9MVnpGGdOL7rooiSsNBAPKA8HAS4PRMVlBgTgB0dhhuJLVwQOAlyWTrkWFYIfHIUW2RbLCg4CXOoRMqgsOwI+FPG2t70tffKTn+zs4iUnnvz0oM8y15n3ZK644oplVnFq3cKRmRqy5S7g91Pe+MY3Jisdy63pZNrBAR5wmaxE5BqGAPzgCM9h6asWBwd4wGXVdK9TX/jBEZ510u0qLTjAAy5d1SHkbg8Bn+52NMsPVNrRaI9z/ZzITw/60Kt+DlNQbDCro2XxnsyeAIcjsyceK/3kk4Zf/epXk7OnKw3EgPLwgAt8BpLicQIE4AY/OE6QfWWywAMu8FkZpWtUFG7wg2MdZP/qr/4qVVW1K7zkJS+pg2zrNOABF/i0zjwYdgYBX/3ad99904c+9KGl+aCPD1/Qh17060xlTCEoR8bxsu3bt09RarmzhiOzHPVbixZvfvOb02//9m/XQmvZiMAFPsumVxv6wA1+bfDqGg+4wKdrcpcgL9zgV4csb3rTm9LJJ5+cPvWpT6X7778/ff/730933nlnMmH413/91zpYtEoDLvBplWkw6wwCJvkHH3xw74ctOyP0FIL6fSX60XOKYp3Jyi5deeWVnZG3aUHDkWka4Y7Q9zsE3/3ud4v71GIp8PnkI3zgVIpMXZADXnCDX3nyLl4iuMAHTouXpjsSwAtu8JtX6rvuuqv3hUYvDf/iL/5ij5yV3fe9733p9ttvTx/96EcTZ8bkwa6NDMr8wi/8Qvrrv/5rj73rQx7ykN5uzmGHHZakS5BfOeU9c5jyTo84aVW1vgskrzx1BLjAB0510Asay4OAY1cHHHBAuuyyy5ZHqSGa0I+e9B2S3Oko78nEjszuKgxHZjcWK32n05911lkrjcFGysMHThvli/TdCMALbrtj4m4QAfjAaTA+nkcjAC+4jc4xeco//dM/JRMeZ+z7Sz3ykY/s7ci8//3v74/e657T8vznP7/3tSe7OZyTF77whT3nZ6/MfRG/8Ru/kTg9yjgOY0coO0Z92Wa+hQ+cZiYwqmDEdxYBL8J/73vfW9qdmMGKsTNDX3oPpnX5efPmzV0Wv3bZw5GpHdLuEbztttvSzTffnE4//fTuCd+ixPCBE7xaZNtZVnCCF9w6q0QLgsMHTvBqgV3nWcAJXnCrQxmOzHe+852hpI488sih8f2RjqN5zo7Q2WefneyGOJ4mfljg/FhRPemkk3rJxx13XDrqqKMSWXoRNfyBD5zgVQO5INFxBHya2A7dNddc03FNphOfvvSm/3Ql681dJ7XsyLAhddLtKq1wZLpaczXKraO/6EUvqpHi8pKCE7yWV8P6NIMTvOqjuLyU4ASv5dWwPs3gBK+6KP7Mz/xMb0dmGL0vfOELw6L3ivva176W0Kmqqvcjwl/60pcmckpOPPHE3nG0/fffP+3YsSNNym8vAUZEwAleI5IjekUQ8GORZ5xxRnr3u9+9NC/2T1p1jonSm/5wmLRc6fmyM1O6nG3IF45MGygvhMfkTP/0T/+096Lr5CVWN6fjH/BaXQQm1xxO8Jq8xOrmhBO8VheByTWHE7wmLzE+JwfEjkzeWbFbYvLvasUz75r0U7Fz8u1vf3tXlDPrdmAcExP+7d/+LeX3bXZl2nnT76g4zoan/Dl4SX9nttr+wwletREMQp1E4BWveEXvPbC8a9hJJeYQmt6XXnppgsMcZIoq6girxY+ihFqQMOHILAj4UthaobjvvvvS0Ucf3RPJi6hVtf7yaVWtX73EWsfZ7fxyq5da0fOyrMlCj3GDf/DT6fGfls2gnHCCF9ympbVK+eEDJ3iN0hu22lZVrbezqlq/em9gmnYhrzLqeRSvieN3ZiRXbptouxen/WhHdfHZyWrXfzjBC267IuNmLwTgAyd47ZU4Y4R3YbxPYtKvnpEx+efgqO/nPOc5onohvy9zww03pHvuuacXZ5KUPwogwgv92qO24/nWW29N//AP/9D7AMD27dtFJTyf+MQnJj/gJwJffaHutgUneMENnwirh8DWrVvTz/7sz3b2xy7rqjE/mgkHeNRFc5F0LJ5ke7JIOUrgHY5MCbWwQBmcHf3VX/3VPSR48Ytf3PsEaV4lHLW6uEehCR5s8ep4/RODCYrNnQU/fPGfm9hOAvCC287b+D8CAfjAaUTyrugDDzxw1ydvc3vz2VsTvV2ZCrnRfrQj7akJkeAFtyZoLwtN+MCpbn3shPhqGaeEA/OXf/mX6X/+z/+ZOC6cGfwuvvji3nNVVekb3/hGeuxjHyu655T8xV/8RW9Xu6qq9IY3vCF51oa1FTs66D71qU9NjpL1Cu38c8kll/Q+8VxV68fRXve61yX5dybV+h9ecKuV6ACxeCwTgc997nPpoosu6oUyJWxXqowFXNrlXD83R8vihzHXcQ1HZh2Hlf37kY98JB1//PET62+1sarWV86tIFpJzCvV559/fhJXVVWSL+/u5NXJnK9/1ZGTZKLQH6dc/3MWTlxVrfOuqip5luZqpQVvtP7sz/6stwLV/5zjXcmhHNmttls5FchZVev0ySDPsAAvuA1Li7h1BOADp/Wn2f6qV/UjVFXVa1vqDDV1qC6rqkq+ECVuWOivV/nRyjRctZGqqnpfj5JX8AUqA53V7F/7tV9L7p/5zGemm266KaFBLmXR8lxV621GfJZB+6+qdZmf+9znpnHtKZeBF9zyc1z3RgA+cNo7Zf4YTkR2pl1/93d/N7FPnFdOrKNinqW9/e1vT5/97Gd3HR/rT5PHc5boyiuv7C0McdCV8ywNTbTREzhT4usO8IJb3XSDXvkIvP71r09vfOMb06GHHlq+sC1ICAd4wKUFdhuxiPSaEAhHpiYgu0jG4GlyZotyEvlN1JwzdT5cWSuN+WiE8h/72MfSP//zPyefEv2d3/md5Is/zo0ffvjh6aqrrpJlr2AiacXQyqdEE0kDvhVMzzmYOPpcaT5T/vu///vJCqkJrTyOedx4443JxGC//fbrHfvof5bnSU96UvI1Icc8PDse4niHF21Nhh0voRcefjsCT/kGA7zgJu9gWjynBBf4wGlePDgRBh7tyFedcnvTFtAWf+655yYvW3seDNrh2tpaTybtzEvY8mhnHBar8OSVRxvQFqymaxe33HJL+vM///PkXlvSfpTNAS00lbeLmdujdmO1XTv68pe/nD7zmc/kImOv8IIbemMzrmgiXOADpxWFYCa14QU3+M1EIAp1EoHrrrsuffWrX02vetWrOil/U0LDAy7waYpHG3TtyAjmPG3wK5nHPiULF7LVjMAAORMtzoaVwf4kTkdVra8yV9XunQ8rlpwMRybkV9Y1B5M6tBzNMPkzufNspyPnGXa1Yuhb75wSMsmfeeT8VjhNVl3FDfJ+xCMe0ftqkDRh8FmcSeqmTZsSBwavD3/4w4kzRkbGIK+Ikv/hD3+4IkOD/PiTdWiGFY+EC3zgtBEUHFBOa1Xtbm/9uxd22qSjpX2hp+60Qw6MeJ+uNVmT1h84Kxwhn8MVrz3m40BkFIe2qzycXI6R50mCY3HarrzakaugfZFJW9WOOcjiNwp0gVuWbaP8q5YOF/jAadV0n0dfeMENfvPQibLdQuDNb35zymNatyRvXlq4wKd5TsGhDQTCkWkD5UJ5mOQ5HjMongmf1bscODDymEBaua6q9Umn1W7xORgs8/0018c97nG97HZK7Mz0Twp7CQ/8McGtqnXe/WfNJR++c9eHo+JeGHwWl3d/fDkILxPXPJG121RV67Q5MlbblRkV4Aa/UemrHA8X+EyCAWfABCu3Ndd89Eb5YfXI2fBytfRxwc5h/5elBvNyjNV1Va2/o6DOlRnMN+qZs6v8YLr2NRg36TPc4Ddp/lXKBxf4rJLOdekKN/jVRS/olI2Ad6KMb6eeemrZgi5IOrjAB04LEqEWtuZjO3bsqIVWl4mEI9Pl2ptT9i9+8Yu941+TknFcRl4TSRNOx7s8zxusGD7taU9LV1xxRe/l1+xc9NPlaGzfvr332wx4O77Wnz7pvRV0A7ovCdk1smJu5d5nGdFE22Q2r9yPostpg9+o9FWOhwt8msKAw8rB2Yg+J4OzMSqfXZzcltX74LsNo8ptFD+P7srCbyMeq5gOF/isou7z6gw3+M1LZ4rykXWBCFx22WVp0p3gBYq5UNbwgdNChZiTuTHMvGhOMp0vHo5M56twdgW+8pWvpMc85jEzETD5977MTIWHFOJg+NGqYcfKBrPbGfI+wmD8JM92fxwve/nLX947VjasjKN1VueHpbOv768AABAASURBVOU4uMEvP8d1NwJwgc/umHrvOL6OmWkD2oL3mYatSnFSOav5vZr+euUs29VRlnReztf2tGvP8wRtGV3vyqA3TT+BG/zm4b+sZeECn430g7sdCNhvlHeWdIsqdbWVWfhvVMbOtfbcnw9u8OuPi/vlROC2225LN998czr99NOXU8GatIIPnOBVE8kayExHwjsy8eWylMKRma7ddCY3L33btm1j5fUJUV/xGJupL9GRM5M/K+I+Jfqa17ymt4NiJbsv20y3HAwvc486VuadA6vwVtl/+qd/uvepU1vDVtSnYZgnwfndC2VNeK3OOK5WVVVCkyx2ZqQPC3CD37C0ZY6ru10Ne0fGEUCT0XE45pf9tUUOzX/5L/9laHa7hmSuqsk/mauN+YFEbVzb5vj6atmkL+17N4Z8nCU0tKuhwg2JjHY1BJQHovQ3+DzwGJcpEIAb/KYoElk7ioAfc33Ri17UUenbFRtO8GqXa73cwpEJR6beFlUQNZ76O97xjnTQQQelc845Z6hkvjDGKehP9H6C0B+X7034vWTtGI6rH5gySUTDNb9LYyL32c9+tvf7Csqi5+U6TkTON5hn8J0V5fpDLou3ySXemQe+6MqjzEbPZCE/feQXxKEt+OwqeugMyimvQGf4uV+lMGu7GoYRbNUlzPuDOGnwVw+5XtWRtoSWOGnKuX7wgx8c+hsc6lhdy/fsZz87cVA4KmjggZc0V8/icxnlrGSjL/0Zz3hGck8ueXP7U0acNHJ5Jiu6f/d3f5f+5m/+ZuIjnNGuJrdXdlzsjFRVlaqq2usT1xdeeGEvftAxtltRVetlcpqdvbW1tR6NqlpPs7OhLgWOtbxVVfV+S0ZcDnZoqmq9DBpoCe7RqKr1NPfKoGXH6Dd/8zd3yYdG1iXny3kzX+l0Fi8/GkJVrX/mG1262Xn0/qJ7eYVVbVd0X7Xgx1z9uOuq6T2LvnCC1yxlSyhjPGZnjD0lyLMoGWJHZlHIN8xXA/fjT3YtvHsyzKGRdsABBzQsycbkDcpWr31m1yRy4xKLzwE3+C1eknYl6FK7ypPJqlqfSBq0fFq5hjY2FnTtuarWedoxMgHl2Iwt9EBitKvv9t6V28heqVufy7aTymH0wYh8nA+Ujob+y7/8S++z26973euSL9wpo24c9bPbqpwd4Hz0UDnOqx1Z78v58AjngPPgU93oKCNfDtK1K/mVE283zlUYRk88+Q455JCefGRAQ9vs1yPzHfxEOD3Q8K4fm4mvHWR6aGd2zu1EupdPWNV2RfdVCp/+9KfTfffdl/wG1qR6c5yrat1eVdXua78jPAktfYut02778+sjHO7B+P48s9zXQRdO8ILbLDJEmTIQ6LQjs3379jJQLFQKnroVYhPub33rW3tNEP793/89PfjBD1649FazTRBcFy7MhALADX4TZl+qbF1pV9o+G6FtCXZV7KQ0XRnaMX455F2kSfhGu3pomsRe5brNk3W7bP0fdvA1PJ/UhrmJPZp2fdUN5yI7s16ClycH716hbWElf/CD08MRQEc+dD27l8aJcPRVOQ6TNnf33XdLTsPoSSCfd6nck4Ezo23268GpkU4WV3zpwXHxnI/H4ouPuFGhjHY1SrqIrwsBX+HaqC0M46VtZ3uVr7lvDcu/THHwgltXdTIeD3tHtKv6zCL3WEfGOxYln7877bTTUsnyVdXu1Y2qav/eOyUGvtww3HNoHDk77rjjequBD3rQg3JyXKdAAG5Wcqqq/XqtqsXyjHY1RUOZMmu0q+/uQmwje2UFuKrW+wIHwC5HLsypEZef89VuhoG/qtbLOYKV01w5Fa79gbPiSGJ/XL4f/NQ2ntnJkWcYPfGj5JPWH8Z9Ilw/tOPXn3/UvXaVJ6ij8kR89xH4yEc+krKDXIc2ub/079q4z7TtiuSjj3Ywc/yoqz6r/6Erj10f9DyLd19V633TvTyC3Ry7PVVVJVfP4oVRR0ilTRLgBbdJ8i4kzwZM48tlafzL/pwEzswGOC4kOTsxJcu3ZcuWnrOQB5BFXL2o3F9Bfijy/PPPT45hVFWVfvzjH/cnx/2ECMBtn332WXj9LqJN4RntasKGMmW2aFeb9kBslL0ykRn3yXS/H8QBQczVs/tZPiE/6Jygk8Ogo4LXKKcnl5nmapJiB0afE2bdVdSuqqrqvZMzDf/I2x0EtI+bbropaTN1S20XUzt0hJLDwoHRB0cduZyV/zA+nBzHSC3A0tFx0rwAYfGi/4imHVH5p+EPL7ihPU25UvI67m2uXoo8i5Bj7I7M61//+mSbvESQrrjiit5kfPv27UXuypAPfouo1MyTDPfee2/v0YRg69at6Zvf/GbKcv3kT/5k+sEPftBL7/sTtxMgADf4TZB16bJEu2quSqNdzWavvOBuUpNrxtfwbrjhht6j66Me9ajky4i9iAf+mIh5X+aBx5EX5TjueMjkXZTsrHBy/CaVhSETKF/QW1tbSwcffLCscwVHynwlEm2ErF4PrkaLnySscruaBJ9lyOMoIsfaUcNp9dG2q2rd0a2q3R+PyHQcv0JXm5zkyGUuN+11GB9HQu3O4o2eI2/5uG7/EU3HM6VPG+gFN/hNW7aE/OHIpPE7MgBilEvb9bAbY7eDbEKp8sFvkQ3d18p00kEHJsu0adOm3lec8nO+2tI1aObntq5WeZp4KXBW+cfhYCIDv1lpd7ncrO0q62wCqZ7Vt8mfPuzIQU4fvMonv3KDaaOeZykzitZk8XvnGtd+9s69HhPt6qFpEnvlHRcrsz5tXVXVXp9MN9nyueGqqhJn5X3ve19iC70LwDlwJMunsSf5hLxyHJQ3vOENvR2Nn/qpn0qOdakx77V4GZ8caIrLuz7u5wl09AEAHwKoqirh71n8OLomZVas+234KrercVgtU5qPP7CTs+ikX9iRyGFw50+bGqRb9+4j+tPymfSIJtrjAtzgNy5PyWnmmiVuOLSF2dgdGUJYvS9t18OKMLm6IB8ZFxHU2Stf+co9dmAG5YhPcg4iMvmzTy/Db/ISy5Gz7nZlkoiml7CXA6H5tIh2tXvHeBBJ/Q0+Od7KbJ54Dftk+tvf/vYk3XGVPPl39ZzjfcZd+0PbNbdD+Xxem6OCn6vJnXLoyiuPNGXEC+K1acG9NHnkzfTQci9OGj3yCrM4afJIc818XT2LRxd9fDz303BPFldpAtzo6L6oEMLUhsAXv/jFiT/zXgdTu5H974PNQnPwHbNhNOrgM4xufxwHCn79cV26D0dmg9oCkBXTUnY98m4MuYjuWrJ8ZFxEgEl29kbxd7b061//+h7JVvFsM/ev6Fktr6qqtxqJrlX0PQrtfJBH2nOf+9xevv4jEFbSPVfVOg0r1TuLJKvmvrwjSPchAvE5yIfmID/P4qtqnR7eyshPfveCe3Hu8covJeJFJvHKWo0Rqmr3lrqygzjInwPc4JefV+UK91naVcZH3TnvbPXLj0w6m4ymepBHvaifqtr7pU7pgjpVBq1h9YqGs9t4WHX3rFwOyilfVYtrP1mWwWu0q0FEdj/rb/DZHRN3kyIAN/hNmj/ydQ+Br3zlK8nvXrUl+bgjl+NkcBTTcTF2mTM+Lq80jowdxXz0y9jMftf51VC4wQ+/LoSQcU8ENtyRkd3ERYMrYeuqfzeGbELp8pGxxPDoRz86ffnLX95DNKt4tpnz7xCYKDra4CU/L/vJ3P87CZ5z8AnApz3tab1VUIaGM2TSaOLqGIhVQsbImW90lXOW3VEMq6Te4xEnmKxaJdHu8qqjeAF/k130yEU+9JyR/fCHP5zwFNyLYzBNbB0BUYZsZJIHPRPeSX+PQX4BbvBzH2FPBOACnz1j15/UpWM+T3ziE9ONN96YnvSkJ60n7PyrPtTLsJc6dyb3/ve3C+1xWL064uMIDh6f+MQndv0wa4/Azj8ltJ+dYgz9Dzf4DU1c8Ui4wGfFYZhJfbjBb6bCUagTCDhKeeihh84kq0W7qlpf2Kmq9StbO44YWz7qyOWocnYSjcned7HI5EjmqLw53i4le26cr6rdR0XrfEcVbvDLPLt2taBfwvx8UbhN5MgAyeRv0bsyg7sxGbTS5ctylnZ9/OMfnzba2nUO9qijjko+18xw+SoI54JzMKiPnRVOkHi/ecBBMNmUn4Mk3uqKc63uBS/riXOfg88pus9l3OeAL3qMoThy+R0HcqLz1a9+NVntEfILgpwneRlPV7JJI5tnckujn5cNxW0U4Aa/jfKtYjpc4DOt7v11pqz6z0duPA+2i43qVZnBUEr7GZQrP8MNfvk5rrsRgAt8dsfE3aQIwA1+k+aPfN1DYNbjg2ysBb7BIN6YaLzlgECEUzHq6OPgkUv5BcchlVHWM7p4WbxUxvM0fJRDa5AuGcmKFj7TBMcu4TdNmZLymgPDpSSZ2pRlIkeGQCXsegzbjSGbULp8ZCwtWLFmYMbJZQDsT+csjDoXe/jhhyer4f353Ts2VFXrqzzK939diFMjTj4hp+mUJp3ihgUrOVVV9fjZCSKnrW5fJ+LU+FIR/Rg85ev6PQa0BLih7z7CngjABT57xm78pN4cIRiWc1S7GFevw+jkuFbaT2Y2xRVu8JuiyMpkhQt8VkbhGhWFG/xqJBmkCkPA4tyosbkwUYsTB27wK06wEGgiBCZ2ZHh8i9yVGbUbk7UsXb4sZ0lXuxAcAEd6RsnleFd/2rjJpq8B5V2OnM8qx7jfe+in7d7Xhl772tcmOyOOpokbDIyO1XirOjlYvbcSo5zv3NMr79oo71vxZMv5+1+clT5NgBf68Jum3KrkhQt84DSNzhxadTuszKh2MUu94rHI9jNMP3Hwghv8PEfYEwG4wAdOe6bE0zgE4AU3+I3LV0payDEbAt4ZefCDHzxb4RUvBTf4dRUGR+3jaNmEtbfIXY9xuzFZ/NLly3KWcq2qKj3jGc9IdjRGyWRy6eU877UYEJ2J5dDmnY7+clbHTRDFcSbks2XrOQdncfPqeo4bdnVEzdE07770p+NrZdFvOYiX7iV+uz6ejz/++IS3LeY8cLtysuggj5cFdfxxOz7yjQrwgltVVaOyrHR8VW3croYBpK3ZkcltSD1pQ/0DTH+7mKVeS2g/w3QXF+0KCqNDVc3WrkZTXGyKdxC08aaliHbVNMKLp58X6B70oActXpgOSgC3jGEHxU+bN29O4chMWHOL2vXYaDcmi1+6fFnOkq7PetazkmNY/TLZhbEbYpB1DtVL8o7i5GNjXs7vz5/vORgXXHBB76tljobJZ+LoRX/lq6ra6/cectnBq3JewPdODgeqPx1d9KuqSiazr3vd65LzsfI4XuadGRNgNMS59r8sOM/vMaAHL7i5jzAcAfjAaViqduQ3eHy17DOf+cyuLIP1lH//o/+lTnlyu0BnVL1mp8gLpYMO66Lbzy6FB27gBbeB6HjsQwA+cOqLitsNEIAX3DbIFskdRqCqqt64++Mf/7jDWixOdLhV1TotcNbsAAAQAElEQVSGi5Nids7mvuHITIHfInY9JtmNySqULl+Ws5Trr/zKryRf9+qXxzEtqxOu4jkJngU7HY5wJQkDweT0Yx/7WO+rZf350FFWGPZ7DyanSHGanOXOz/j205FHwF88egL64oWc5gVCzzmg7TiZ/K6epQ3yQCuXdS+/q7w5wAtu+TmueyMAHzjtnZJ6P0yo/tSDnS336kFe9SIe7pxVbUHcqHYhLed39YyOcsoL7sXlkNsIHkJ//ea03AZyGXTRl9/VszRyk185z2jlsu7ld5W2UYAX3DbKt8rp8IFTkxhYOLEQUlXrE5u82zu4g2KhRxxZ8s5wVe352XBlfdpdqKrJP++OZl0BXnCri17QKRMBCz4/+MEPyhSucKngBr/CxQzxRiAw8TsyuTzPj5Fv6wtmk+7GdEW+LGcp1yc/+clpn332SbfcckspIhUtB5zgBbeiBV2wcPCBE7wWLEpR7EcJAyd4wW1UnohPCT5wgldTeJT8ee5pdYYTvOA2bdnI3y0ELCQ6mtstqUdLa5HAYsHoHPWlwA1+9VFsl5J5eezITIl5m7se0+zGZDVKly/LWcr1BS94QXJ8bB55Blem56FVclk4watkGUuRDU7wKkWekuWAE7xKlrEU2eAErybkcQzRDlv+UMiiPu9el25wgldd9BZEJ9hOgID3UX1cZ4KskWUAAbjBbyC6U4+r7MzsM0tNAayNXZlpd2OyLqXLl+Us5XrKKaek9773vaWIU7QccIJX0UIWIhyc4FWIOEWLASd4FS1kIcLBCV5NipPf6fMelpflffXL+3dtfd69Lt3gBK+66AWdchE45JBD0te//vXGBVy2o5cAgxv83Hc1mPfuuSvTVU2ml3smRwabNnY9ZtmNIZtQunxkLCUcccQR6ZhjjkmXX355KSIVKQd84ASvIgUsTCg4wQtuhYlWlDjwgRO8ihKsUGHgBC+4NSFiqZ/nnlZX+MAJXtOWjfzdQ+DRj350+vKXv9y44Mt09DKDBTf45ee4dguBmR0Z3l+TuzKz7sZk+EuXL8tZyvXMM89MvhI1jzzLXhY+cFp2PevUD15wq5PmstGCD5yWTa8m9YEX3Orm4cMQvr64yM+716UTfOBUF72gUzYCj3/845OdwyalXLajlxkruMEvP3fxas4bOzIz1FyTux7z7MZkVUqXL8tZwtVXbbzsdvXVV5cgTnEywAU+cCpOuIIFghfc4FewmAsT7eqrr07wgdPChOggY3jBrYl2VernuaepJrjAB07TlIu83UWAA+7rjm1osCxHLzNWcINffu7ilSPjK51dlH1emWfekcEYcE3sysy7G0M2oXT5yFhSePWrX53a+kpISXpPIgtc4DNJ3sizJwJwg9+esfEEAbjAx32E6RCAG/ymK7Vxbp/S9sK/T2cL/Z/Pzmn5E9uZms9x+yy3/K6epQ1+BAWtXNa9/K7y1hngAp86aZZFK6QZRMBvqtlZ8A7LYFqdz8ty9DJjAi+4wS/HxbVbCMzlyFC1iV2POnZjyCaULh8ZSwknnHBC8jLrW97yllJEKkIOeMAFPkUI1DEh4AY/OHZM9EbFhQdc4NMooyUlDjf4wXFJVZxJLXjABT4zEYhCnUSgqqrkd7l8nKIpBZbp6GXGCF5wq6oqR3Xyethhh2386/6d1Gxjoed2ZDZv3pzq3JWpazcmPfCvdPkeELOYi98HuuCCC1r5+kkxSo8RxNdM4AGXMdkiaQME4AdHeG6QdSWS4QAPuKyEwg0pCT84wrMhFp0iCwd4wKVTgoewtSDwrGc9K91www210BpFZBmOXvbrBi+49cd18d5cN96RmaPm6tz1qHM3JqtUunxZzhKuzomed955SahBns6TgIMAl84rs0AF4AdHYYFiFMMaDgJcihGqg4LAD45CB8WvXWQ4CHCpnXgQLB4B70R9+MMfblTOfLzSsUih/2hkTsvHJ7Mgjlo6cim/q2dpizp6iXcO8IJbfo5r9xCYe0eGyjzBOnZl6t6NIZtQunxkLCls3bo1fe1rX0uXXXZZSWK1Lgv94QCP1pkvIUM4whOuS6jexCrRHw7wGF4oYqdBAI7whOs05ZYtL/3hAI9l0y30mQyBJz/5yWmfffZJt9xyy2QFVjwXnOAFt2WAInZk5qzFOnY9mtiNyWqVLl+Ws5TrO97xjnTWWWelT33qU6WI1Koc9KY/HFplvOTM4AlX+C65qkPVozf94TA0Q0TOhAA84QrfmQh0vBC96Q+Hjqsyu/hRsofAC17wgnTttdf27uPPeATgBK/xubqRasG+G5LWL+U+dZEE4jy7Mk3txmT9Spcvy1nK1QrFO9/5zvTSl740+apHKXK1IQd96U1/OLTBc1V4wBOu8IXzquhNT/rSm/5wEBehHgTgCVf4wrkeqt2gQl960x8O3ZA6pGwKgVNOOSW9973vbYr8UtGFE7yWSqkplVmG7LU5MsCYZ9ejyd0Ysgmly0fGksLLXvay5OzoqnV0+tKb/iXVx7LIAlf4wnlZdJpED/rSm/6T5I880yEAV/jCebqS3c5NX3rTv9uahPR1IPA//sf/SIceemi6/PLL6yC3tDTgc8wxx6QjjjhiaXVcFcVqdWRm3fVoejcmV2bp8mU5S7r+wR/8QXrYwx6W1FE9cpVNhZ70pXfZknZbOvjCGd7d1mQy6elJX3pPViJyzYIAfOEM71nKd60MPelL767JHvLWi4D3I4477rge0QsvvDBdeumlvfv4MxwB+Jx55pnDEyO2UwjU6sjQfJZdjzZ2Y8gmlC4fGUsLf/Inf5K+853vpGXv9PSjJ31Lq4NllAfO8D711FOXUb1dOtXSrnZRi5uNENCuvvSlL6VoVxshFenLgkB2Yhzv/+hHP9o7SbFp06Z09dVXL4uKteoBF/jYyayVcBBbCAK1OzLT7npYUdqyZUtSrg0E8FlbW0uTfme/bflSof8++MEPprvvvntpd2bUM/3oWWgVLJVYBl598C//8i+Tz3HCf6kUfEAZekW7egCMFi//4T/8h3TnnXeGvWoR8xJZrYJM27dvT4cffnh6z3vekyzUZp1f/epXpze96U35Ma59CMAFPn1Rnb/dvHlzMq52XpEZFKjdkSGDzrR9Z+eaBNQ2d2PIJpQuHxlLDCb5P/rRj9KJJ564NB8A8KIsfehFvxJxXyaZ2IRf/dVfTT//8z+ftm7d2ls5hDv81YP6WAZ96UEfetFvGXTqig7GFLJ+7GMfS/BXD+pDXNcDPehDr2hXXa/N+eW3GGSxxC6MBdp+iieccEJ61KMeld7ylrf0R6/8PTzgAp+VB2M4AJ2LbcSR4RnqVDrZOER0wC1btrS2G5NlKV2+LGeJV8c2fu7nfi49/elP7/ynmX2ylB70oVeJeC+DTJwXtqCqqvT4xz8+XX/99T1H2I+nPfWpT+2pCH/1oD7USy+yo3/ITw/60KujanRWbOOKxSoKwF89qA/1Iq6rgfz0oA+9uqpHyF0PAto5p/32229Pa2trQ4myuxdccEH6+te/PjR91SLhAA+4rJruy6xvI44MwAwkG+3K6ITyyd92wLdk+drGYxp+Xiw9++yz01Oe8pR6fjRzGuY15fXjceSnB31qIhtkBhAw2Dr2YPdF0r333uvSC1aVjz322N69P+pBfagX9SOua4Hc5KcHfbomf9fl9bKzttY/sVMP6kO9qJ8u6khu8tODPl3UIWSuBwELQ9q5KydmHNUnPvGJ6bzzzuuFcflWJS1jAZdV0XkV9GzMkdlo18MEZxG7MblSS5cvy1nq1ac+rRBeddVV6eSTT+7Mio8VGfKSm/z0KBXjZZDLuW2DrWt+sdJODN04Nf0TTnHqQ72oH/WkvsSXHshJXnKTnx5Nyxz090QgL0xZpNozJSX1oV7Uj3pSX4N5SnwmJ3nJTX56lChnyNQOApwXTgy76TjZJFw59l/72tc6u+g4iY6T5LEYAAd4TJI/8nQHgcYcGRAYUPLg4rk/LHI3JstRunxZzlKvfnzt4x//eHLUQXD2tFRZyUU+cgrkJr/4CM0iYNHAooVjZAbgzC07Nvk5X9WL+lFPgnrLaSVeyUdOgdzkL1HOZZfJ4hiHeZSe6kX9qCdBvY3KW0I8+cgpkJv8JcjVcRk6K765lN1tbdzcZRpF3vGOd6Szzjqr88fBp9G5P69FAPrDoT9+2e6NtZzdZdNrI30adWSAauIyeB7RgGNiI30jAZtMx79k+ZrUvU7aVjgMtDfeeGN6whOeUNwnH31qkVzkIyd569Q/aG2MgEHY4oVVxMc85jFp3333Tf/tv/23sQXVk/pSb+pPPY4t0HIiechFPnKSt2URgt0DCBhjsj1/IGrkRT2pL/Wm/tTjyMwLSCAPuchHTvIuQIxgWRACbKd5E/tpzjKtaJzgd77znemlL31p7/3Eact3Ob8PZNCb/nDosi6Lk71szo06MlS3cmAS0+8l6pTipS86kKNk+RaNz6T8nTn1FZ3f+73fS+9617vSk570pHT55ZdPWryRfPiTgzzkIh85G2EWREcioH85DmElUaa3vvWtva9JTTIgqy/1pv7Uo/pUr+gsKuBPDvKQi3zkXJQ8q87X2GKyn9vXJHioL/Wm/tSj+lSvk5RtKg/+5CAPuchHzqb4Bd1uIMCB4ajP6sRkLR1LtAt+yimn5KiVuNKX3vRfCYVXUMnGHZm8SqYjwlenLGE3hixC6fKRsUvBJw0ZXAPxddddlw466KB0zjnnpFtuuWUqNWbNjA9++OJPDvKQa1aaUW52BEwyOTHqIDsu+twhhxwy1dcK1R8a6lO9ql/1rL5nl27ykvjghy/+5CAPuSanEjmbQMCYwonRrqalr/7Uo/pUr+pXPavvaWnNkh8f/PDFnxzkIdcs9KLM8iCQbaer9wxnad+DaPhQxMMe9rCl/X2lQX3ZBvrSezAtnpcHgcYdGVD173qUtBtDNqF0+cjYtWAFxIrizTffnBgSBsVZb4O2Adt2bx06oYMeuujjgx+++JOjDj5BY3oEDMCHH354MjHLTgwqBuSbbrrJ7dRBfapX9auen//85/fe0VL/2oH2MDXRIQXQQQ/dDrWrIZosd5TxhIYWx1xnDYPtih1R7+pfO9AeZqXdXw4d9NBFHx/tWHvWrsnRnz/uVxMBtlPbYDfZzzpR8Onuf/zHf0ynnnpqnWSLo3XmmWem73znO4m+xQkXAtWKQCuOjImLDmlSY8DxXKsWcxIjT8nyzaneQosfccQRybGPL3zhC+maa67p7dA4WnTggQf2jp8x1hdddFH6wAc+0HsR0Vd6DPY//vGPe3K7ehbvhT355FfOMQx00LOiiT4++OHbIxB/FoaAOrJSrm8NCqHPDcZN8/y+970vXXLJJenZz352tKtpgFuyvHb6LUTVpRa7wX6wI+wJu8K+sDPsjTbN/rBD7BG7xD6xU2Rw9SxeunzyK6c8Ouihiz4++OGrfIRFIFAWT0dxzZVe8pKXpLraNsdICf0tzwAAEABJREFUX+FAa4f/9//+3/Rv//ZvS7szo7/dfffdyeJAWbXbrDTGVXXdLJfyqLfiyFA7d8h8FVdSyHLla0myLYssXrR77Wtfm2644Ybk07tevjvmmGMSg3PllVcmKyh+8O2Rj3xk+omf+Im0zz779K6exUuXT37llEcHPXTRXxasuq6H42SMqoWLunRhoA3GVVWl3//93++9tOp3aNS7+tcOtAftQvvQTrQX7Ub70Y60q6qqol3VVSkLpGOysra2NvLHAOcVbdp2FfZqXsSjvB1G7douzKy2k53kDLGVT3va01JVVennf/7n09atW9Mf/uEfps9+9rPJApNJvt/yOvHEE3u2dBnQt4hAH3rRbxl0KlaHggTbpy1ZTGp0Tte2eE7Dh1wlyzeNLl3IW1VVOuqoo3q/73DxxRf3Vk6sYFrJ/P73v5+sbObgWbx0xkl+L+4pX1VVF9RdKRk5MRQ2WLrOG+64445kULZKaTBGj8Piura25rIrVNXG7UpmbSvaFSS6G0z62lp4qqrx7eqP//iP09FHH92zW9GuutumFik5B4adMw8ZtGvTyrVjx47EVn7iE5/oFTXB79088Cc7SY5dOeJoocf4+kByJy/kpwd96NVJJULomRBozZEh3bydE40mQ+nyNal7ibSrag8npUQRQ6YBBLZt29aLMRj3bub8Y3Dvd2D6yXFmLED0x01yrwznaJK8kadMBLQLkzF1uWgJOVQvf/nLe7vMi5Yl+HcTAYs/bFIdL/XrExx8tF75ylemTZs29UJGRr/J965ehD/77LPTU57ylM7+aKYfuyQ/PehDrwirg0CrjszqwBqaBgKrh4AJnVCXEwPBPCDb3ckvQj/iEY+QlPJz72GKPwZ6k4YpinQ463KKrp1pG4vWjkOVnZhvfOMbixYn+HcMAXaIE8Mm1Wk3wYCmz3dv2unIeM5hWL9xwsGOxlVXXZVOPvnk5AREzl/ylZzPec5zErnJT4+S5Q3ZmkEgHJlmcA2qgcBKIbB9+/be8S8OR52KG4wFq4gHH3xw730IOzF4eHaNsFoI2PXTHrSLRWrOibn22mt37cR861vfSiami5QpeNeEQAtk2Ey7zV7qr9tuEl/71Fc4SLmvnHTSSSnfy9MfvBP28Y9/PDmaJbzlLW/pTy7unnzk/Nu//dt0xhlnJPIXJ2QI1AoC4ci0AnMwCQSWFwEDslVFg3FTxzPxEAzKj3nMY9K+++6bvOi/vKiGZqMQ2Lp1a21fcxrFY6N4k8R+J0Z+K9/hyEAiwkYI2FHUhtgzTvlG+adJ1wbZY1fHyzgu++23X4+EL+j1bsb80b84NDfeeGN6whOekK6++uoxudtPIg+5yEdOGHLY2pckOA4isKjncGQWhXzwDQSWAAGDpUHTYNKUE5N5cJRA5vO1vkozKz8DO5poRegWAiYsJn7qcFGSm4CaiOJ/8M5dwjxJ9BwhENgIAW1YYM9mtWGjeFjsscuDLpuc8zmGO02/cSTNh3X8QOu73vWu3k8lXH755ZncQq74+3Q0echFPnKyBfSF6UIEK4ipcQ0eBYnUiijhyLQCczBpBoGgumgETOqaGJD79cLDoGywEu8qzGqwlbvzzjuRitAxBKwWDzvj35Ya2YEhxx/90R/1PhnPmbEb893vfjf5WlRbsgSf7iFg0YezYaeEDatTgyuuuKL3uzBs5WAfsXvtCNu0/E444YSEHsfBD7n6/SO/RXPLLbdMS2qm/Pjghy/+5CAPufoJ0pf+/XFxvzoIhCOzOnUdmgYCtSJgUDYYW+mrlXAfMTw84uOaA+cp38d1SgQ6mt1ERVvjiC5KBfy1PRMn93fccUfvmJtdQjJdf/31LhECgT0Q0E7YMm3XRHyPxBoeLPbYkdA2B20l8vgOi5c2SbCjYwfk5ptvTg972MN6DpP3UzgZHIzBzztPQnNYHnTQQxd9euGHL/7kGFYu6yf/sPSIW24EwpFZ7voN7QKBRhAwKCNsQufaRDAwozts4DdwSZslHHbYYfFS9izALbiM9tBke5tFve3bt/c+QMGpMYmMY2azoNiNMrNKqY3k417ayKx0hpXLDpLruF2eeexlP98jjjii9/s0X/jCF9I111yT7JRw4g888MDe8TOOhPdwPvCBDyRfEfNVMc6J3+1Cx9WzeOnyya+cY2PooIcu+vhs3bo14av8uMA2wBoW4/Itcxrd66rrLuEUjkyXaitkDQQKQMCgQ4xhDob4OoIByQBW98BPNoaewXcfoRsI2I2xoqzuSpJYO8oycWZMpkqSL2RZLALsmEUftrLutqHtZQcJ/bY19ZWw1772temGG27ofbnvne98ZzrmmGPS3Xffna688sresUs/UPnIRz4y/cRP/ETaZ599elfP4s8888xePvmVU94XKdFDF/1pdNIPYWzBY5pykbdRBFohHo5MKzAHk0BgORAwoTQ4Nz1wcpbwMDgtB3KhxTwImJzMcsZ/Hp4bldUXOC/9+Thb/c9xv7oIaLPZjtXdLthgTgwbafK+aJSrqkpHHXVU8jsuF198cXIMzI6LnZfvf//7yU5MDp7FS5dPfuWUr6pqLlXgDBthLkJRuFMIhCPTqeoKYTdEIDI0hoDBweBs8GyMyU7CVjBNEA1KOx/j/4ojwGHQFoSSoNixY0fyEnVJMoUsZSDAhrGX4457zSrptm3beu+osMOl9YlxOlXVfE7KONo5zcIXxw5GOS6uy49AODLLX8ehYSAwNwIGZYOzo14Gi7kJjiCQByCD0Ygsc0eT37GMuQktEYGSVSnVYdAnujSRLLmOl0U2doWdZGM4GnXqlWlrd004SHXKukha+iSs4LRIOdrmTWftrm2+JfALR6aEWggZAoGCEWAgDc4GZoNEU6IaeKy+49MUD3QZezoJniOUjYA2YYeuJCnJpP1oSyXJFbK0isAezLQHx73YSAs+eyTO+YA2G4x20/ZxTlEXXlyftBCWF8UWLlAI0DgC4cg0DnEwCAS6i0AeQA2eBtGmNMl86p4AjJKXLniOSo/4MhDgMJTmxGRkSpUryxfX9hCwCMOJYSdNouvknGmzjXXTrlPOkmix7+SBnWuEkhCoX5ZwZOrHNCgGAkuDgJdVTdjywNCUYvhs3bq19ynbpnj006WPI0v9cXFfHgK+flTieyjaTolylVeDyy+RlX/2ixPDrtSpcZO065SzNFp5V0a9lCZbU/JYmKN3U/RLphuOTMm1E7LVgkAQmQ0BRxmUbHoVMA82TfOhSw4modu3b8+PcS0UAXXEkS5NPHLVPWktTceQZ2ME2EhtoYl3VpqkvbFm3c+hf5rY29XtvjahwTgEwpEZh06kBQIrioBBlOpWGV2bCiYBQtN8BuU3wFnBGoyP510ILPzGBKREJwYw2o425D7C6iGg/puykZm29tW2XVy2mrQ4Zldr2fQKffZEIByZPfGIp0Bg5RHIhr/pQTQP2M5+tw26SQKenCjXCOUhUOqxMu02t5/yUAuJmkZA/Xsfxor/3jZyPu7sEdp+M2kRdnE+6csrrY70VYsi5UlXr0TaJV3rpdoNauHIdKOeQspAoBUEDKSMft0D9DDhHSnb2uJ7MYMyrKrRH8Sh1GdtscQdGXKZIJWKW8jVHALqnqPBPlrtr5OTBSQ2Ee0S232durZJi0MI2zZ5Bq8pEZgzezgycwIYxQOBZUHAIO24BMPftE6cJTzqngygOWkwGfXS9qT5I197CGgfpU7m7rzzzhROcHttoRROJsPZ0WA76pSL3WV/m3jXpk45u0hLX1Vf6q6L8ofMGyMQjszGGEWO5UMgNBpAwLa0wdRqIKM/kFzrI14GlUU6MRSKF/6hUGZwrMzxmhKl034PO+ywEkULmRpCgG3kXNftaGhLaJtss70Nib/yZI01HEV4rzwYSwhAODJLWKmhUiAwDQKMez4u0bQTQy5OjF2fNnjhNyqYPNB9VHrEDyLQ3rNJx6LbxyhtS5ZtlMwRPxsC7ANHQ2lOjGtdQTtid7Vz9rAuukFnbwTYejjbVds7dTliVnmnOByZ5WjDoUUgMDMCbToWeSAp4diQwU0woZgZvChYOwJWvktoH6MUM7nVbkalR/xyIKCeOTEmwHPtlgyBQxtnd9G1WzAkS0TVjACc2Xr1WjPpILdgBMKRWXAFBPtAYJEIGKhNytqYOBpEvNxf2upjvCezyBbYLd7asIltt6QOaadFQD3bLWGrTICnLT8uPwfGgg4nJtrSOKTqTTPOqUvY10s5qDWBwDQ0w5GZBq3IGwgsEQKcGOoYrF2bDgYQg7cBpWlek9I3sJm0TJo/8jWPAMfS+0vNc5qeg9Xcktrv9BpEiY0QYKc4G2xVnY6GtsPmujqmFu1oo5qoP119wn8ZbT69VvXdvXBk6u8rQbGTCKyW0AZrGhusXZsOmZ+BpGle09A3mTCoCdOUi7zNIaAuSmsnWdtVPoeeMVjmKwfGsS+ORp1tUJu2w4NmWzZ3metpVt3Yex8RyePRrHSiXFkIhCNTVn2ENDUjYJXCwMRwuRpQambROXJwENoaUGHuSFlb/KapEAMb2ewCTFMu8j6AQAMXfVa9NEB6bpJkW9VVz7nBK5iAeu3fLalTVLaWg8T+2QGuk3bQmh4BzqRSxiXXZQnacKl2s2mMw5FpGuGgvxAEGCkrYAanPEl15dCId5VnIcItkCmd6d7WcTKq5kHcfYnBCp3JRomyrZpM2ueqDsarVtel6GsCaJwwweVs1CkX28feoot+ph3XxSHAvrD56mVxUtTPWTumW/2Uy6cYjkz5dRQSToGAzmzwEEzWHRFwtRLmakARkGTIBPerEEwSDdhwaGtQVQ94CaVizPgL8ClVxlWRS/8tua1oIyXLtyrtpC491aeFLTbRGFEXXe2YrXU1BrEvddEOOvMjkPuw+p+fWhkUtLUlb2cjgQ5HZiQ0kdA1BBglg5LObPDIxmpQD+kGLYOXlfhVcGYYOQMrJ24ULoM4zfusPgQ4z0ur6fJW6PwIY9N8gv54BOyalvqi/3jJI7VrCLD9FlrqtolsrXGInUW7a7isgrx5DqD+V0HfZdcxHJllr+EV0c+EmVEycHBSJlGbMZNfXgOPAcj9rrBEN7DhUBhc21Ir82yL3zx84KINzUMjyi4/AmwEu7H8mi63hmyTBSz2X9+vS1s2xFiC7qTjUF28g850CKh3fZlDO13J8nKvul0KR6a8NhkSTYmAwcNuwywTdYbMgON3VNBgEKZkX3x2ejHadGxL2MwT37Z4zsNHOxC0pXnoRNmU5sEA/qW2GbZBG5lHvyi7WATUIdvkate+zvrkGHGQODGltuHFol8ed2O/eitPsukk0p7rbMvTcV987nBkFl8HIcEcCOjABqZ5Bw8GzUQfrTnEKa5o1od+bQlnMqpeOJZt8ayDD4yWYVCrA4tF0dBuSh2QS5ZtUfXVJb7qj6PByTBe1CU7uuwsu8c5Qn8G2lFkAQioK/am67sy2iA9FgBhESzDkRy5ZzcAABAASURBVCmiGkKIWREwMJkwM0iz0sjlTGTRQTPHdfma9ahz0J4ED86AOpkkb0l5DAQmI0JJcoUsZSCw6pOFMmphNin0aUe+vAvHzs9GZe9S2gQnxrjRtp3dW5qImQUBY5Uxa5ayUWbRCKzzD0dmHYf420EE8gBiJ6Uu8Q1yBr2ur9CQnx5tD674qgsDu2uXAkdGW/LCeZfkDlnbQSB+DLMdnOvmwiZZ1GEL9e+66LOvnCMTYeNGXXSDTrsIsPvGK22kXc71cVt12xSOTH1tKSi1iIBBxGpY3QMIo2bAs0KDx6BKXXgmN/kdc2hbXoOBgb1tvnXx055MfOqiF3SWBwH2Jn4Ms1v1yQ4KbLrJal3So8nW1U23LvmCznQIsPvGTX18upJl5Cb3KtumcGTKaIchxZQIGEiamjBzZhg2A9WUYi08O2Nsp6opbMYpiO/WrVsT/MblKzmN7CY82lfJcnZMthB3QgTuuOOOxJHW/lz15wmLRrYBBNgj+FnQ0a8Hkmd+7KfLVsxMKAoWg4D2oS71u2KEWqAgXbND4cgssLEE69kQMMAryfC4NhG2bNnSm5B3ybAxPgbZRawSmjDgzwFsoj7apEkHbYw+bfINXquLgP7jmJL+m482urI/4l3lWV2EJtdcv4WjySlbOHnJ8Tmbojuaa6S0iQC7r48JbfKtg5e2qb3PS4vu7I3+w/6g58r+iHeVR3xJIRyZkmojZJkIATslbew44NGVCS1DxvgYuNfW1ibCsc5MDBy86qS5KFoGBI4snRYlQ/AtDwF9TNuoUzI02TNB/7F74GpS5ao/C3hqj4L7CMMRMMky4fJSP/yG55o+tim600sSJZpCQN/W77r4w8jsCPlnxUZ5NkjQb7pmh/ZwZGYFIcoFAm0hYLLe1vElhqErE1oGiKyLcGI4e+p/EbzxbSKYCJm8CE3QD5rdQ8BgzybUJbm2ZdKNponDqP4j3QTLBENfC2dmeA3Ahh3k+LGFw3NNHwvvJuhOL0mUaBoBfVC/FJrmVSd9tmlWenTtuh0KR2bW2o9yrSOgw+mwBvW2mE8woW1LlJF8OHcS28QFvxwM8ovinWWo+5onjyYxddMOeoEAW6bfmHRP2ne0SfmhZ+LBFrqPkJJ+KnD2TEbrwoRtVVfjHM26eAWdxSOgj+mP2tLipZlOArJPVyIlbXsZ7FA4MtPWfORfGAKMCyPTpgCMA554t8l3Ul4GWnnzBMd9m4ERtPq5trbWJttWeGWdGPtWGK4Uk9VVVnvSb2eZdGd7pM+hEc5MSnCAaZ3OBlzRhfeibOvq9pDFat41u6+taqfToqbPaOPLYIfCkZm29iP/QhDQWXU8A3jbAmSeji60zXscv+xcLWqgVR8wYQjHydnVNIMDJ5az1lUduii3vl6i3OTSJuaRDQ2TB302T5hmoaddsktozVJ+GcpkLNUJPOvSiV2z46V+irNtdSkZdEYioD05idEVu68fkHmkQkMSlGE79BvtfEiWiaJKsUPhyExUXZFp0QiYtHs3ZlFy6LBkWBT/Qb4GW04EQzSY1tYzPJZ9oGfkDRKwbgvXVeYDb4PssmJgcqTP0HNeHdkkdNCcl1bXyrN/TTgbbBo82VX4dg2XkLceBCwSsPvaWT0Um6PCXpJ1Gg7a+DLZoY0cmWmwibyBQGMImEhaJWmMwQaETRgYC3JskLXxZMbVagpD1DizEQwyDgz+iCxLEw1nE5ylUahgRfQzn/ssTcRZJguDOuiz9Kuzz5hsswe5Pw7yXMZn+sKybmfD5A6O6KqnZcQudJocAX1Lm5i8RDdy6jva9zLZoXBkutH2VlpKk0idjiOxSCAYNrJsLENzOUyoGKJFD7YMPDya07Qcytodw7/oui8HkeYkOfbYY5OJanMcFkOZTvpu3X1G22QLtE08FqNde1zpyfbQeW1trRbG6oVNdfWeDUxrIRxEOo2A9qUtcG5LVuTOO+/s/ebdJDKyEdr5stmhcGQmqf3Is1AEGJK6O94sCmXDxhjMUn7eMgyQ4xR1DuKzyGQywbGExyzlu1hG+9MOF1X3XcRsapl3FjBx0M533hb1n0xkm1UofcbO3qzlx5Ujl/Zpgj8uX9fTOBv6H2ejLtuDHpuKHrvadYxC/noR0Gf13Xqp1kuNbTrssMMmIkoXOk2UecpMi7RD4chMWVmRvV0ETB4NMjpJu5yHczNhYAyGpzYba6LCCMGjWU7jqXtXCQ7jcy1XqvZHZ3WwXJqFNk0jwIbh0WS/tbCgjS7KNtGvqWCixomhX53OhnrRn9HUt5uSv2m6Qb85BLQ5YRn6lfYOqWW0Q+HIqNkIxSLAgCzy3ZhBYBg1cVbyXHMYfM7xdV3zQG7CUhfNWeioDzJkHGah0dUy9BZMfrqqQ+lya1dC0/1pWhxMpsk1bTn5tRcLEO6bDHiYrJC1ST5t0qZL3jGhX1281QlbxolpcmJXl7xBZ3EIaHf61eIkGM9ZH5nENmnzdBlPbf5UPOBFrvmpTUZhBkdmMsKRKxCYFwGdAY2SBhoGg2NlECSbQM5LLrnEbSOBE4MwA+G6yLCKuzH9eKt7Brq//vvT435+BPQxGM9PafEU9F19hk5NS4MHR3tZ2iZnlhPD2ahrx0S7UieujqjBrOl6CfrdRkAbMQfhCJSoibZMxnGyafPLbIfCkRlX+5G2UAR8vaiuAaxORRg1xsNAKzBwD33oQ8ezmDE1T0oM5jOSqK0YWUyUNjKatTEskBDdOZTqXShQxM6LBGMvsHZdEe2DnWjThnG08RW6jB9bw66ye+xtHbqoC44ReujWQTNorAYC+rA+pQ2VpjGZ2MxRcmW56TAqT93xbduhcGTqrsGgVxsCdjoMOrURnIMQY5EDo8EoGGytdCArzrXOQH+hlEHXig6969Sxi7TUNRxMtLSJLupQssy+XKbdk7HLgX3QTtrUIbdNvNvkWycvNtXky45JXfZfe+LEsKVt10md2AStxSCgX1nEK61f6SdkG4cKmdtu82TCE+9xstWVFo5MXUgGnVoRMPAwHDpErYTnIGaANRhWVZVMYhkR5OzGTPrVEPknCWgzAlb/J8nfdB6ylFYfTes8jr4JFjy0g3H5Im16BGCr3+sD05cuowQHl/zaSNsSZZ5saNu85+EHMzYWDQ6Hax2B7RLQ1LbqoFk4jRCvAQTa3mWYRAV9Zlyblr4KdigcmUlaS+RpHQHHyqzMts54BEMTKwPhSSedlPbbb78RueqJZngM6JyYcUaqHm6TUYndmL1xMrCJNUlyjVAfArDtMq5k12fqQ2Q6Sm2uhk4n2fDcJlwWidg7dnZ4ruli0WRH2dM6d3emkyJyLwsC5gD61ZVXXtkZlVbFDtXjyHSmWkPQriBgNXHLli1FicuQXXTRRenMM89M++677y7Z3EvbFTHHTR58DeZra2tzUKqvKGOoLurSsT7JFksJHpxNbdVkabHSLBd3bV9fELqomTbBGVuU7PDTPsmxKBkm5avvcGLYPBPFScuNy6fdcGLggO64vJEWCEyKgPakvQqTlmky344dO9K4BV/9fxXsUDgyTbayoD0TAjqfifNMhRsuZHLw1re+NZ1//vnJkTLs7r333ol/WVf+HAavBl9HlUyOGczB9EU9W1mua4KxKB2a4qs9mCipN/XXFJ9VowtXfaCU1U8fHyDTJPWwbdu2xH5t3rx5kuyN5dFnydIYgxoIk0/f0YfUdw0kk0kmx4gdhUEdNINGIAABfVqb0m49LzqMG3PIuCp2KByZRbfE4L8XAiYvi1xF2EugIRGM2dvf/va0adOmxJEZkmXqKAP62tpaYnymLtxQgVKMYUPq1UJ28+bNSXuwAjxuYKmF2QoRgalFjQGVi38kM9kXLShbom2a2C9almH89RdY1Xnsi71iR+t0jIbJHnGri4B+RfsS+pXxJstDpv6gb62KHQpHpr/m437hCOiYDMSozjmPgPfff/88xfcqy+H4zGc+kw455JC90qaNMKgrU4LhIUcOsRuTkRh/1RYE9agNj88dqZMgYBIusAeT5C8hj8kD20XuEuRhT0zuS5Aly6B/6CeeOTGudQQ0tRU01UEdNJeHRmhSFwL6tn7FYa6L5qx09CXyDJZfNTsUjsxgC4jnhSKgYzoSMK0QnJRbb701vetd70rnnntueu5zn5t+6Zd+KR166KFp//33Tw960IN2Bc/ipcsnv3LKozMNb0bkpptuSq7TlOvPmw2iVcT++EXfM4ZbtmyZS7dF69Amf4MbvEyotOM2eS8rLzuzpU3Ex2FNVjKPy9NmWrZLJvj9fAef+9OavNcv9I+1nTvPddm7TJOuddFsEoOg3X0EtF/tbVH9CILaPRncD4ZVs0P7DAJQ13PQCQRmQYCBMBmcpOynP/3pdOGFF6bjjz++9yWxM844I918883p4IMPTiYTl112Wfr4xz+e7rrrrvTDH/4w3Xfffb2rZ/HS5ZNfOeX322+/Hj100Z9EjlHGZJKyDA5jWOIATDaT80n0iDzrCMBL+zVZM9Csx8bfWRFgD+Coj8xKo61yHH+8yOxaQmCb2Dh9OctDzksuuSQ/tnZVh3W/u5Jp0nGWBbDWlA9GS4cAW58XIRehnLY/zNbo3+QZliZ+EaFpOxSOzCJqNXjOjMBtt92WHHc68sgj0ymnnJK++c1vpnPOOSfdc889yTEvg9l5552Xnve85+3akfFSvh0ZTF095x0Z+eRXTnl00EMXfXzww1f5OgNDxOg4CvEA3WIu5GIIGaBihOqIIAa4cGbqqSztD5bem6uHYnNUfEFI3TfHYTbK+nF2Btmc0047bdeHSmajOH0pjhS+FmzIMz2FvUv009RG9s4RMYFAcwhox+yTsbI5LtNTXkU7FI7M9O0kSiwAgeuvv753XOyYY45J3/ve9xLH4+///u+TL4idcMIJtQ3MnBz00EUfH/zwdQyNHHWob0Jx3HHH9fSog17dNK688srerlbddFeFngmtyZU6vuOOO1ZF7Ub0tNoOw9EThkbYTk2UfCY3UxdsoAC8cjDZ0h5N/LVH7MS5thE4MLCxYFMXPvRgQ+uk2QYWwWO5EDA/0K8WodWOHcM/vayv1dXP5tUr2yBXNqcpOxSOzLw1FeUbReC6665LBq3XvOY1iYNhp4STcfTRRzfKNxPHBz988ScHeciV80x71anRqHN1cloZxuU3QRBKMYbjZC05jdEOZ2b+GjIALnLCQAN99rDDDnM7NJg8qGuyDs2wgEg2xlGuqqoSZ0KfJobFmnG6yFNHgBkZXDkcddKEM/tZB82VpRGKz42Adihs27Ztblp1EFhVOxSOTB2tJ2jUjsDnPve53g4Mx+FlL3tZ79jY6aefXjufaQji7/gZechlh4ac09AwqBvcDcKlOgp2Y0zKptEr8g5HIJyZ4bhMG2uyoE2akE9bto38o1ZH2+A9jAe82JiTTjqp9/7gsDxNxmU7x8aRow5eHDGOGZoc2zpoBo1AYF4EtEUOxLxz1gmqAAAQAElEQVR0pi2vP+gL/eVW1Q616cj04x33gcBIBLyT8vSnPz0985nPTJ///OfTqaeeOjLvIhLIQy7ykZO8k8phIrZly5a0trY2aZHW8zGQJuCtM15ShrBU5xxYE7wlVbNxtRwx0zaFxplNycBERh1PWazR7JyZiy66KJ155plp33333cXLvbRdETXfqB8Ohwmetl8HeSvebCenqC6adcgVNAIBfcl4rn22iYaxBO9+nqtqh8KR6W8Fcb9QBHwljGPg3RThVa961ULl2Yg5+cgpkJv848qYyDI80w3E4yjWn8YQMsrkrJ/66lJU5ya62oABaHWRmF1zbdLkuO0JA4nxvvPOO93uFfQZdbtXQgER5HY09vzzz9/1HuG9997b2CfVYaF+OBzsSB0Q6DPoOp5WF8065AoagUBGgH3nwC/Stusjq2qHwpHJLTGuC0XA77j4XZcXv/jF6dprr+39/stCBZqQua+fkZfc5KfHsKIGY/EmYq6lBsfKjj322FLF67RcBjv1ry1YYe60MgsS3kRWMFkeK0KLifqM3aIWWU7NStt7+9vfnjZt2pQ4MlMTmKCAOtGu63JiTAr1Faw5Ma4RAoESEbBgwInQ/tuQj9PCDvbzWmU7FI5Mf0uI+4Ug8Fu/9VvpbW97W/rkJz/ZOwaxECHmZOr4BvnpQZ9+ctm4GeD740u8t6rEIJco2zLIZPDRDgxEuV0sg15t6mBSrp0KbfIdxstkmxzqdVj6PHHT/jjvRrz0a+/4HXLIIRtlnSodBhwOVw6HSd1UBIZkhqnjaXDVX4ZkiagGEAiSsyNgMUO7FWanMnnJ/n6m7+Grv0xOYbKcXbBD4chMVpeRqyEEfv3Xfz05muUHKu1oNMSmFbLkpwd96IUp42LS2oXB2MTaZIfcEZpDwACU24PJmkGoOW7LRxl+drbsACwaO/zJMi3KJge33nprsoN77rnn9j5swn7Y4d1///2T37vKwbN46T4wIr9yyqMzDW/Y3XTTTcl1mnKj8tJfPZhA5TY9Ku+k8ewlmuhxWictF/kCgUUioE9pr8bRpuVwzBW/zEc/XGU7tGBHJldDXFcRAYOyF08/9KEP7Tq/3XUcfNqUPvT65V/+5d6no2cxMIvAwQTCqtIieK8aT4OQQY/jaDW7jcFvmTA2cc7YGcQXpVuWYxL+3qG78MIL0/HHH9/7ktgZZ5yRbr755nTwwQf3frPpsssuSxZC7rrrrvTDH/4w3Xfffb2rZ/HS9U/5lVN+v/3269FDF/1J5ND2Jsm3UR6LNBxxMmnLG+WfJJ0Doy9wYmA7SZnIEwiUgoA2yx7pG03KhEf/J9TxZQ8n4clOsBfLYofoHI4MFCK0jgAnxoDclUn+tADRi6HxQ5qMzLTlx+ZvIJETY4LTBVkbUH9hJE0ATdrgbwK3MEE6yBh2Bm+OYKni33bbbclXDY888sh0yimnJL9Hdc4556R77rmn90l5duK8885Lz3ve85IdFzsvFkPsxtDJ1bN46fLJr5xjYuighy76+OCHr/JNBe31tNNOS9quOpiXj4nZcccdl1zrOp42r0xRPhCYFgFjKLtUty0fdIw8TzNWswfsAvvATrAX7Ab7wY6wJ+wK+8LOsDfsDvsDA1fP4qXLJ79yyqODHrro44Mfvso3HcKRaRrhoL8XAo5dHXDAAckK416JSxRx9dVXp8c+9rGJvqWr5fvzVlZLl3MZ5TP4mRCaxOXJ3DLq2YROJg0m0nYGNqLfZvr111/fOy5mIeN73/teMuA7cuoLYieccEJtO9AmF+ihiz4++OFrsYgcs+rNWbnjjjv2Km6SJmiz00ym9iL0QAQe6g8tNB+Ijksg0EkEtGOCczZc6wjGZ30u09Jn8v24q/7PDrAH7AL7wE6wF+wG+zGu/KRp6KCHLvr44Icv/uSYlNYs+cKRmQW1KDMzAl6E18A19JmJdKggPelL75LFNmkxISxZxmWWjTNjUm4QDGdmupqGm7Zrh2C6kpPntrs6yeThuuuu6x0n9YO5BnYrlAb3o48+enJmc+TEBz988SeH9kSuackqO6gzWiZode2asDucGA6MepxWxsjfOALBYEoEsi2v0x5ZZLTDcdBBByX9j0j4uOfg6JfictDfxenD7AB7wC6wDzlPk1d88MMXf3KQh1xN8A1HpglUg+ZQBLygyjO/5pprhqYvayR96U3/EnU0mTARLFG2VZLJwGQyx/ll9A1Qq6T/PLoa6OFX5+RhGnk+97nP9XZgDNgve9nLesfGTj/99GlI1J4Xf8c+yEMuK6PknIQRm3D33XcnK8Hyc2i0SRhzOsTNG7RvAb21tbV5yUX5QKAYBLRnfUU/qkMotH7u534ufetb3+odQ/XlwaqqeosmHBxjBj76t36uv+v3+j87IG1RAX9ykIdc5CNnnfKU58jUqV3QKgYBL5h5OfXd7353bccqilFuA0Fsu9Kb/nDYIHvrySYr8dsxrcM+kqFB0OTOIGjyaBI5MnMk9BAw0HNmYGVy3Its6Y+JxNOf/vT0zGc+M33+859Pp556akucJ2NDHnKRj5zk3ajk//7f/7uX5YMf/GCy6mvXBL55wtRLnPGPOtKu0bWzs7a2NiOlKBYIlIuARak6bdHJJ5/c+x2o7373u+kb3/hGT3Ef+zBWsH/6tf6tn+vv+n0vUyF/yEMu8pGTvHWJFo5MXUgGnbEIvOIVr0iXXnpp74XWsRmXNNELcvSHQxMqzkPThCImE/MgWH9ZA5MBSr2Y9NU5INYvbRkUYWairT23gZdFCQOyM+HCq171qjKAGCEF+cgpkJv8w7LC77Of/WwvidNhl0tb3LJlSy9unj/oac9ra2u9DwXMQyvKBgIlI6CNs0lXXHFFLWIOLjZyYjg3P/VTP5X0Z/1a0M9rYdgQEfKRUyD3KDs0DftwZKZBK/LOhADP+2d/9mc7+2OXMyk9pJAfzYQDPIYkLyRq+/btvS8FMbgLESCYjkRAnVjVM4mUyaq4+nIfYTgCMOPMmDxM4MwMJzJBrGOiFide/OIXp2uvvTb5ms8ExRaehZzkJTf56TEo1JVXXrlHlBXgSy65JM3b9pTXhtWPdr0Hk3gIBJYQAW29LjvEtumLGaZNmzYlL9Prx/qzfq1/5/SSr+QkL7nJP8wOTSN/ODLToBV5p0bAWciLLrooCVMXXsICcBDgUoJ6VkjrWGktQZdllcEAZuInWB03MKq3ZdV3Xr3g1e/81Y2VD3e87W1vS5/85Cc7uzhjUYX89KBPP+acwPxs4iQ4YmYnhSPCIcnpk161WW1Xvazt3I2ZtFzkKxGBkGlSBNgi7V3bn7TMqHxoPfWpT92VjK7+qx/rz7sSOnRDbvLTY9AOTaNGODLToBV5p0bA5OuNb3xjZ1Ysp1ZwygJWIuABlymLNpJ9x44daXDLuhFGQXRuBDicJoIImVRu27bNbYQhCBj09TGYwap/cp6zc3AmnZSjJ79PqTsS4QcqrSRmWl28kp8e9KEXHeDkyIr7fLXye/7556etW7f2joOZQEmfJMAM/nCO92EmQSzyLBsC7JD2X4dev/d7v9cj84QnPCH927/9W+8HdPXjXmRH/5B/0A5Nq0onHJlplYr8ZSDgU3tf/epXkzORd911V/IJ06qqUlVVvXtxJP3rv/7r9Au/8AspP4ubJsxbvp/XX/3VXyUD9b/+67/2R++6Fy9dPvKSG/9dGSa4gQdc4DNB9kazmLiY7DXKJIjXhsDmzZuTgZFDY3C0Qm6yWBuDJSOUsXJcatDxg58jU5Oq/Ld/+7dp3333TR/60IeW5oMlPkRCH3r5mtAFF1zQe6GY0/I7v/M76f7770/f+c53em0OltrfpHhpl1aildFeJy0X+QKBZUJA+zdn0Bfm1QutRzziEenJT37yUtuhaXEKR2ZaxCL/xAi8+c1vTr/927/dc1BsiZ511lm9gdHg6F4cZ+AXf/EXk5dLH/nIR05Mu6mMz3nOc5IJjgF+GA/x0uUblj5pHFzgM2n+GfJtWCScmA0hKjaDAc3kkBNqxXtwkl6s4AsQDFbOqmvv/Tjdeeed6f3vf3+vv28k1oUXXpie/exn937YcqO8XUyHz0Me8pDEFrPJnBZhVl3YSE52XV86m1WOKBcIlICAvqRPcO7nkeeVr3xl+rVf+7WltkMHH3xwsqgyDU7hyEyDVuSdGAG/m/Ld73639ynSf/qnf+o5MMcff/yu8l7y8vCpT30q2dH4hV9Y35Gx0+FeqKoqGVyly2uAzbs6a2trvV2cnCY9BzSqan3nRz67KDnN1bN4+TwLBtw3velNSZw0edxX1TodfPEXL/3P/uzP0gtf+MLkXRefEyRHTquq9TLKoy3NS/5CpuNThPCBkzyLCI6Vbd65wr8I3sGzHgQMkBwa1Ewc+yfq4iKsI6Cd9+NkQmGHReo555zj8kDY++LY1X/8j/8xXX311XsnLlEM/Q444IBE33nU0gatPsOboz0PrSgbCCwDAuwPW61vzKqPfql/XnbZZbOS6EQ5+tGTvpMKHI7MpEhFvqkQ0Bjtuij0uMc9Lj3qUY/qfXo5T+7tvlgRHbazwTnwHsn3v//9dNRRRyUvgqHjqAMnwo7Or/7qr6YvfelLovcInIbf+I3fSBwk5SV6ds3BroryVmPFcVDw7He0/BicYxbo4EeXq666SvZecH78fe97X3riE5+Ybrzxxt5KZuaDr+MaPo1IHgXuueee5BgLnekuDk04uV9EsEIU78csAvl6eeZB0sTxiiuuSBya7du318tkCahlnEyu7WLl/s+pGYWXF1C/973vLe0K6GC12pmhL70H0yZ5hiss432YSdBaojyhyoYImLuMszXjCOiP+qX+OS7fsqTRk770nkSncGQmQSnyTIXAbbfdlm6++eZ0+umn98pxHAxuJvcnnnhi7x2Zqqp6ux+9DAN/7Fx4AUw5Dofk7GycffbZHpMdncc+9rG9+/4/N9xwQ89p4jwpf+655ybOg92S/nxHHnlkIhO6nJVNmzYlZfrz9N87Cib0x/Xfo48Pfvga0Dlh5JHvwAMPTD/zMz/jdleAD5zgtSuyxRtGdW3nzlaLLINVgwiYqHNmrPxZEQ+HZjjYdl+zfZHDzqjdVff9wSdB7Zhec801/dFLf09fetN/UmXZEjYvt8FJy0W+QGBVENA32J5pd2X0Q/1Rv1wVrOhJX3rT3/O40FVHZpxOkbZgBDTAF73oRXtJYffF7obAebCDkXcs+jObgO2///79UcnxtG9/+9t7xI16cGRK+aqqEsfJ6qBdkv78JjK2L9G1Msth4oDkPM5p/vEf/3FybKyqqmTCz1nJ6YNX9PHJ8Wg5RpafH/7wh+/lyEiDE7zctxms3FuZbpNn8GoeAYOles0OjUFTf+K0N8+9fA4ZD32+vz9zZvSJrIEfaTvjjDPSu9/97qV5sT/rttGV7aI3/eGwUX5tSxszSbOSulH+SA8EVhUB8whOvz4zCQb6n36oDrEi9QAAEABJREFUP+qXk5RZljz0pTf94TBOr3BkxqETaTMh8Kd/+qfJsapc2HEyg1x+drX74bgZR8LzRsFuBmdgo3zS7dZwlnKwU5KPc0kXPDsW9oEPfCDZwuw/ViZd8OKrTxyiw9HheIkfFjhOBvOcZpKEb34edYUTvEal1xu/mxpnL46V7cZj2e5GOTT9k/Vl03mcPiYPdgz8CJt8Bkm7sO6Fe++9N5133nlue+EVr3hFuvTSS3vHYXsRK/bHQg/94TBOdY6h3T+OMwd6XN5ICwRWHQF22Y65ftOPBbvMRvXHudf/9EP90fOqBXrTHw7jdA9HZhw6kTY1Ajzn++67Lx199NG7ymqMViA4NDnSwHfrrbcO3aXIefqv2fHI78t4X2XYOzIcErTzTg8HyioIx6KfnvuTTjop/a//9b+S1ViOlbgcPvOZzyQfHHD0LMc5jpbvB68mRnZgLr744oQXGehHnsG8/c9wghfc+uObvlcfjGrTfIL+4hEwwdQeDaBXXnll7x0aA+fiJWtPAm0dBl/84hd7Hx75m7/5m2QRww4CO6Cv+wSxCcbWrVuT461+rK0n4Yr+oT8c4DEMAo6hdmQnmo0dlifiAoFAYE8Ecl8xBkvhwFgMcN8f9Dv9Tz/sj1+1e/rDAR6jdA9HZhQyET8TAs402r3oL8wJ+cQnPpF41VW1/kUvOxH5Jfn+vOPuf//3fz/p/FVVpW984xtp2DsydlG8VM95qqqql99L+RyNQdry6CDkHUx/0pOe1PsqmZ2gqqrShz/84TS4I2MXZtOmTb3jZxwnfPEQ70ib1V/yiBsX8IfbuDxNpJncNUE3aJaJQDg0u+tF219bW0sw8QVCCxcWLZ773Oemiy66qBd2517du4yFj6FkFEy8ODGeOTGuEQKBQQTieTgCmzdv7v0uU3Zenve85+2VUX/LfW+vxBWMyFjAZZj64cgMQyXiZkbgIx/5SBq2C8GZcdTKMS3Bka08yXf97Gc/m+TxHg1nJTsWXrC3ikwg6ZmG33TwQ20cjf7y8qGBhyC/cuIHg3jpeOQ0ZTN/8WgIOY5c7uXL91mX/Cy/IA+6g/KJ6w/wglt/XNP3JiMMatN8gn55CGzZsiXZnVjlHZpRtQITX0w89NBDR2VZqXg4wAMuFGf7Dj/88LS20wnUhsRFCAQCgekQ0H+Mv0972tPS3//93+/1Hp7+pt/pf9NRXs7ccIAHXIZpuESOzDD1Iq5NBEzeb7rpptTEuxeOa+n8VVX1vnpmR+cv/uIves5Pmzo2wQtecINfE/QHaZqMwHIwPp5XC4FhDo2jVauFwm5tr7vuuvTVr341vepVr9odGXc9PODi93ZOO+20XU5wQBMIBAKTI+AYpgn5QQcd1JvDGIct4Ho/zyKoxUXUwg5BYe/ALrND8BlMDUdmEJF4nhkBXyLzHolOOTOREQXR1PFN9oW8CzIie6ei6QY3+LUhOINpNajHK/6sPALZobHCrm1YcTdh1d9WCZw3v/nNyS7sKuk8qa5wcTxYG4lFkElRi3yBwG4E2Nk/+ZM/ST/60Y+SI+lSODH9V/dhh6AwPLBD8BlMDUdmEJF4nhkB5xe9NDszgRUuCDf4tQGB43Rt8Ake3UKAc+vld5NV93ZnODWu3dJkemm9o+ajH6eeeupEhVctE1xMuhyDWTXdQ99AoC4ELAJ4H49T4+Mig3TDDg0isuczO8ROw6k/JRyZfjTifi4EfBHIzsJcRFa0MNzg14b6Vt0dZ2uDV/DoHgKcGGeROTSC9sKhWeZdmssuuyydddZZ3ausFiWGD5xaZBmslguB0GYnAuzrW9/61nT++ef3dmby7szOpKR/6WfuIwxHAD5w6k8NR6YfjbifC4GvfOUr6TGPecxcNBTO78P0f65Z/KRBOSsf6PSX8WUxOx++TtQf3/Q9OchDrlG84Aa/Uel1xpuYMqZ10gxay4mAdrLsuzS33XZbuvnmm9Ppp5++nJVYk1bwgRO8aiIZZAKBlUXAYpHdmf322y/ZZfAlVv1LP1tZUCZQHD5w6rdDy+3ITABKZJkMAeflNzpioiN6mW0yiqNzeWcEv/zVr9E5lycFbvBrQ6NwZNpAebl4cGgMvHZoODba0LLs0lxzzTXpRS960XJVWEPawAleDZEPsoHASiHArv6///f/kt9IueGGG8IOTVj7g3YoHJkJgVv1bDrcO97xjuSLG75eMwyPf/7nf04//dM/vUfSm970pt4XOqqqSg95yEOSXZG8Q/HEJz6xl+Z3HOxYDD7bwfCDlmhkou7FeZZeVetfMauqKnkWPyzYhXn+85+fvIfy1Kc+NXkmC5mqqkp+zFLcYFl57OL85m/+Zk9W+cXlfOSpqnUZctqgfn6DJud3JWfO61mAG/zcNx1MQtXnKD4RHwiMQ0Bf5cxwarQjR844NRstdIyjuci0P/3TP02+grhIGbrCG07w6oq8IWcgUDoCbKhFok9+8pNhhyasrEE7FI7MhMCtejadzY8Sffe7301XXHHFUIdG2gEHHLALKhP2Sy+9NP3TP/1T8qWxk046Kb3tbW/blc5BEO8HIUUOPotThiPAORDci+NM+IFKX/pCw49lXnzxxUke5QaD34zxuWbOkq/vSOfYvO51r+vJZnL2whe+cGj5L33pS+lf/uVfevnkP/fcc3v5ptUPT2XIyWnx+zLiBLjBz32TIZyYJtFdLdpsggHYjyJybLQtDs1xxx2XODWeS0fk05/+dLrvvvuSH8a1mFFVVW/BoqrWr4MLDmP02ZVkcYM9GWWLdmXsu7E4o5woto0tHLawIn3a0E+v/35aOvIfffTRPbzg5jlCIBAIzI+A/sQO6V+ZGpsg5OcmrmwUW2VeUgf9TK+qqtSk7HCCF9zIHY4MFB4I27dvf+AuLsMQ0OAd+zLh/ta3vrWXQ/Pv//7v6cEPfvCuoo6G+UIWJ0KkF9pdc9joWT4/eOnb4f/wD/+QBLx/6Zd+KXECvva1r/Wu8g3SEjcucK44Dy9+8Yt72c4+++zeb0jg0Yvo+3PggQcm6aLkJ4N80+p3yy23pAsuuCC9733v2+sHsOAGPzyaDNq4emySR9BePQS0Kc6MXRoDGCeGUyNwarS7ElHx9Zu8kEK+D33oQ70FC4sjwjJ95p1+7OZnH/jxYc+zBHjBbZayUSYQ2BOBeIKA/qRfuW8zmM+xzeYydfD9/ve/3yNjfnXllVf27pv6Ay+4od+qI2NAM8BhXGJwRKJk+aqq2mO1sKrafTYpMYnPdeeeQ+PImVVYA/+DHvSgnNzbtTDBqap1OX/nd35nV5qbQedj8Fmexz3ucelRj3pUb1fHGVI7KtkxMmGqqnXaJ554ouwTBx3tO9/5zq78HKaHP/zhu577b8RL749z37/6UFVV2ki/j33sY8kqtl0k5fsD3OAn9MfHfSDQJQS07y1btiROTf9ODfvAfhgDDJyl6PSRj3wkHX/88RuKk3cy1tbWdtng/lVM91W1fnzWmfdRBO26VNW6zcq7PeKuuuqqnv3wJSM7xfkI7P/5P/+nt1uUd4zs0mReVbVOx3PmR050q6pKZP3yl7+cBun17/YoW1XrdORn0wT3/fbVfeYBL7jl57gGAoHAfAjoT/rVpFT6+7mdZHYhl2VPqmq9T7MF8urn+rS+LZ84doB9EC89x3muqvXy4uUXMl00n/vc5+6140IGx/Z37NjR++jTe9/73r1sFx7KV9Wex/nxwRfdqlpPE5fz4k2G/gAvuIlr1ZHhJGzbtg3fhYVRjLMTU7J8W7Zs2WO10KS37dD/qUBYPuIRj+h9RtBKbFVV6cc//rHoXrjkkkt6V146OR3/6kVM8ceKAc/7/e9/f/rCF76QHCtTXCM3IeKQoG0lVfykgWNiRybnR+fb3/52ftzjKl66SFfP7qfVz5E0OzKCTo9GDnCrqnXjkePiGgh0GQFOzdraWs+p0UfZCPqwtSU4NWS66aab0qSfInfElC1Szs6sI6ImBgZnZ7bZIEdGLfDQczCwWcOO2vqRN/TYx9/93d9N/Udg2dd77rknsTV2t9mfUUdq2ZT//t//e7rxxht744QJzp//+Z/vRS/L1S83Gy0ebVcBP/H0Yn/lFw8vuMHBc4RAIBCYHQH9SH/Sryahop9bnLj22mt7/ZyNzcfiR9kYp1jYJSdJ8MiLwgcffLDHXWGcjWODLMJyfnxtbVehB24sMDu2b7GZDXr84x+f+m2XbOR2PJ/O/XJL4wCxX2yO8eEVr3hFwov9YTfpLV8O8IIbWvvkyDauzlObfHJo2uA3DQ/vfRhoS5YPftPoVHdeGPlRNHQNsFu3bk3f/OY3U5brJ3/yJ9MPfvADyXsFjVBj3Cthggiet4FU3eiQg0VMJkwqBuPHPXNk7MhYCZXPuzt2fuwAee4POqOOL851WL5J9XO0Q0fPfNEU4AY/9wWGECkQmBsBjg1bYaeGrUXQwpFBi3Ojf4trKxiU7QJbLMk87exW1fqCQlWt72qwL9IdMWWL3OcFFfeci6OOOirZdULLgoX4weD4BufAgC8Nb9eNAr7slXzsx6gjtfTZtGlTyjbM0Q5OknLDwjC51cHdd9/dy85pow+b+9jHPrYX5484suPnOUIgEAjMjoB+pD/pV5NQkV8+/dLVsXdOCgdglI3Zf//9e79ZY/7CnuV3jZXvD2zNMBunHPvG/rBffsulv9yoe/Sy7WJvLB5btJGf3Pm1Ac92nekEB4sweOClvHLy9Af54AaPVh0ZAxkvzODVL9Ci7w2idjvIJpQqH/wWiZWvlWk8gw5MlskgyjnIzxqsSYtOZMvxNa95TTKQO3ee80xyNTCbKKgbDVsZncoESCP3xS8rorkzSx8W5CUfWaRb+XzDG97QOypiAB/27op8BnEdv6qqxBnL+WbVz8orGnmFEw9ywc99hEBg2RFgyzg1HBrBM7urT7PHFk2axsDxLccr+vlY/bPClwO7wObJM+qIqZ1i6XuHPWNMINiwqlp3lAaPou6Ze/fTIF/HvKpqnQbHK+ecVI5R+dnH/gmDSULOO3iFG/wG4+M5EAgEpkNAP9KfpillMUN/raoqmfzbSeEojLIxbJiFCTbCroy5knKDPAdtTU5XLt9Pc+2nRz7znFye/NLzM9tvrpifJ7nCDX6tOjIEM3gZHEralTFokqsL8pFxEUGdvfKVr9xjB2ZQDg6FoxU5ntPBcTEpcD3zzDMTOvK5Wj2QVyeb5NkKo/xCLoM2xwjt/BIruujJI28OWR6yuLe6oCwaOS7nHbxyPAbzoaFcjicDvhvppxxZ8c984KZcfo5rILAqCHBi2F8OjeDIgL5eVVUyuHFs9Ktp8TDGcI5GlfviF7+Yxk3WR5UbjJ+UhqMZylo5ZTMsaHieJjg6AguTAjQ4Xrn8pHKMyo9m/0Qj5xt2xQt+w9IiLhCYC4EVK6wf6U/TqM1GZjvCDpjHmE+MszF2Wkz6b7311uRUiHnIpDynlW8YXY5L/0IJe8BZ0/wAABAASURBVJOP6Q/LP0kcueDXuiNj0FpbW+t9njMV8M8gaTeGXMRxLVk+Mi4iwMRkYxzvQw45JH39618flyXSRiAAN/iNSI7oQGAlEGB/2WMOjd1cHw0QxyHh1NiJdW8yPwkg73rXu0aONV/5yld6L6VOQmdcHiubZCWzFdFJjrlOehR1HN9BXiYK/Uc17NwIo2jIb1LTLzc7P3huflj5xzzmMQl+w9IiLhAIBCZHQD/SnyYt0W9vlPEivKNYbIrnHDw7QZKfnWxx6uPlL3/5rneNc9pGV04QO+EUySDdjcrmdPbGQkk+Vj/uOH8us9EVbvBr3ZEhmAmxgciKmedFhiuuuGLXOx5ZjtLly3KWdn30ox/dezmrNLnmkccqh92TaVYvZuHnpTb4zVI2ygQCy4gAB8bEmj02iAru6cqZ2cixUT6/y+dYrHL94Rvf+EY69NBD+6OSo1pVtX5sq6rWr3ZB9sg08MA2+HKjso5G/PIv/3IyYRjIlkYdReWQWFl01MykJA/4jsD6KmQ/HY4cveWxg9t/pJatsiJrolNVVe8Yr2d5TSAG6cnvheEsNz7yu24U4Aa/jfJFeiCwygiY57JV4zDQj/SnYXlM+qtq3Q5VVdX7Uhh741i8vl9VVXI83rP4cTbGCRXHy/K7KMP4jYpjK3wIhG1hR9iMUXlHxZOPnOStqirBxkkXco0qs1E83OC3EEfGALO2tjZypSy19G9wNyazLV2+LGdpV1+pmPUsZWm6tC0P3ODXNF9tu54FhKYlDfqBwJ4IaLvGDc4Mp0aw46A9myyY4LPpjgobJJWW31WcQU9ez0L/cU6DbD4m6qhGf3BU1UDev6AhDo88CHvOZXx57IMf/OBevxXVzwOvfBQVDS/lK++a88nzn/7Tf0r9fOXFV17HSdDoT++XQz75x9Ebll8ZZaXBSXk8YOBZ4ETBz32EQCAQGI4Am2WR46CDDkrDFlOU0o/0J/f9wfFa/bw/iJNHX9T/pbl6Fq+vshviXdkHfVmfls6+iJfPs3jp+joa+nlOEydNHnmVRffv/u7v0t/8zd8MPZarLBpoCe7FKS+IIy86/XIM8qInfrnMIB3xAtzgtxBHhgAGIyD1Dyzi2wwGN3IM4ym+ZPmGybzoOOcuNbhFy9FF/nCDXxdlD5kDgUUgYJLgGJrjZ5wawdlxn/HMjo1+ZXfEy612Pp70pCelPOaI6z+z3YoOS8IEbvBbEnVCjUCgEQTYqIsuuijpK+abwxwaafpTIwLURNSudFWt7wzZdXaULTsaNbGYiQzc4LcwR0YFWy0z4MykwZyFrNwZBMkxjJT4kuUbJvOi42w72lkwYVi0LF3iDy+4wa9LcoesgUBJCLDZbHq/Y+PIpoGOnK4CZ8a48+///u/pwQ9+sKQIUyIAN/hNWSyyBwIzIdDlQuaRdjXYHkdFBx0a/Uh/KllHOyZ2UXKwY1KCvHCD38IcGSAsctdDY8KfHKOC9EXtykwi3yi5FxVfVVV6xjOekayILkqGOvk6tuLMep00h9GCF9yqqhqWHHGBQCAwAwIGW4sEuaidGcGzjwAYlB/0oAd5jDAlAnC77777ep+ur6oqroFBtIERbeDwww/v7cjkLpYdGkfOvPMWdigjM/2VHYLfQh0ZK2i8Vatj06swe4mNdmNSWqddunzrUpb191nPelby40llSVW2NPCCW9lShnSBQHcQsADl965+9KMf9V6893sDPh//gQ98IHnx3VcCq6pKP/7xj7ujVEGSwm2fffZJJhER7g8c7g8MxvWDvICSu7CPkJx//vnJcdiqCjuUcZn2yg5VVZUW6siknf8WsesxzW5H6fLthLCo/7/yK7+S/Hhkk0JZZeUAV9X6SqDzm/gN7qDYTREnzWcDH/KQh/RWjZzv9AlB8cqa5AhVVSV55FXWF0Pyl4TkbSrAC25N0e+nyznP7wj0x9dyH0QCgQIQ0L7txnBk9G+Oy2c+85ne1ynZjSziT/7kT6Yf/OAH+TGuUyAAN/hNUSSyBgIricAVV1yR7r333p7uHBh26Zvf/GbPHonUj/Qn9xGmQwBu8Fu4I2NiZXBpa1dm0t2YDGfp8mU5S7k++clPTlbqbrnllsZE8hlAzogVED8I5zOEnI+TTjqp50RxdAQOgjhOy/Of//zkU6PKaG8vfOELkzyE9CNRb3zjG5MfmDrqqKOS75t7kc2nDP1onXv5mghwghfcmqA/SFN7Fmey5xohEFg2BLRx78lYhNLXR+lnlZSTMyq9rfi6+Fi0sQBTF71xdOAGv3F5Ii0QCARS72tl3pEZdGAyNvqR/pSf27ia+7CNFnra4LcRD/M3i8nmahvl7U+HG/wW7sgQyoCzffv2XV+TEddU4B3jNw19+UuWbxpd2sj7ghe8oOc0NMFLQ1cXHBT0nTHlfPiVWL+XkH8Q7h/+4R9651K9QP+pT31K1uTezdlnn91L47h4zt9VZ2x8Z11cW4FzBa+2+OFjorcsjgw99GkLIa7aBh0jBAIbIZA/3blRvkjfGwGfPIXf3ikREwi0gkAnmBiPHGnt34EZFFw/0p8G4+N5YwTgBr99Ns7afA4TK96hyUiT3KbdjcmylC5flrOU6ymnnJLe+973NiqOH2Sqqir5FKCX5X3163GPe1x61KMelTg13jvxOeP8DfOvfe1riaNTVVXPofnSl77Uy0dIL+Oh477tACd4tclXe4ZZmzzr5mWAUG8c2ayLKxsi3lWeuvkGveVB4JBDDknelWlao7z6WVXLcRQWXnCDn/sIgUAgMBwB81oL4cNT12P1I/1p/Wnjv3n34jd/8zd7R+Xzcfhc0q5sVa3bmpyWbZA5UVVVvZMrOb+rnZmc13N/kFZV6/Tog5a4fC9vlslCs3RpVbVeRt7+PINpyjgx42SMH9v0LP8kAW7wK8KRIbDKNvGwwuq5iWDFFp9ZaL/+9a9PJcs3i05NlTniiCPSMcccky6//PJGWPh2uF0Wx8RycPwr76i8//3vTxybvGtDCL8vYQcm5/ejTH6cSdqiAnzgBK82ZeDItMmvTl7sgwUJwfGh22+/Pbnq165enhTw5MwI7iMEAoMIPPrRj04+zzwYX/fzMh2FzdjADX75Oa6BQCAwGwL6kf40TWkLsf/yL/+SzGde97rXpXPPPbd3VJ7TcOmll/YWaaWZAzkqn2k7viW+/+SJMhdffHGyuzE4J+KgsF/mW+ZP6Hh2usXX15x8Ede/cCx92NF/+ciNNxkc3cfXIvJf/MVfJE7WJz7xiZQXn+XfKMANfsU4MiZXa2trqamJh4mP3xjAJ83wT7mS5ZtBpUaL+EVZHapuJhq5Bp87p45mJUFnxOv4449PHBlOp84mztWEN09wrVjoaNN4/ujUHeADp7rpbkSPUwefjfLVkV4nDTLbbdEX1af+OIy+9OzYWLxoyqYM4x1x3UHg8Y9/fG/Bo0mJ2Rjt1oQCHzuIXT4KSwfBQhH83EcIBAKB2RHQj/SnaSgceOCByRF5ZTgE2anwey933nnnLmfgyCOPlGVXGHz2ju4FF1yQ3ve+9yULwbsyPnDDQXHKxWkX6Rwm9L3X690U6XZg+t9HHmXvkCS3OZr7bBPdzxrgBr9iHBmKmHwA4Y477vBYazChQX8eosqXLN88utVd1le4NPSrr766btLpkksuSTpTVa0fE7MioQNjpMOZKJjkcnrEufL4fRSgqqr0hje8IXkWL31U0Omb+moZXOADp1H8m4o30W+ijzUlL7r6ncUIzqh+KG6jQE/55eMAdU1nckdoDgELIp/97GebY9BHecqjsKnUo7BZJbjBLz/HNRAIBGZDQD/Sn6Yp/fCHP7x3VH6wDKfC3Keq1o91mb/05zGn6X/+2Mc+loyTdlz64/vvHdm2a1JVVWLHLCL6DSk7KxwJuzIcKQvGuZx8VbXn0X9po+SWNkuAG/yKcmQAqhLqXkE1AZpnNyYDXLp8Wc5Srq9+9auT3Y+65bEyYGJre1JwrCzzyGk+v5rjXG2ZOk4mv6tn8RwgtJTzjFYu615+V2l1BrjAp06ak9LavHlzKx/WmFSejfKpHyvZjo6xDxvl70+nK8dny5YtCY1wZvrRWe17A6+B2ODfJBLLchQ2YwQvuMEvx8U1EFg8At2UQD/Sn/SrSTX49re/3Ts+Jr93gj27t8jr6hiYuYuvrnoeFeyw2JER7B4Py2fHB60cLCJbBLaz4r2WW2+9tXcsTJzyo+ydtDoDvOAGv6IcGUqadJi41DnhqGM3hmxC6fKRsZRwwgkn9F6+f8tb3lKKSEXIAQ/btfBZlEBra2tJP1sU/0n5sgMcEDsraztlnrTcYD79Njszg2nxvJoIVFWVnvGMZyQrjk0hYHC3YrgMR2EzRvCCW1VVOSqugUAgMCMCVTW9Hbrnnnt2/fC4413mE06j9IvAMXF8vT9u2L1FXTbK7+YNpnNWjL2O8EvziXfjMCcCP6dKXv7yl6d8TGwje4dGXaHfDhXnyFhBBVRduzJ17cZk8Pvly3HzXOuWbx5ZmiirHnn7vi7RBP2u0YQDPOCySNm1Y07CImWYhLf+MctOzDDanBm2Bc1h6RG3egg861nP2jUhaEp7q6RWMauqu0dh+7ExcYJbf1zcBwKBwOwI6E/61aQUHvvYx/a+PFZVVeKs5Hdc7J44+uUomC+AveY1r+kdw3cKZRxtOzdoZIcl5+XksF92PaqqShY/5XOCRXC8LP98RS4j/yh7l/MMXn1R1m/CkJkDNpg+7BlecJNWnCNDKBMOgNUx0apzN4ZsQunykbGUwNM/77zzklCKTIuUAw4CXBYphxf+8xG6luWYmJ2dmLWduzB2UiYutEHG3HfZhQ2yRvIKIOAdNS+qNqmqAd94lo9m9B9VzWmDfdEEwuRDGVfPZFzUUVi8c4AX3PJzXAOBQGA+BPQn/WoaKhwK9oHTYCdEWVfPOd7HhNgev7Xiyn7Il+1OflbO+ybZzsiTgzzoCWjLm9PYssG4TFt+QR750cYjl0eXTPKLQ0dwL/9GAV5wk69IR8ZqsQnMvKvWVl5NgtCjbF0BvZLlq0vPuuhs3bq19/LqZZddVhfJTtKhv5d44bFoBbTfOhYKmtKDgSMfx6NOHvqurXK2BY86aQet7iHw5Cc/OfkCj6/3lC19GdLBCV5wK0OikCIQ6D4C+pN+pX91X5vmNYATvOCGW5GODMFMYEw0TGY8zxKsuqIzS9mNyqBbsnwbyd92+jve8Y501llnpXFfx2hbpjb50Zv+cGiT7yheJvSCNjwqzyLjORqOlDUhA731XwsdTdAPmt1C4AUveEG69tpruyX0gqSFE7wWxD7YBgLTIdCh3PqV/rWRyIM7GxvlX8Z0OMEr67ZPvintarJh1diEZhbZTFKa2I3JspQuX5azlCvP+Z3vfGd66Utf2vvhplLkakMOL8bRm/5waIPnJDxM5mftX5PQnzWPBQhl9X/XJkK2DSXq34S+QXM0Aqecckp673utptveAAAQAElEQVTfOzpDpOxCAE7w2hURN4FAIFALAvqV/lULsSUnAid4ZTWLdWQIaKJlxXiWXRmTIeXRaSqgPyDfxKzakG9iYVrK+LKXvSw509jfAFtivVA29KU3/RcqyABzzvgsfWuATO2PFiGa2o3pFxYP/bBEDPrljPtmETjiiCPSMcccky6//PJmGXWcOnzgBK+OqxLiBwLFIaBf6V/6WXHCFSQQfOAEryxW0Y6MiZZV2WlXTU2E8oprVrSJa+nyNaHzvDT/4A/+ID3sYQ9L6mheWl0oT0/60rs0ebVfwWR+sbLt5u4Ff+8QkWt3bDN3eLAT09qXZqQJqotEwEuxvv6zSBlK5w0fOJUuZ8gXCHQVAf1LP+uq/G3IDR849fMq2pEh6Cy7HiZmyinfdMBn2l2ZNuVrWv9Z6P/Jn/xJ+tKXvpROPfXUWYp3pozO5pOC9C1VaO138ItJi5I19yMytSWD7+LjK7TFM/iUh4Ad002bNqWrr766POFGSdRiPFzgA6cW2QarQGClENC/9DP9baUUn1BZuMAHTv1FindkrJpOsytjBdwqq3L9ijZ1j0/J8jWl97x0//Ef/7H3fXP1NS+tEsvT6+67704f/OAHSxRvl0zabylHq+yMtOnEAIH+eOLtOcLqIvDqV786velNb1pdAMZoDhf4jMkSSYFAJxAoXUj9TH8rXc5FyAcX+AzyLt6RIbCJhhXTSSZci9jtKF0+GJYUHB/i/H3sYx9LP/rRj9KJJ564NB8A8GI/fehVuhOjTWzevDlt3hn0G8+LCvq2Pm4Rom0ZMs9FY9C23sFvTwROOOGE9KhHPSq95S1v2TNhxZ/gARf4rDgUoX4g0DgC+pn+pt81zqxDDOABF/gMit0JR8ZEy8R3o1VTq+AmJfIPKtrkM36j5dvNeVHy7ZZg8XcmqyatXrQmjWNXP/dzP5ee/vSnd/7TzD6xTA/60It+XQgc8UUfL9O3vRuzKLxgQIZF8Q++ZSCgDVxwwQXp61//ehkCLVgKOMADLgsWJdgHAiuDgP6m3+l/K6P0GEXhAA+4DMvWCUeG4CYaeRLseViwoirfsLSm4/AtWb6m9Z+EPgfGbkx2YnIZL8KfffbZ6SlPeUryo5E5vktXcpOfHvTpkuwccXWj/S5Kbn3X+yq7+Ld8YyECDuRomXWwKwiBJz7xiem8887rhYLEWpgoGQu4LEyIYBwIrBgC+lvueyum+lB1MxZwGZahM46MSYbJxiiPbNG7HaXLN6zy245TR5wY9TjI26eJ7WhcddVV6eSTT+7MiqiVAvKSm/z0GNSt9GdtlyM+qm81LT++i9hJHdRrkRgMyhLPi0PAzuDXvva1Ti6q1ImaxRk4wKNOukErEAgENkZAv9P/9MONcy9vDvrDAR6jtOyMI0MBEw2rxlaPPfcHK6nS++Pavse/ZPnaxqOfn8mqZxNW12HBj0V+/OMfT45mCc5EDstXShz5yCmQm/ylyDatHNm51H6nLTtv/hL6Lh1gwKlbBAb4RygHgXe84x3prLPO6vxx11kRtShDfzjMSiPKBQIdQaBYMfU//VB/LFbIBgWjN/3hMI5NpxwZkwyTjTwpzopZ6TdBlp7jFnHFv2T5FoEJniaGJqsf/ehHPW4YeN4cgxtvvDE94QlPKO6TqD4BSC7ykZO8GypVeAZtlyOuL7Upqnahz+DfJt9RvGAwaF9G5Y345UXAosQ73/nO9NKXvnRpPkQyaW35YAm96Q+HSctFvkAgEJgdAQv0xsN+Cvqffqg/6pf9act+T1960x8O4/TtlCNDERMNE2OV7llQ+eLdLzqQY0P5Fi1ky/yHvRezkQjOQvrq1+/93u+ld73rXelJT3pSuvzyyzcq1mg6/uQgD7nIR85GmbZIPDsU+lNbbDkNi3w3ZlDP7FDpw/1pg8/9aXG/nAg4Jur3Ck455ZTlVHCEVvSlN/1HZInoQCAQqBkBY4+P7hxwwAHpv/7X/5rymKMf6o/6Zc0siyZHX3rTfyNBO+fIqOy1tbVkApR2/rOCXMJuzE5Rev9Ll68nZIt/ODF2LNTZLGx9as9ODsfhuuuuSwcddFA655xz0i233DILuanL4IMfvviTgzzkmppYBwpwxHPfalrc7DBN2jaalgd9/Zdj1Y8BOS+55BLJEVYMAR/ueNjDHpaMM6ugOj3pS+9V0Dd0DARKQsA7xN/97nfT9ddfn573vOf15jvGov/8n/9z0i9PO+20ksRtTBZ60ndSO9Q5RwZyJlu81TvuuCNdccUVybP4UgJ5tm/fnkqVry2cdEC84OE6T+CZ2wG5+eabd3Vo76ZwMjgYtiHnoZ/LooMeuujnDoUv/uTIeZfxyqkwmdevmtZvx44dxfVdOsNA39WHBW3goQ99qKQIK4iAT6l/5zvfSWeeeWYXtZ9YZvrRk74TF4qMgUAgUBsCxt6LLrqo9/t6HJpvfetb6Q//8A/Tc57znMS5+cQnPhF2aAjanXRkVLbJxuGHH562bNnS+0G/IbotLGrz5s1pbeeuUanypRb+mQBu3bo1WWGok90RRxyR0P3CF76Qrrnmmt6KxVvf+tZ04IEH9o6fmXQyBB/4wAd6L+r6qhjn5Mc//nFPDFfP4r1IJp/8yjk2hg56dmDQxwc/fHsEVuCPOtu2bVvjmnKW9JPGGU3A4I6diyI56L+cbxgcd9xxvdLiejfxZyURsIhx9913L+3ODPtHP3quZAWH0oHAHgg0/5DHG3MlY42gH1pIvffee3cJwKHZd999k/BHf/RHST+Vb1eGJbqhF/2mtUOddGTUm4lG/9V9SaF0+ZrGSqd0BKvJCaAXwF772temG264Ien4Xgo75phjeh3dWdMzd66g+oHKRz7ykeknfuIn0j777NO7ehYvXT4dRznl0UEPXfSbxqlE+uqMg8GoNCUfJ6a0RQhOi8WHqqp6E1YDDP3txhx22GFuI6wwAgbXH/3oR+nEE09cmg8AWNShD73ot8LVG6oHArUikB0VY535kGCMEaqqSq6C+Mz42GOPTXnumPr+vfKVr0x33XVXb4FcP9Vf9Vv9ty9bZ2/pQR960W9aRTrryJhsNT1RnhbM/vzTytdftuv3OicdTIZd2whVVaWjjjoqeTHs4osvTjqDHRc7L9///veTnZgcPIuXLp/8yilfVVUb4hbPgzE1kWeMmxDWsTJGuwnas9DM/fWkk05K++233ywkoswKIODYlSOnFkLYjy6rTH560IdeXdYlZA8E2kbA2GiMHHRU8mKYeZBgrMuyGVeF+++/P91+++29YB4rTtiyZUu64447emPQpk2b0sEHH5xyeqbhqr/qt/qvfiyuq4H89KAPvWbRo7OODGXX1tZcig2ly9cEcDq2jqjzNUF/HppVFU7KpPiZ2DOsDPGkZabJx/gz2tOUGZK31ig6O2Zop842fibuXlp+jutqI+AF1LPPPjs95SlPSX6srYtokJv89KBPF3UImQOBJhEwjzGfMVbZNXFCwXjY76iIH3RUzH36HRVHtY2lgjmhME7ufLSsfxdmWH79Vv/Vj/XnYXlKjyM3+elBn1nl7bQjM6vSUa4ZBHR8HV3HbYZDUG0TAY4Go8uA18nXwIB2nTTrosVh8Y7U+eefnxwpQ9dxQ/HuIwQCELCDayXxqquuSieffHKywyu+9EDOk08+OZGb/PQoXeaQLxCoGwFzFYGjwhkRjHPmL1VVpapaP/olPjsqThBwRvodFffmO+IF4+W8Y4V+iS56G+mt/8qvP6+yHQpHZqOWEukTI8AQbN26tXeOc+JCkbFoBBjT7du37/rceR3Cei/JJ47roNUUDXq//e1vT7b3OTJN8Qm63UXAO3R+ENeRCOEtb3lL0cqQj5wCuf8/e2cDq0dV5vHnFJeo0KW1EilLoaCANhtagYrS1b41TRbSpYuEuKUoFBdIAyyxVESIwL2wUbAWki40DRBoUWrjEmDLEtikkXthS6x81o1VPgTaGqgBlnYLyrqA29+8nHvnvn3fe9+P+Tgz82/63Jk5cz6e85t3zpz/nDMz+B+0w3JOBLokgEjBuGmGGMEQKZhzdZHCOuG+CC9UGE3BmPqFoPBChZtvSQgVX16zJT5TLuU0298sjPOY85nzGuM8bxYvlDD8w08Mv/G/V98kZHolqPQRAd8g0AGMAvSnFAS4u0RjzgUBQdNrpWioyaeThrrXMrtNz4Xr6aeftkMOOaTbLJSuAgT6+vqMC/IjjzxixxxzjN11111B1Rp/8Av/8BN/g3JQzohAhwT8dYTrEn0PDGESn/bFth9NIXv6JpgXKQgGrm2EYbT3bVyXyCo143rbbeac15zfnOec75z33eaVRjr8wS/8w0/8TaocCZmkSFY4HxoVfpTcuagwhtJWncaVhp4Rt14ryW8lrd8JF6he/WtMT90fffRRY9m4T9si4AlMnz49esEIH8y97bbbolfB33rrrX53LkvK55Xy+INfvNgEP3NxRoWKQAcEuE5ww8sLFa49CJO4UEG8NAoVri1cBxApGNtcuzBECtaBG4WLyvnNec75znnP+U87kGdFKB8/8Ae/8A8/k/RJQiZJmmnkVYA8aWRoMNTZK8DB6tJF7lZhHOsus4iScSEhn2ijgz9cnB5//HGjMVy6dKmdeuqpdvzxx9uUKVNs/Pjxts8++wwZ24Szn3jEJx3pyaeDYqOo+l1HGPSnDQLz5s2L3jLEBZuHdvke1ZIlS2zTpk1tpO49CuVQHuVSPn5w1xm/es9dOYhA7wQQKRhCBTGCcV1BqDjX2/MpXF9697D4OXC+c95z/tMO0B7QLtA+ZFE7yqE8yqV8/MAf/Eqj/HFpZKo8q0OARojaLlq0iIWsxAR4roULkD/maVf1qaeesu9973s2d+7c6HWU559/vm3cuDF6JSW+8MYThqh5v/6f/vQne//9940l24Szn3i8wpJ0pP/whz8c5Ue+5J92HZR/NQmcdNJJ0QgNv7sDDjgg+i4Rc8K5uHNh57sJSZAhH/IjX/KnQ0h5lMudT/xIohzlIQLtEuAagfnRFK4XiBTMuXCfT2m3fkWKx/lPO0B7QLtA+0A7QXtBu0H7kUR9yIf8yJf8KYfyKJfy8SOJclrlISHTiozCxyRAY6UpZWNiKk0ERiYYeeMCxd20NCr23HPPGb+padOm2cKFC+21114zGsc33njDeF6F8i+99FL7yle+MjQiw9vFGJHBH5Zs+xEZ4hGfdKQnH/IjX/KnHMqjXNLLRCBJAkcddVT0e96yZYutXbvWuEPJW/EmTZoUTT8755xzjFd+33vvvcbbh3irGJ0CvnmFHyzZJpz9xCM+6ZiuQT7kR77kTzn8nimX9B2aoovAmAS47tP+cx1ApGBz5syx+LQvtnmpi8+MqV0YI+JM+cK4Q08Yxo3QWq3mo2uZMAHaA9oF2gfaCdqLq666yv7yL//SJk6cGLVF//AP/2D//M//XMh2SEIm4R9MlbLjYkoHkQ5ulepd5bpyDYbS4AAAEABJREFUrDnmHHsuaEmxeOihh6LpYrNmzbJdu3YZZfzmN78xOmkMRyNOkiiLfMiPfMmfciiPcpmGhh9JlKM8RKCRAG/nueKKK2zDhg3Gm/BuueUW43e3Y8cOW7NmjfH9Ij4MN3nyZNt3331t3Lhx0ZJtwtlPPOKTjvTkQ37kS/6NZWpbBDol8PLLL1tcqNDWI0ziQgXxMjg4OJQ1YoS2tJVQqdVqJX6b6RCGQqzQTtBePPnkk3b77bfbzp077Ve/+pVxkwQhM3PmTOM6z4jKX/zFXxSiHZKQKcRPLzwnV69eHTnFnZRoRX8qQ6C256LEcefixkWvl4ozHE0+l19+uSEwGClBZJxwwgm9ZNt2WsqhPMqlfPzAH/xqOxNFFIEOCTjnjA4D34FYvnx5NA2NERdGXnbv3m2MxHhjm3D2M02D+KQjvXOuw5IVvcoEaK+xgYGB6JX6/f390bRH2jznXE/fT+G6UGW2Raw713EE6H777Wf/93//Z//7v/8bVYNp2txMYWp2EdohCZnosBXrT97e0hByl4YTIG9fVH4+BLgDRyPIBZDfQ6debN68ORqBQTjQKWPa13nnnddpNonGp3z8wB/8OvXUUw0/Ey1EmYlAmwScc23GVDQRqBOgLca40YhIwWijMef0fEqdkv56AvxWtm7dajNmzBj6ADT7JkyYEH3olz6ec+G3QxIyHDVZRwQQMX19fTZ16tSO0ilyuQh0K2b47TBV5ktf+pL98pe/tDPPPDMLMG2XgT/4hX/4ib9tJ1ZEERABEUiJwMsN0768UIlP+0K0MAXRu0A7jbWa9rVo0SKr1Wo+upYlJ8BviN8N/Th+N4jeeF8OEcNNPERMUVBIyBTlSAXiJz96XKFhZCmrNgF+BxgXTxrI0WjwljCEAc+mYJdccslo0XPfh3/4ieE3/ufulBwQgUwJqLAsCdCGMu2L66zvbNK20uF0rj6iQgdUz6dkeVSKXxa/K/974rfEbwzx4sXt7Nmz7d1337UiihiOjoQMFGRtEeBkoBGl49pWAkWqBAHu6PGb4ILLb6RZpfmOC991Oeuss2zdunXR91+axQstjLef4S9+4z/1CM1H+SMCIhA+AdpGjE4knUqM6yntpnOu7edTeOMXd8tpczFGU7DwCVTIwwCqym/N/8YaxYt/Y1zcTV4cUrSRGO+/hIwnoeWYBGh0mWajRnNMVJWLEBczXKjjAL797W/bihUr7Be/+EX0Zqb4vqKs88Yo/Kce1KcofstPERCBbAjQccT8aAqdSEQK5lx9NIV1wr1H3AlHjHBnHEOk0Mn0QoV2lestd899Gi1FoBUBfn/8vuirtSNefD6M8PFb43fnw4q0lJAp0tFq7Wvqe2icKYRGl6VMBBoJ0BDy+6AR9WLm61//ujE1iw9UMqLRmKZI2/hPPagP9SqS7/JVBESgNwJ0EmnXuBbSWcQQJnQYnRsWKvHnU3jrE22iFyleqBCG0WYiVHrzTKmrTIDfJb9Frrv8FvmNInz5zSGK+Z2NxQdBXVQRQ90kZKAgG5UAJwonSTsnxKgZaWfpCXBhpvHk9/I3f/M39qEPfcjWr18/4o0o4UDo3BO+Q0N9qBfD8J3noBQiIAIhEuA6RyfQCxXasEahQhh3r73/XBPpANJpRKRgtH+EY7SHEiqelpZJEeC32qt4ifvC7zS+XbR1CZmiHbEc/KXx1pSyHMAXtEjuBh155JH213/919GHLQtajVHdpvNy0EEHmcTMqJi0s4wEClgnOn4YQoUOIMZ1DaHinJ5PKeAhrZzL/H7977bbkZeyQpOQKeuRTahe3J0iK+4usZSJwFgEmHb1V3/1V7Zq1aqxohZ6P/WbOHGiUd9CV0TOi0DBCdDJw+JCBZGCOTc87YuOoK8q02m4rjGagvnRFG5SEM5dakZTpk6d6pNoKQJdE+gmIb9pfrOIbomX1gQlZFqz0Z49BDiBaNT3rOq/CIxJgAfhd+3aVdqRmEYAdHqoL/Vu3KdtERCBZAjQoUOkcGONjh2GSMGcay5U9HxKMuyVS7YE+K3z+6bvJfHSHnsJmfY4FTBW7y5zIvm7Ur3nphzKToBXEz/00EO2du3asld1RP2oL/Wm/iN2aEMERKAtAnTe4kKFaw8ihY6cc3WhQljj8yncZPOjKX5EhTBM16620CtSAAT4/Uu8dH8gJGS6Z1fqlP6iwh3nUldUlUuEAB+LPP/88+32228v9oP9XdDgBQDUm/rDoYsslEQESkuAThrGNYXOGoYoQag41/z5FKZzIUZ4cD4uVLgeEY4x7QsrLThVrNQEOCf8uYBg5/zgd8/vnd89v/FSA0iwchIyCcIsU1acYFw0ylQn1SU9AhdddJGtXLnSeEVxeqWEmzP1pv5wCNdLeSYCyROgQ4bREeO6gSFSMOec0UljnXBf+ljPp9CJQ6TQsfNptBSBohPgPOE8QMhzXnDO8BuXeOntyErI9MavlKmZh0zFGJpnKROB0QjwRrtDDz20sB+7HK1unezjo5lwgEcn6RRXBEImQOeLDhfXBTphGMIEc64+7Yt1wn094s+n0EnTtC9PRsuqEeD84dxoIl5MIy/J/BokZJLhWKpcOOm4I1aqSqkyqRDYvHmzLVu2LLJUCihYpp4FXArmutytKAE6WnGhQocLYcIdY+fqQoUwPZ9S0R+Iqt0xAc4p+lGcN5xHnF8aeekYY9sJJGTaRlWCiG1UgZOPIX2sjeiKUnECCN5rr73WpkyZUnES9erDAR5wqYforwjkR4AOFUZHirYdo3OFUHFOz6fkd2RUctkIcJ5xfnFuSbxke3QlZLLlHXRpnIhMi1EnLOjDFIxzDzzwgL344ot2ySWXNPWJ6SXO1TtLzg0vDzvsMHv11VebpgkxsFOf4AEX+HSaVvFFoBMCtNlYK6FCh4qOFR0sn6+eT/EktBSB3ghw7nFucY5hnIfcBGY6paaN9ca2k9QSMp3QKnlc7tQhZBgCLXlVVb0ECPzwhz+0yy67rGVOa9asMRr0V155xXh2ZP369dH21q1bbfLkyS3TlWEHXOBThrqoDvkRoKOEdfJ8Siuhwg0qjGcf6WylXCtlLwKlJMD5GBcvVJKbdvHnwAiTZUdAQiY71kGXxJ0ETlAudEE7KueCIMB3U3bu3GlnnnlmV/48+eSTtt9++5lzzhpHaO6///4o3DlndLjeeustw1jnguFcfXSHdV94PI1zztj2+66//vooP8o79dRTzafzeTpXz8+nwTeEF+Z9I41z9Xis+7xbLeECHzi1iqNwEaDNpe1dvXq10TniZhJ3dhlJcc4Z69jg4OAQLNpojJsEdJ4wf/eX8EWLFkXnzVACrYiACPREgPO0v79/6HwkM841zj2WnHOE9W7KoRsCEjLdUCthGi6get1yCQ9sSlVatWqVXXDBBV3lzrSy0047zdatWxeN0NRqNTvjjDMisYKIuPDCC+2JJ56w3bt3R/mzHa3s+cNoDuGM7tx9991GfIw4pKFzd91119ny5cuH8rv55puj/F544QV7+umn9+RS/08ahAppyG/BggVRfux94403jHSUR74sKRfbtWvXUDzitjL4wKnVfoWXmwCdH2xgYCASKXSEaGcRJs7VRTHrhHuhwmg4HSOECb9LOkoYbTPhGOcLVm56qp0I5EuAc5dzk5sKnKd4w/nH+chS5yBEwjAJmTCOQ25eUDAnKxdQnZjQkI1F4LnnnrONGzfaeeedN1bUpvsRBuzg2yssL774YmP0ApGwYcMGO+KII+zoo482PjS5dOlSQ0S8/fbbRLWTTz45CiftkUceGYUdd9xxtm3bNmNJwLRp01hERn5chNjHdDbEBTsQU3QwTz/9dDajO20zZ840psERMGnSJDv44INZHWH4dN999w2VNWJnwwZ84ASvhl3aLAEBOjoYvyPaUCwuVHwHiHDiUWU/7YvOkBcqiJZGoUJ7THyZCIhAdgQ4Tzlf/blLyZybnK8SL9AI0yRkwjwumXnFidvX12ecrJkVqoIKTWDt2rX2ta99rac6IDwQCs656COazz///JCI4O70+PHjo+lg8+fPNy4iiBwKjIsUtr0x3cu5+l1u0vjwLVu2+NWmS+I654zyKNfH/9jHPjYkZE455ZRIQBHHOWfXX39907yaBcIJXs32KSxsArSNmJ/2RQcHUYw55yLxyzrhviZeqCBSMH67caHCFBRuGEmoeGJaikC+BDjHOYclXvI9Dr2ULiHTC70SpOUOIkJGF9YSHMyMqvDTn/7UmIbVS3F0+BAndPYwRlwYNSHPs846K5pyRjjGiMxBBx3ErqbGsy3cFWc0hfhME/MRWwkf9k+cODGackYabzykz75GI5w4+Pzggw+OeAanMW58G07wiodpPQwCdGD43TQKFTo0zrkhoYLA9R5zVxbjt4BIwRAqhGFeqPj4WsYJaF0EwiDAuY94ca5+nuMV5zHnM+cxNxsIkxWDgIRMMY5TKl5yEeeE5sRNpQBlWjoCTz31lL3//vt2wgkndF03poVxweDCQSY8jM+zKkz3mjt3bvS1Y557YR8jLVxUEDpsj2U8wM/zMT5ePD/yX7lyZbSLaWbTp0+3FStWRNuUx8sAEEVRQOwP/uGHD5owYcLQaI0Pa7WEE7zg1iqOwtMhQNtGG0eHBeOmDSMoztVH7lgnvFGo8LuMCxVGq2kjMX6LWDoeK1cREIG0CNAecL47NyxeuA5hnNvB3sxNC0iJ8pWQKdHB7LQqnNScwJ2mU/zqEuAtXDyn0gsBRMQ999wTjeo45+yaa64xtglnVIaH7BE7zjmjI/qTn/wkesNZqzLpkHIXnalqn/jEJ6J8d+7cGb0sgPx4qJ/8Pv/5zxtTyXw+lMNoj3Mumt521VVXGdPI/H6/JL2Px/SyL3zhC209I+PTwwtuflvL3gnQKcH4fdCOYa2ECvEokVFA2js6Ll6oIFoahYo6NNCSiUDxCfj2wTmJl+IfzdY1kJBpzabUezjBucAzFaKhotoUgZYEfvaznxmjHC0jNNmBQEEIxEUCAoNRFjqULNn2SYlHOEY60vOQPb9Z9hGPsGeeeSYSFH4f8clr8eLFxj7i2J5/flrYr371K/uv//ov89PN4ulIS7w90aM84+lbxSNuOwYvuLUTV3HqBGibsNWr668lRqggWDHn6p0S1gmvpzDzQoVjiSFY4kKFtq5Wq5mEiiempQiUj8DAwED0lkDnnHFzgxr69oAbGTr/IVIuk5Ap1/FsuzZ0ALgT2XYCRaw8AS4Gjz76aNRhLAoMpoo5V59KxGgKU9i8YEm2Dq1zo4MNN/i1jlWtPYgUOhyNQoWRNeeGhUrjtC86InBEpGAIFcKwRYsWWa1WqxZI1VYEKk6AtoT+DKLFuebipeKISl99CZnSH+K9K0jngVBd9KEga5cAr01mNIMRinbT5B2PERw6vt7WrFmTuUvwghv8Mi88pwLpXCBU6GBgdDIYQXGuLipZJ3xwcOSHHhEmHDbSAj0AABAASURBVCtECsbNFkQKRnuF5VQlFZsGAeUpAl0QoH3p7++PRly4+UFbw0iLbztoL7rIVkkKSkBCpqAHrhe36czpRO+FYDXTbt682WbMmFHNyvdYa7jBr8dsgkhOJwKj80BnAmslVIiH04xK0eYgTnxnA9HSKFSmTp1KdJkIiIAIjCBAW0Jbw00QjPaH9oL2hLaE9mVEghJvqGojCUjIjORR+i2NxpT+EKdWwV//+tdDz5ekVkhJM2ZEBn5FqB4dBoy2go4DRscBc2542hfhvj5eqNCpwBAsdC68UPHTvuh4+DRaioAIiMBoBGiHaGdoezDi8gZJ375IvEBEJiFTsd9Ad6MxFYOk6jYl8Nvf/tY+9alPNd3XSSCvSGaKEM+vdJLOxyUd6cnHh7HkFcqMfPCaZbazMvzAH/xqVSbc4Ndqf5bhdA64m9koVJii4dywUGmc9kWnwYuUeEeCcC9UsqyHyhIBESgfAdqnVuLFtzXlq7Vq1AsBCZle6BUsLR0XXKbTxVImAp4AHVsuHn672fJ3v/udTZkypdmujsJ4ZoTyeH6lo4RFixzzF27wiwWltkpHAL6c7xxTP+0rLlQIbxQqjKDEhYofTaHzQJuBpea0MhYBEagsAdos2iTaKD/yQrvDDROW3CipLBxVfEwCEjJjIipPBI3GlOdYJl0TpvzcdNNNduCBB9qSJUuaZv/73//e+E5LfCcfi3Su/gA3H5RkVMSPUPDBSeec/eu//qvRCW7cZgSDaQLk4fNknTC22e9cPW/nnLFNeDNjFOa0004znkPhezFs4ws+OeeMt5UR1piWOIzi/NM//ZM556Lv1RDm4+GPc3UfyIt9jfV78MEHffRoiZ8+bhSw5w/c4Ldntaf/XPAxhAoXfswLFefqftIRINwLFT/tKy5UWG8UKvwGenJOiUUgAQLKohoEaMdop5yrjwJTa9okL164ZhAmE4GxCEjIjEWoJPu5O0tV1DhAQdZIgE7ssmXLjA9J8ltpJmjYN3HixKGkdNhXrlxpr7zyinEn//TTT7cVK1YM7UcgEM4HIQls3CaMNAgBxAHGOmEIBj5EyZu+yOO6666z5cuXG3FI12h8M4aPaiKWfv7zn0e7ETZ85JL0/O7POOOMpumff/55+5//+Z+oDsRfunRpFK/T+lEoafAT0RL/Ng7c4Eec0YyLO8Yx4CKPIUww5+oXfNYJ9/l4oUI9MToCcaHC3UzqzzH2abQUAREQgawJ+BswztXbMsqnvcIYeaGdIkzWMYFKJ5CQqcjh12iMGY1oRQ53V9Ws1WrGtC863K+//rrRmY4Lmj/84Q/2kY98ZChvpob5D1YSyAPtLL2NtU28gw8+2F588UV79tlnI6Ps448/Pvoo5bZt26Il8RrzImw0Q1whHs4666wo2sUXXzxUThQQ+zNp0iRjP0HExwf86bR+mzZtsiuvvNJ+8pOfRBzJzxvc4IdI4XcIW8QIhjBhSoVz9Ys72340hfRc4DEvUrjoI1QIwxYtWhSNeBFXJgIiIAKhEKC9o41j1Ni55t940Q2WUI5Wcf2QkCnusWvbczpNRK7VaiySsSa50GDRcDXZFUQQjWnI/jnnoulNzuWzpDNNJ94fLNYRNEw5o3NNR3qfffbxu6NRC35TztX9/c53vjO0j5VG8dG4TZyjjz7ajjjiiGhUZ8OGDcaICqMr7GOKmXP1vOfPn09Q24aQefPNN4fiI5g+9rGPDW3HVwhnfzyMdUZ/Oqnff/7nfxoXZUaRSB83uL3//vsGR86TRqGCMIEvIgVjigUiBcMHLJ6f1kVABEQgRAJcY2njaOswbtzQLvr2jTYtRL/lU3EJSMgU99i17TmNShaNh2/A2nYsw4hexMAiw2LbLgr/uLNOY5+nTZgwYYTPH//4x+273/2u0dF2ztl77703tP/mm2+O1nfv3h1Ny2L6VxTQwR9GgJh6dvfdd9uWLVuMaWUkZ4oWF0AECTzWr19PcNuGMGFExicgn//+7//2myOWhLOfQJZss95p/ZiSxogM1vg8DtzGjRtniBRYNgoVLvSUKRMBERCBohHw136EC4b/3Ijy7V0W/Q/KlFWTgIRMyY87ozF0krK4o0tjReeTRi00rHCgAxmyf/DLkxuM3nnnncgFBExfX5+99tpr5v366Ec/an/84x+j/Y1/6LjzvExjeDvbc+fONYQMx4ZpZY1pGBnhuZPG8NG2ETKMyNx5551RNJ7dYeSHEaAoIPbnjTfeMEaDCGLZLF679eO5GEaVfLnkicENfqzLREAEOiKgyAES4DrPjUFG8r144VqBeGHJjbkA3ZZLJSQgIVPCgxqvUpbPxnjBROMW9yHvdT/agZjDQvUPfnmy4m1ljJA0ChjvE6M1iAO/zfMkXLTGjx9vvCns8ssvN56Zefvtt32UtpaIi5kzZ0bPefhpZVwYuUAiSHjj14IFC4ypboz+tMqUuPiHL8Th4f9rrrkmmq6HSGr27ArxjjzySOMlA845Q4z5eN3Wj5Ep8uCFBeSP4Rf8WJeJgAiIQBEJePHiXP15PurA6DLXAcQL11fCZCEQqI4PEjIlPtZ03mh4smxcaMx8uaGgZaQBv/CHZcj+4WMeBpNvfvObI0ZgGv1AUPA2Lh+O6EC4MPWL5eLFi418iMeSh+WJizhqZxvRTXzMpyFvhBF5P/PMM0aZ5Et+xCGuN/bhB8Y6oyOkJQ8f5uM2LhEejfHIg3Q+HB8od6z6kQ5fKd+XAzfS+W0tRUAERCB0AvQfaPO4+efcsHhBuGBcT2u1WujVkH8lJyAhU+IDTMeQhiarKlIOowo0bDR8bOdtfjQGv/CFZcj+4WMeBpOxfiuHHHKIbd++PQ/3Cl8m3OBX+IqoAiIgAqUmgHjh+s21k1FxllSYGzpevHAdJUwmAiEQkJAJ4Sik5AMjEXnMU6VDzF0cGsSUqtZ2tjDAn3gCtkP2L+5rSOuf/OQn7YUXXgjJpZ59YdSE0RNGUXrObJQM4Aa/JlEUJAIiIAK5EuBajXhhSi/G9RGxEhcvuTqowkVgFAISMqPAKfIuOvB5iBiY0QByh5+Gke28jDtJMMCfuA9sh+xf3NeQ1j/zmc9EbxYLyaei+MIb2eBXFH/lpwiETUDe9UqgUbyQn940BgVZ0QhIyBTtiLXpLyKCRqnN6IlHC2HUAzGHH80qRzh3nWjMm+3PImw0/7Iov9MyeBsXoxedplN8M7jBTyxEQAREIC8CXO/oGzBljJEX/OBa6KeMceOPMFlJCZS0WhIyJTywdJD9qENe1fPl02jm4UOr0RjvS+j+eT9DWvJqZEYWeB1ySH6F7gu84Aa/0H2VfyIgAuUiwA07rsPODT+srzeNlesYV702EjIl/AXwkH+eozEe6dVXX200otwF8mFZLRFz3GkarTz2h+zfaL7nsc85Z1/84hdtcHAwj+ITL5Nz5Prrr08838YM4QU351zjLm2LgAiIQKIEuN729/cbIy7OOeOmHgXEn3ep1WoEyUSgFAQkZEpxGIcrQSNG5zyEIeK8Rj1ouKk/5Q+T2XuN/TToNPp7700vpF3/0vOg+5y//OUvD308svtcqpWSj2zCrf1aK6YIiIAItE+A6z7XMcQLRkqubXHxQphMBMpIQEKmZEeVxoxOfCjVymPUo53RGM8ndP+8n6EsTzrppOjjkWn6w1QsLsLOuehjlvfff39UXOMICqMphLGTj0/ut99+UfzDDjvMXn31VYKNtDNmzLAZe8w5Z8QhLmnvvPNO+853vmOsR5FT+sPHNuGWUvbKVgREAAIVMy9eWj3vwrWtYkhU3YoSkJAp2YFnNCakBmzq1KlGpxSBZRn863S0I3T/MkDWURHHHnusjRs3zjZt2tRRuk4iX3jhhYYY4W7i+vXrbcGCBYb4OP300yMRhdDBEAiEIVpOO+00W7dunZGG39sZZ5xhxKHczZs327XXXmu7d++2mTNn2ooVK+yyyy6zs846y6677rponXhpGJzgBbc08leeIiAC1SDw8ssvG9d3rqXODT/v8vDDD5t/WL9Wq1UDhmqZGIEyZCQhU4aj+EEdGImgIaNz/kFQEAuEFQ0wDXHaDsGA8joph/gh+9dJXbKI+9WvfjUSDWmUhSjhWCBQyJ9pEoiPV155xQ4++GB78cUX7dlnn41s586dxgP0TzzxBFGjdVYuvvhiYx/Che1DDz002rf//vvbySefTFBmhriCV2YFqiAREIHSEOCaiXChHWTkhXUqxw0bL15Cu97jn0wEsiQwLsvCVFa6BEJ5yL+xljS0CKx6I9y4N7ntTkdjfMmh++f9DGW5cOFC+/GPf5yqO/Pnz4+miY0fPz56uQBv/Tr66KPtiCOOMEQNz53wOuPJkydHfmzbti0SOs65SLQ8//zzUTx20gEgH9azNjjBK+tyVZ4IiEAxCcTFCwKGWjCFFvHC6As33giTiYAI1AlIyNQ5lOIvd7IRDCFWhsYX/2ik0/Jv9erVRjnd5E+6gYEBC9W/buqUVpqjjjrKZs2aZbfeemsqRUycONEYZeHC7Y2pYH5E5e67744+zOlHbXBi9uzZ0dQxH//tt9+24447jl25GXzgBK+enVAGIiACpSXAdYcbfc4NTxnjmuRHXUJ67rW0B0EVKywBCZnCHrqRjtOJD7mxS3vUo9vRGE8xdP+8n6EsFy9ebCtXrkzcHUZYGGnhORYy59kYHtDnoX22586dawgZRCfTyghjyQWfu5Vs8/A+z9gwTY3tvAw+cMqrfJUrAlUnEGr9vXBpFC+0YxgiJtSbkqEylV/VJSAhU5Jjz7Qy7kqHXB0aZzqgNOJJ+4mQI/9e8iV9yP71Urek0/IWrgkTJthdd92VdNZ2880329atW6OpZYiUq666yk455ZSoHKaX8cwMF3lED4Es77nnnuilAM45u+aaa4xtwtnfyqZNm5baW8vgAh84tSpf4SIgAtUhwHUP4cJ0Maa7cq2h9owie/HCDTXCZCKQI4HCFS0hU7hD1txhGsWQR2TwmkaaDiiNOdtJWa+jMd6P0P3zfoay/Na3vpXKq4uZQsbvmQs8xrQyX2e/D+Huw1gyjYzpZMRnyTbhCCDyIh3b5OXTsk58luxL0hgVgk+SeSovERCBYhHw4gXhgoDBez3vAgWZCCRHQEImOZa55cRoROgixsNpOurhd3a5pP7k22XyEcnIh44vF6ARO3rYSNK/HtxIPOm8efOih+9vuOGGxPMucobw4KUE8ClyPeS7CIhA5wS4fnCzzrnh513uuOOOoVckF+Va3XnNlUIE8iEgIZMP90RLHRwctNCnlfkKT5061ZIclUlqNMY++Be6fx+4Gcyiv7/frrzyStu+fXswPuXpCBzgAZe0/VD+IiAC+RPgphfnO9ci55yxxCumi2HcHOOaR5hMBEQgeQISMskzzTzH1atXR+Ig84K7LJCGfSChN4RRd/Lr0pWmycgvZP+aOp1TIA/mX3rppYbl5EJQxcIBg0tQjsnpLfBjAAAQAElEQVQZERABCCRiXrwwXQwjU26CMVXVixe2CZeJgAikS0BCJl2+qedOR56h6iI1mvjKHSruYvUCiDtfadQ9dP96YZZG2r6+PuM7LqtWrUoj+8LkSf3hAI/COC1HRUAExiSAcOHmFtcs54anjHHTywsX1sfMSBFEoJAEwnZaQibs4zOmd0WaVhavDI0+FwYuEPHwTtYRceTTSZp245JvyP61W4+s4t100012wQUXRN9/yarMkMrhuzfUHw4h+SVfREAEuiPAtQnhwogLD+tz44yc4qMu3JAjTCYCIpAfAQmZ/NgnUjKd7SI2pmONeowFh4tKGqMxvtzQ/fN+hrI89thj7ZZbbrFvfOMb9tZbb4XiViZ+UF/qTf3hkEmhKkQERCBxAl68IFwQMBTg3zLmR14Ik4mACIRDQEImnGPRsSeMSNDhxjpOHECCXkY9qDvp06wG+SMUubh1Wk4W/nXqU9rxzz33XOO7KQsXLky7qKDyp77Um/rn7JiKFwER6IAAbTujLphzw1PG9JaxDiAqqgjkTEBCJucD0EvxTCsrqoih3vjOaBIXEbbbtbRHY7wfofvn/Qxp+YMf/MAOOOCAoTf3hORbGr7wW6S+1DuN/JWnCIhAsgS8eGHEBRsYGIgK0JSxCIP+iEDhCEjIFO6QDTvMaEFRXrs87PXItW5GPbIc7Qjdv5E0w9j60Y9+ZG+++aYtXrw4DIdS8oL6UU/qm1IRylYERCABAlwruWHm3PCoC20708UefvhhYz2BYpSFCFSHQEA1lZAJ6GB04wojGt2kCyVNp6Me3AFP89mYRi6h+9fobyjb9913n+3YsaO0IzP8Dqkf9QyFufwQARGoE4iPujinb7vUqeivCJSTgIRMgY8rjTUd7QJXIXKdu2HcMaM+UUDzP1FolqMxUYF7/oTu3x4Xg/xPJ//dd9+1+fPnl+YFADzYT32oF/ULErycEoEKEuD6wahL/EF9bvTFp4yV4XpZwUOrKovAqAQkZEbFE+5OOvSMTITrYfuecXHhgsNFaLRU3AWnzsQfLV7S+ygvZP+Srm+S+THt6tOf/rSdeOKJhX81M69Yph7Uh3olySm9vJSzCJSTAMKF6yDXDeeGp4zFH9TnJlQ5a69aiYAIeAISMp5EwZZbt241OtgFc7ulu1xwxhqV4aJFvJaZpLiDckP2L8Wq95w1D8JffPHF9rnPfc74aGTPGeaQAX7jP/WgPjm4oCJFoPIEEC8IFx7SZ+RlzZo1EZP4qAs3naLAXv4orQiIQGEISMgU5lCNdJQG/bDDDhsZWOAtRBkXIC5SzaqR12iM9yV0/7yfoS55NTEjGnfeeactWLDAtm/fHqqrI/zCT/zFb/ynHiMiaEMERCA1AlznuIHEdcG54VEXbiwhXvSgfmrolbEIdEwgrwQSMnmR77FcGnc6/j1mE1RyLk7Ui4tXo2N5jsZ4X0L3z/sZ6pKPRT722GPG1CzshhtuCNXVyC/8w08Mv/E/2qE/IiACqRGg/Ue4+FEXbmJRGG8Yw2iHy3bto34yERCB7ghIyHTHLfdUNPaMEuTuSIIOUB8uUFzE4tlyIWv+bEw8VvrrofuXPoFkSujr6zOEwSOPPGLHHHOM3XXXXclknFAu+INf+Ief+JtQ1spGBESggQDXMm5g0e47t/eoixcvtL8NSbUpAiIgAiYhU8AfAaMTdOwL6PqYLnO3jYsaFzcfmfoS7rfzXOJHyP7lyaaTsqdPn2689ev73/++3XbbbfbZz37Wbr311k6ySDwu5eMH/uAX/uFn4gXlnaHKF4GcCdC+I1w06pLzgVDxIlACAhIyJTiIZaoCd93iozKhjMZ4xqH75/0synLevHnGPHeEwwMPPGAHHnigLVmyxDZt2pRJFSiH8iiX8vEDf/ArEwdUiAhUgADChRtAiBfnijnqUoHDpCqKQCEJSMgU8LANDg7a7NmzC+h5ey7HRz1CGo3x3ofun/ezSMuTTjopGqHZuHGjHXDAAdGHNHk2BZGBwOD7LUnUh3zIj3zJH6FMeZTLCAx+JFGO8hCBqhNAvPT395tGXar+S1D9K0wgk6pLyGSCOdlCuEAwMpBsruHkRt0YlTn88MONKXRTp04Nx7k9nuBPyP7tcbGw/4866ijjmZQtW7bY2rVroxGaG2+80SZNmhRNP0N4LFu2zO69997ouzS8VQxx8t5770V1Zsk24bxljHjEJx3TxsiH/BiBIX/KoTzKjTLQHxEQga4IcF3SqEtX6JRIBESgBwISMj3AU9L0CDDqQe5+yXpbllEk75dfZlRspYrhLWFXXHGFbdiwwd555x275ZZbbNasWbZjxw7j+xGLFy+2E0880SZPnmz77ruvjRs3LlqyTTj7iUd80pGefMiPfMm/UkBVWRFImADiJT7qwjpF8IA+RvvITR/CZCIgAiKQBgEJmTSoppwnFw9GBVIuJtfsqR/PKrDM1ZEWheNXyP61cLuwwc45mzlzpvEdl+XLl0fT0BhxYeRl9+7dxkiMN7YJZz/TxYhPOtI75wrLIE3HlbcItEOAa89ooy60iYgX2sd28lMcERABEeiVgIRMrwSVPjUCod/JC92/1A5MoBk7J5ES6KGRWwUmgHhhpMU/68I61WHEBUO4VLQtBINMBEQgZwISMjkfABUvAiIgAiIgAqEQ8MIFweLc3m8Y06hLKEdKfohAEQkk77OETPJMU8+RC42G7lPHrAJEQAREoBIEuKYgXBh1wZg+RsUZccE06gINmQiIQIgEJGRCPCryKVECykwEREAERGCYQFy4ODdy1AXholGXYVZaEwERCJuAhEzYx0feiYAIiEAeBFRmiQh44cKoi3PDwuXss8+2P//5z4Z40ahLiQ64qiICFSIgIVOhg62qioAIiIAIVIPAyy+/bAgXpophAwMDUcURLRjChe90RYH6kxABZSMCIpA1AQmZrIn3WB4XJz0f0yNEJRcBERCBkhHg2uCFi3PDoy4IFoSLpouV7ICrOiJQFgI91kNCpkeASi4CIiACIiACWRPwwgXx4tywcNF0sayPhMoTARHIk4CETJ70uyib0RguYF0kVZJhAloTAREQgcIRoO1HuDBVDPPTxeLPuWi6WOEOqxwWARHogYCETA/w8koqMZMXeZUrAlUmoLpnTSAuXJwbHnVpnC6WtV8qTwREQARCISAhE8qR6MAPCZkOYCmqCIiACBSEgBcujLo4N1K4xEddarVaQWokN00IREAEUiUgIZMq3nQyl5BJh6tyFQEREIEsCSBcmB6GcGGqGMY2Pki4QEEmAiJQRQKd1FlCphNagcRFyGzdujUQb+SGCIiACIhAuwQQL164HH744XbOOedESTVdLMKgPyIgAiLQEQEJmY5whRF59uzZ5u/aheFRGbxQHURABEQgeQJx4eKcposlT1g5ioAIVJmAhExBjz4Xx4K6LrdFQATKQkD12IsAbbMfcXFOwmUvQAoQAREQgQQJSMgkCDOrrJhallVZKkcEREAERKA1AS9cEC/ODQsXfc+lNbOq71H9RUAEkiMgIZMcy8xyQshw8cQyK1QFiYAIiIAIGO0uogVzbli4gCb+gL6+5wIRmQiIgAgkQqBlJhIyLdGEvaNWq5mekwn7GMk7ERCB4hNAuKxevdoQLjycP2fOnEjMULOXXnrJMB7UxwiTiYAIiIAIZEdAQiY71omWxEVzzZo1ieapzBoIaFMERKByBBAuAwMDkXBBtGC+rb3jjjsi4cKSNpjR8coBUoVFQAREICACEjIBHYxOXOECysUW6ySd4oqACIhAmgSKmDfihREXRAujLq1eiVzbMxJexPrJZxEQAREoKwEJmYIeWYQMc7AHBwcLWgO5LQIiIALZE0C0cAOov7/fEC7ODT/nwihL/DkXCZfsj09FS1S1RUAEuiQgIdMluBCS8VYcLsgh+CIfREAERCBEAo3CpXHERcIlxKMmn0RABERgLAL1/RIydQ6F/MuoDEIGK2QF5LQIiIAIJExAwiVhoMpOBERABAImICET8MEZyzWEjKaXjUUp2f3KTQREICwCCJfVq+tvFWOqGMa0MbzUVDEoyERABESgvAQkZAp+bJlexkW84NWQ+yIgAuUlkGjN4sKFaWIIF/9WMYQLr0N++OGHjXU945IoemUmAiIgAsERkJAJ7pB05hAXakZmBgYGOkuo2CIgAiJQAAIIF0ZYMOfqD+Z74cJrkCVcCnAQ5WIXBJREBESgHQISMu1QCjwOdx7960IDd1XuiYAIiMCoBJoJF8JIhGjBNOICDZkIiIAIiMAIISMcxSTAqAwmMVPM4yevRaDKBBApjLZgztVHXDwPRAvGyAs3bBh99vu0FAEREAEREAEJmZL8BrjIM70MK0mVilIN+SkCItABAS9ceLbFuZHCJf4qZNo0CZcOwCqqCIiACFSQgIRMSQ46F3wu/NzVLEmVVA0REIGCE0C08DIS2qWRwsWih/EbhUvBqyv3RUAEREAEMiYgIZMx8DSLY3oZ+dNxYCkTAREQgSwJIFwQLZhz9dEW/2A+N1riwsW3V1n6p7JEoNAE5LwIiMBeBCRk9kJS3ABGZZhLTieiuLWQ5yIgAkUggGhhKivtTeNoC/7zbAumB/OhIRMBERABEUiDwFhCJo0ylWeKBBAzixYtMj34nyJkZS0CFSSAcEG0YAgXzLczjaMtbNMWVRCTqiwCIiACIpAhAQmZDGFnVRQfyfSdjqzKVDlxAloXgeIT8G0IgsW5+jQxwqgZQoXRFoz1Wq1GsEwEREAEREAEMiUgIZMp7mwK404oU8z8tI9sSlUpIiACRSWAQPHtRVy4UB9Ein+2hXYlNeFCYTIREAEREAER6ICAhEwHsIoU1YsZHvxnKkiRfJevIiAC6RJAuNAuYM7VR1tYp1SEihcurGOEy0RABMIjII9EoOoEJGRK/AtAzPCgrcRMiQ+yqiYCYxCIi5bG0RaS0kYwRYwlooURGMJlIiACIiACIhA6gS6ETOhVkn9xAhIzcRpaF4FyE0C0+BsXcdFCODVHqDSOtki4QEYmAiIgAiJQRAISMkU8ah367MWMnwPfYXJFT4KA8hCBhAkgTvw5HRctzb7bomdbEoav7ERABERABIIgICETxGFI3wnEDJ0Zf7c2/RJVggiIQJIEEC48x4I51/q5ljJNEUuSn/ISAREQAREoHwEJmfId05Y1QszQySHC4YcfbnSMWJeJgAiERYBzE8GCxUdbvJecx3quxdPQUgREIEZAqyJQKQISMpU63GaIGebJL1q0yOgg0VGqGAJVVwSCIoBo8SOlnJPO1UdbCMdRzlc91wIJmQiIgAiIgAiMJJCMkBmZp7YKQIDOEXd1fQeqAC7LRREoNAGECcbNAywuWgYHB6O6cV560cJUULZrtVq0T39EQAREQAREQARGEpCQGcmjUluMziBmqDRTzXhwmHVZNgRUSnkJIFg4n/r7++2cc84xzi9szpw5Q5VGpEi0DOHQmGZG7wAAEABJREFUigiIgAiIgAh0TEBCpmNk5UqAmKFDxd1fOl1YuWqo2ohAugQQLX5kE6HiXH1qmD+XZs+ebZxfXrRwvmEaaenquCiRCIiACIiACAwRkJAZQlHtFTpVdLagwJ1j3wljWyYCIlAngGjh3MDioqXZK48Z7USw8Dwa51c9B/0VAREQgawJqDwRKC8BCZnyHtuOa+ZHZ+iAkViCBgqyKhJAsPipYY2ihX0wQaT4URbOGbYlWCAjEwEREAEREIFsCKQmZLJxX6WkQUCCJg2qyjNEAoiSVoIFAeN9RqR40cLIJdsSLZ6OliIgAiIgAiKQDwEJmXy4F6JUCZrMD5MKTIlAN4LFj7JItKR0UJStCIiACIiACPRIQEKmR4BVSN5M0PAmJu5kV6H+qmNxCEiwFOdYJeepchIBERABEagqAQmZqh75LuodFzS8iYmpNzxHI1HTBUwl6YmABEtP+JRYBESg6gRUfxEoCQEJmZIcyCyrgaDhTUxMvcHY9qKGpUZqsjwa5S3LixX/amMEc/xNYfzWfO2Z/uWfYeE3yTam51g8IS1FQAREQAREoHwEshQy5aOnGhkihg4jnUcMJHQwGalhKVEDEVkrAl6s8FvBGsUK24ODg1FyRgH5rUmwRDj0RwREQAREQAQqT0BCpvI/geQANBM1dEQRNSy5sy5h0ynv4sdvFCuMqmDOOWPZ399vxKGmjWLlpZdeMv+WMEYBNcICJZkIiIAIiIAIiAAEJGSgIEucgBc1dEQZqaGDyp11Oq0IG9+BlbBJHH3mGb788svGcUSocnwxji/H2blhseIdY1QFi4+sSKx4OlomQkCZiIAIiIAIVIKAhEwlDnO+lUTUcDedziqiBqMji1d0ep1zRqdXozYQCcuaiRSOE0LFOWfODQsVhKr3nuPL8Y6LFcKwWq1mtT3m42opAiIgAiKQPwF5IAJFJCAhU8SjVnCfETZ0ZOnUImoaR23oKCNsMNYx7vZz15+OdcGrH4z7sIQpbBGUGKzHEimMrnHsECmYP34IF8Ixji8WTGXliAiIgAiIgAiIQOkI5CxkSsdTFeqCAMIG86M2vmOMyKHTjHG3n442nWwEDsY6HW864nTIsS6KL0USRAkGA3hg8MJgBCsMbs61HkmBNUIEgYL5YxEXKRwniZRS/GxUCREQAREQAREoNAEJmUIfvvI6j7DB6DRjdKQRNr5jzTodbjreXuTQYXeuPk2NDjtG551wb3Ts6eRjdPoxBEAeJCk3bvgSN3z0ht8Y9ZgzZ070kDz1c25YlFBX4sAD83WCEawwuCFQMM8StuzDYC2R4slpKQIiIAIiIAIiEDIBCZmQj458a0oAgYPR4abjTUecDjodczrorHujc05H3hsZ0snH6PRjCADn6oIAcdCOkaZdi+fnXL0c5+qCK54HvsQNH73hN0Y9qBNGHakvRt0xwuCBEQeDEawwuJGPTASqRkD1FQEREAERKB8BCZnyHdPK14jOujc673TkvdGxp5OP0enHEACIAYztdox82rV4fpQRN8r2Fo/HOj5682VRD+qEUcfKH2wBEAEREAERSIuA8hWB4AlIyAR/iORglgQQB+0YQqJdi+eXZV1UlgiIgAiIgAiIgAiUmUB4QqbMtFU3ERABERABERABERABERCBRAhIyCSCUZmIQL4EVLoIiIAIiIAIiIAIVI2AhEzVjrjqKwIiIAIiAAGZCIiACIhAwQlIyBT8AMp9ERABERABERABEciGgEoRgbAISMiEdTzkjQiIgAiIQA8EeCtgD8mVVAREQAREoEAECiFkCsRTroqACIiACKREAJHy+OOP22233WZLly61U0891Y4//nibMmWKjR8/3vbZZ58hY5tw9hOP+KQjPfmk5KKyFQEREAERyJDAuAzLUlEiIALZEVBJIlAKAk899ZR973vfs7lz59qHP/xhO//8823jxo120EEH2dlnn22rVq2yxx57zF599VX705/+ZO+//360ZJtw9hOP+KQj/Yf35EN+5Ev+pQClSoiACIhABQlIyFTwoKvKIiACIhAygeeee876+vps2rRptnDhQnvttddsyZIl9sYbb9jTTz9tfCj20ksvta985StDIzL7779/NBpDvRiZYduPyBCP+KQjPfmQH/mSP+VQ3nPPPUdymQiIgAiIQEEISMgU5EDJTREQAREoO4GHHnoomi42a9Ys27VrVyRYfvOb39iNN95o8+bNM8RJEgzIh/zIl/wROJRHuUxDw48kylEeIlAJAqqkCORIQEImR/gqWgREQAREwOyBBx6wOXPm2OWXXx4JFkZKEBknnHBCJngoh/IoF4GDH/iDX5k4oEJEQAREQAS6IlBUIdNVZZVIBERABEQgHAKbN2+ORmAQDueee240bey8887L1UHKZ/oZ/uAXIzT4matTKlwEREAERKApAQmZplgUKAJlJKA6iUA4BHgm5cQTT7QvfelL9stf/tLOPPPMcJzb4wn+4Bf+4Sf+7gnWfxEQAREQgYAISMgEdDDkigiIgAiUnQBvCUMY8GwKdskllwRd5Uv2+IefGH7jf9AOyzkREAERqBABCZkKHWxVVQREQATyJMB3XPiuy1lnnWXr1q2Lvv+Spz/tls3bz/AXv/GferSbVvFEoIoEVGcRyIqAhExWpFWOCIiACFSYwLe//W1bsWKF/eIXv7DFixcXkgR+4z/1oD6FrIScFgEREIESESiRkCnRUVFVREAERKBEBL7+9a8bU7P4QCUjGkWuGv5TD+pDvYpcF/kuAiIgAkUnICFT9CMo/0WgFwJKKwIpE+CtXx/60Ids/fr1iX0HJmWXx8ye79BQH+pF/cZMoAgiIAIiIAKpEJCQSQWrMhUBERABEaCTf9BBB0UftiwTDV+XO+64w6gf9fRhWoqACIiACGRHQEImO9YqSQREQAQqQ4BpVxMnTrRVq1aVus7Uj3pS31JXVJUTgd4IKLUIpEJAQiYVrMpUBERABKpLgAfhd+3aVdqRmMYjy8gM9aXejfu0LQIiIAIikB6BcguZ9LgpZxEQAREQgSYEeDXxQw89ZGvXrm2yt7xB1Jd6U//y1lI1EwEREIGwCEjIhHU85I0I5E5ADohAtwT4WOT5559vt99+e2ke7G+XBS8AoN7UHw7tplM8ERABERCB7glIyHTPTilFQAREQARiBC666CJbuXKl8YriWHAVVqM6Um/qD4coQH9EQAREQARSJSAhkypeZS4CIiAC1SDQ19dnhx56aGE/dpnUUeKjmXCAR1J5Kh8RKCcB1UoEeicgIdM7Q+UgAiIgApUmsHnzZlu2bFlklQbxQeU9C7h8EKSFCIiACIhACgQqJ2RSYKgsRUAERKDSBK6++mq79tprbcqUKZXm4CsPB3jAxYdpKQIiIAIikDwBCZnkmSpHESgbAdVHBFoSeOCBB+zFF1+0Sy65pGWcKu6AB1zgU8X6q84iIAIikAUBCZksKKsMERABESgpgR/+8Id22WWXlbR2vVTLIi7w6S0XpRYBERABEWhFQEKmFRmFi4AIiIAIjEqA76bs3LnTzjzzzFHjVXUnXOADp6oyUL1FoCMCiiwCHRKQkOkQmKKLgAiIgAjUCaxatcouuOCC+ob+NiUAHzg13alAERABERCBnghIyJj1BFCJRUAERKCKBJ577jnbuHGjnXfeeVWsftt1hg+c4NV2IkUUAREQARFoi4CETFuYFEkERGAkAW1VncDatWvta1/7WtUxtFV/OMGrrciKJAIiIAIi0DYBCZm2USmiCIiACIiAJ/DTn/7UFixY4De1HIUAnOBlo8TRLhEQAREQgc4JSMh0zkwpREAERKDSBJ566il7//337YQTTkiUw5NPPmn77befOedG2GGHHWavvvpq22URlzT3339/22mIePbZZ0fl1mo1e+uttwhKxOAEL7glkqEyEYEKEVBVRWA0AhIyo9HRPhEQAREQgb0I8Bauk08+ea/wJAImTZpkTzzxhP35z38esq1bt9rkyZOTyL5lHgiXXbt2RWUPDAzY/vvv3zJuNzvgBbdu0iqNCIiACIhAcwISMk25KFAEREAERKAVgZ/97Gc2d+7cVrtTC2eEZcaMGTZjjznnotEbRnEoECHCSIpzzs444wyCmlo8nnPOyJOwv/u7v7N/+7d/s+OPPz4Ka5q4h0B4wa2HLJRUBERABESggYCETAMQbYqACHRJQMkqQYCRkkcffdRmz56dS303b95s1157re3evdtmzpxpK1asiPy48MIL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)
# ## **:תרשים טיפול - אירוע נטישת תור**
# ![QueueAbandonmentEvent.drawio.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAh0AAAM7CAYAAADu4iJ4AABEsnRFWHRteGZpbGUAJTNDbXhmaWxlJTIwaG9zdCUzRCUyMmFwcC5kaWFncmFtcy5uZXQlMjIlMjBhZ2VudCUzRCUyMk1vemlsbGElMkY1LjAlMjAoV2luZG93cyUyME5UJTIwMTAuMCUzQiUyMFdpbjY0JTNCJTIweDY0KSUyMEFwcGxlV2ViS2l0JTJGNTM3LjM2JTIwKEtIVE1MJTJDJTIwbGlrZSUyMEdlY2tvKSUyMENocm9tZSUyRjE0My4wLjAuMCUyMFNhZmFyaSUyRjUzNy4zNiUyMiUyMHZlcnNpb24lM0QlMjIyOS4yLjklMjIlMjBzY2FsZSUzRCUyMjElMjIlMjBib3JkZXIlM0QlMjIwJTIyJTNFJTBBJTIwJTIwJTNDZGlhZ3JhbSUyMG5hbWUlM0QlMjJQYWdlLTElMjIlMjBpZCUzRCUyMm10WWxjOWY4dDgtSHJjLVZIUXowJTIyJTNFJTBBJTIwJTIwJTIwJTIwJTNDbXhHcmFwaE1vZGVsJTIwZHglM0QlMjIxMjIwJTIyJTIwZHklM0QlMjI2MjYlMjIlMjBncmlkJTNEJTIyMSUyMiUyMGdyaWRTaXplJTNEJTIyMTAlMjIlMjBndWlkZXMlM0QlMjIxJTIyJTIwdG9vbHRpcHMlM0QlMjIxJTIyJTIwY29ubmVjdCUzRCUyMjElMjIlMjBhcnJvd3MlM0QlMjIxJTIyJTIwZm9sZCUzRCUyMjElMjIlMjBwYWdlJTNEJTIyMSUyMiUyMHBhZ2VTY2FsZSUzRCUyMjElMjIlMjBwYWdlV2lkdGglM0QlMjI4NTAlMjIlMjBwYWdlSGVpZ2h0JTNEJTIyMTEwMCUyMiUyMG1hdGglM0QlMjIwJTIyJTIwc2hhZG93JTNEJTIyMCUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUzQ3Jvb3QlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteENlbGwlMjBpZCUzRCUyMjAlMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteENlbGwlMjBpZCUzRCUyMjElMjIlMjBwYXJlbnQlM0QlMjIwJTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhDZWxsJTIwaWQlM0QlMjJDS1JNWUl3VmFmYl9OVG9xSWh5dy01JTIyJTIwZWRnZSUzRCUyMjElMjIlMjBwYXJlbnQlM0QlMjIxJTIyJTIwc291cmNlJTNEJTIyNHVBcXk0cTlLdWtZNmoxV1lzcFItMSUyMiUyMHN0eWxlJTNEJTIyZWRnZVN0eWxlJTNEb3J0aG9nb25hbEVkZ2VTdHlsZSUzQnJvdW5kZWQlM0QwJTNCb3J0aG9nb25hbExvb3AlM0QxJTNCamV0dHlTaXplJTNEYXV0byUzQmh0bWwlM0QxJTNCZXhpdFglM0QwLjUlM0JleGl0WSUzRDElM0JleGl0RHglM0QwJTNCZXhpdER5JTNEMCUzQmVudHJ5WCUzRDAuNSUzQmVudHJ5WSUzRDAlM0JlbnRyeUR4JTNEMCUzQmVudHJ5RHklM0QwJTNCJTIyJTIwdGFyZ2V0JTNEJTIyQ0tSTVlJd1ZhZmJfTlRvcUloeXctMSUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214R2VvbWV0cnklMjByZWxhdGl2ZSUzRCUyMjElMjIlMjBhcyUzRCUyMmdlb21ldHJ5JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGbXhDZWxsJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhDZWxsJTIwaWQlM0QlMjI0dUFxeTRxOUt1a1k2ajFXWXNwUi0xJTIyJTIwcGFyZW50JTNEJTIyMSUyMiUyMHN0eWxlJTNEJTIycm91bmRlZCUzRDElM0J3aGl0ZVNwYWNlJTNEd3JhcCUzQmh0bWwlM0QxJTNCJTIyJTIwdmFsdWUlM0QlMjIlRDclOTQlRDclQTElRDclQTglMjAlRDclOTAlRDclQUElMjAlRDclOTQlRDclOUUlRDclOTElRDclQTclRDclQTglMjAlRDclOUUlRDclOTQlRDclQUElRDclOTUlRDclQTglMjIlMjB2ZXJ0ZXglM0QlMjIxJTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhHZW9tZXRyeSUyMGhlaWdodCUzRCUyMjYwJTIyJTIwd2lkdGglM0QlMjIxMjAlMjIlMjB4JTNEJTIyMzUwJTIyJTIweSUzRCUyMjE4MCUyMiUyMGFzJTNEJTIyZ2VvbWV0cnklMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZteENlbGwlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteENlbGwlMjBpZCUzRCUyMkNLUk1ZSXdWYWZiX05Ub3FJaHl3LTElMjIlMjBwYXJlbnQlM0QlMjIxJTIyJTIwc3R5bGUlM0QlMjJyb3VuZGVkJTNEMSUzQndoaXRlU3BhY2UlM0R3cmFwJTNCaHRtbCUzRDElM0IlMjIlMjB2YWx1ZSUzRCUyMiVENyU5NCVENyU5NSVENyVBOCVENyU5MyUyMCVENyU5MyVENyU5OSVENyVBOCVENyU5NSVENyU5MiUyMCVENyVBNCVENyU5MCVENyVBOCVENyVBNyUyMCVENyVBMiVENyU5MSVENyU5NSVENyVBOCUyNmFtcCUzQm5ic3AlM0IlMjAlRDclOUUlRDclOTElRDclQTclRDclQTglMjAlRDclOTEtMC44JTIyJTIwdmVydGV4JTNEJTIyMSUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214R2VvbWV0cnklMjBoZWlnaHQlM0QlMjI2MCUyMiUyMHdpZHRoJTNEJTIyMTIwJTIyJTIweCUzRCUyMjM1MCUyMiUyMHklM0QlMjIyNzAlMjIlMjBhcyUzRCUyMmdlb21ldHJ5JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGbXhDZWxsJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhDZWxsJTIwaWQlM0QlMjJDS1JNWUl3VmFmYl9OVG9xSWh5dy0yJTIyJTIwcGFyZW50JTNEJTIyMSUyMiUyMHN0eWxlJTNEJTIycm91bmRlZCUzRDElM0J3aGl0ZVNwYWNlJTNEd3JhcCUzQmh0bWwlM0QxJTNCJTIyJTIwdmFsdWUlM0QlMjIlRDclOTQlRDclOTUlRDclQTElRDclQTMlMjAxJTIwJUQ3JTlDJUQ3JTlFJUQ3JUE5JUQ3JUFBJUQ3JUEwJUQ3JTk0JTIwJUQ3JTk0JUQ3JUExJUQ3JTk1JUQ3JTlCJUQ3JTlEJTIwJUQ3JTkwJUQ3JUFBJTIwJUQ3JUExJUQ3JTlBJTIwJUQ3JTk0JUQ3JUEwJUQ3JTk4JUQ3JTk5JUQ3JUE5JUQ3JTk1JUQ3JUFBJTIwJUQ3JTlFJUQ3JTk0JUQ3JUFBJUQ3JTk1JUQ3JUE4JUQ3JTk5JUQ3JTlEJTIyJTIwdmVydGV4JTNEJTIyMSUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214R2VvbWV0cnklMjBoZWlnaHQlM0QlMjI2MCUyMiUyMHdpZHRoJTNEJTIyMTIwJTIyJTIweCUzRCUyMjM1MCUyMiUyMHklM0QlMjIzNjAlMjIlMjBhcyUzRCUyMmdlb21ldHJ5JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGbXhDZWxsJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhDZWxsJTIwaWQlM0QlMjJDS1JNWUl3VmFmYl9OVG9xSWh5dy0zJTIyJTIwcGFyZW50JTNEJTIyMSUyMiUyMHN0eWxlJTNEJTIycmhvbWJ1cyUzQndoaXRlU3BhY2UlM0R3cmFwJTNCaHRtbCUzRDElM0IlMjIlMjB2YWx1ZSUzRCUyMiVENyU5NCVENyU5MCVENyU5RCUyMCVENyU5RSVENyU5MyVENyU5NSVENyU5MSVENyVBOCUyMCVENyU5MSVENyU5MSVENyU5RiUyMCVENyVBMCVENyU5NSVENyVBMiVENyVBOCUyMiUyMHZlcnRleCUzRCUyMjElMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteEdlb21ldHJ5JTIwaGVpZ2h0JTNEJTIyODAlMjIlMjB3aWR0aCUzRCUyMjE4MCUyMiUyMHglM0QlMjIzMjAlMjIlMjB5JTNEJTIyNTM1JTIyJTIwYXMlM0QlMjJnZW9tZXRyeSUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14Q2VsbCUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214Q2VsbCUyMGlkJTNEJTIyQ0tSTVlJd1ZhZmJfTlRvcUloeXctNiUyMiUyMGVkZ2UlM0QlMjIxJTIyJTIwcGFyZW50JTNEJTIyMSUyMiUyMHNvdXJjZSUzRCUyMkNLUk1ZSXdWYWZiX05Ub3FJaHl3LTElMjIlMjBzdHlsZSUzRCUyMmVuZEFycm93JTNEY2xhc3NpYyUzQmh0bWwlM0QxJTNCcm91bmRlZCUzRDAlM0JleGl0WCUzRDAuNSUzQmV4aXRZJTNEMSUzQmV4aXREeCUzRDAlM0JleGl0RHklM0QwJTNCZW50cnlYJTNEMC41JTNCZW50cnlZJTNEMCUzQmVudHJ5RHglM0QwJTNCZW50cnlEeSUzRDAlM0IlMjIlMjB0YXJnZXQlM0QlMjJDS1JNWUl3VmFmYl9OVG9xSWh5dy0yJTIyJTIwdmFsdWUlM0QlMjIlMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteEdlb21ldHJ5JTIwaGVpZ2h0JTNEJTIyNTAlMjIlMjByZWxhdGl2ZSUzRCUyMjElMjIlMjB3aWR0aCUzRCUyMjUwJTIyJTIwYXMlM0QlMjJnZW9tZXRyeSUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyMzYwJTIyJTIweSUzRCUyMjQ4MCUyMiUyMGFzJTNEJTIyc291cmNlUG9pbnQlMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteFBvaW50JTIweCUzRCUyMjQxMCUyMiUyMHklM0QlMjIzNjAlMjIlMjBhcyUzRCUyMnRhcmdldFBvaW50JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGbXhHZW9tZXRyeSUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14Q2VsbCUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214Q2VsbCUyMGlkJTNEJTIyQ0tSTVlJd1ZhZmJfTlRvcUloeXctNyUyMiUyMHBhcmVudCUzRCUyMjElMjIlMjBzdHlsZSUzRCUyMnJvdW5kZWQlM0QxJTNCd2hpdGVTcGFjZSUzRHdyYXAlM0JodG1sJTNEMSUzQiUyMiUyMHZhbHVlJTNEJTIyJUQ3JTkzJUQ3JTkyJUQ3JTk1JUQ3JTlEJTIwdX5VKDAlMkMxKSUyMiUyMHZlcnRleCUzRCUyMjElMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteEdlb21ldHJ5JTIwaGVpZ2h0JTNEJTIyNjAlMjIlMjB3aWR0aCUzRCUyMjEyMCUyMiUyMHglM0QlMjIyMDAlMjIlMjB5JTNEJTIyNjM1JTIyJTIwYXMlM0QlMjJnZW9tZXRyeSUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14Q2VsbCUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214Q2VsbCUyMGlkJTNEJTIyQ0tSTVlJd1ZhZmJfTlRvcUloeXctOCUyMiUyMHBhcmVudCUzRCUyMjElMjIlMjBzdHlsZSUzRCUyMnJob21idXMlM0J3aGl0ZVNwYWNlJTNEd3JhcCUzQmh0bWwlM0QxJTNCJTIyJTIwdmFsdWUlM0QlMjIwJTI2YW1wJTNCbHQlM0J1JTI2YW1wJTNCbHQlM0IwLjYlMjIlMjB2ZXJ0ZXglM0QlMjIxJTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhHZW9tZXRyeSUyMGhlaWdodCUzRCUyMjgwJTIyJTIwd2lkdGglM0QlMjIxODAlMjIlMjB4JTNEJTIyMTcwJTIyJTIweSUzRCUyMjcyNSUyMiUyMGFzJTNEJTIyZ2VvbWV0cnklMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZteENlbGwlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteENlbGwlMjBpZCUzRCUyMkNLUk1ZSXdWYWZiX05Ub3FJaHl3LTklMjIlMjBwYXJlbnQlM0QlMjIxJTIyJTIwc3R5bGUlM0QlMjJyb3VuZGVkJTNEMSUzQndoaXRlU3BhY2UlM0R3cmFwJTNCaHRtbCUzRDElM0IlMjIlMjB2YWx1ZSUzRCUyMiVENyU5NCVENyU5QiVENyVBMCVENyVBMSUyMCVENyU5QyVENyU5OSVENyU5NSVENyU5RSVENyU5RiUyMCVENyVBNyVENyVBMCVENyU5OSVENyU5OSVENyVBQSUyMCVENyVBNiVENyU5RSVENyU5OSVENyU5MyUyMCVENyU5MCVENyVBNyVENyVBMSVENyVBNCVENyVBOCVENyVBMSUyMCVENyU5NSVENyU5QyVENyU5MCVENyU5NyVENyVBOCUyMCVENyU5RSVENyU5QiVENyU5RiUyMCVENyU5NyVENyU5NiVENyVBOCVENyU5NCUyMCVENyU5QyVENyVBQSVENyU5NSVENyVBOCUyMCVENyU5NCVENyU5RSVENyU5NCVENyU5OSVENyVBOCUyMCVENyVBOSVENyU5QyUyMCVENyU5NCVENyU5RSVENyVBQSVENyVBNyVENyU5RiUyMiUyMHZlcnRleCUzRCUyMjElMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteEdlb21ldHJ5JTIwaGVpZ2h0JTNEJTIyNjAlMjIlMjB3aWR0aCUzRCUyMjEyMCUyMiUyMHglM0QlMjI1MCUyMiUyMHklM0QlMjI4MjUlMjIlMjBhcyUzRCUyMmdlb21ldHJ5JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGbXhDZWxsJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhDZWxsJTIwaWQlM0QlMjJDS1JNWUl3VmFmYl9OVG9xSWh5dy0xMCUyMiUyMGVkZ2UlM0QlMjIxJTIyJTIwcGFyZW50JTNEJTIyMSUyMiUyMHNvdXJjZSUzRCUyMkNLUk1ZSXdWYWZiX05Ub3FJaHl3LTMlMjIlMjBzdHlsZSUzRCUyMmVuZEFycm93JTNEY2xhc3NpYyUzQmh0bWwlM0QxJTNCcm91bmRlZCUzRDAlM0JleGl0WCUzRDAlM0JleGl0WSUzRDAuNSUzQmV4aXREeCUzRDAlM0JleGl0RHklM0QwJTNCZW50cnlYJTNEMC41JTNCZW50cnlZJTNEMCUzQmVudHJ5RHglM0QwJTNCZW50cnlEeSUzRDAlM0IlMjIlMjB0YXJnZXQlM0QlMjJDS1JNWUl3VmFmYl9OVG9xSWh5dy03JTIyJTIwdmFsdWUlM0QlMjIlMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteEdlb21ldHJ5JTIwaGVpZ2h0JTNEJTIyNTAlMjIlMjByZWxhdGl2ZSUzRCUyMjElMjIlMjB3aWR0aCUzRCUyMjUwJTIyJTIwYXMlM0QlMjJnZW9tZXRyeSUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ0FycmF5JTIwYXMlM0QlMjJwb2ludHMlMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteFBvaW50JTIweCUzRCUyMjI2MCUyMiUyMHklM0QlMjI1NzUlMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZBcnJheSUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyMjMwJTIyJTIweSUzRCUyMjY0NSUyMiUyMGFzJTNEJTIyc291cmNlUG9pbnQlMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteFBvaW50JTIweCUzRCUyMjI4MCUyMiUyMHklM0QlMjI1OTUlMjIlMjBhcyUzRCUyMnRhcmdldFBvaW50JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGbXhHZW9tZXRyeSUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14Q2VsbCUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214Q2VsbCUyMGlkJTNEJTIyQ0tSTVlJd1ZhZmJfTlRvcUloeXctMTElMjIlMjBjb25uZWN0YWJsZSUzRCUyMjAlMjIlMjBwYXJlbnQlM0QlMjJDS1JNWUl3VmFmYl9OVG9xSWh5dy0xMCUyMiUyMHN0eWxlJTNEJTIyZWRnZUxhYmVsJTNCaHRtbCUzRDElM0JhbGlnbiUzRGNlbnRlciUzQnZlcnRpY2FsQWxpZ24lM0RtaWRkbGUlM0JyZXNpemFibGUlM0QwJTNCcG9pbnRzJTNEJTVCJTVEJTNCJTIyJTIwdmFsdWUlM0QlMjIlRDclOUIlRDclOUYlMjIlMjB2ZXJ0ZXglM0QlMjIxJTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhHZW9tZXRyeSUyMHJlbGF0aXZlJTNEJTIyMSUyMiUyMHglM0QlMjItMC40OTE3JTIyJTIwYXMlM0QlMjJnZW9tZXRyeSUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyLTklMjIlMjB5JTNEJTIyLTEwJTIyJTIwYXMlM0QlMjJvZmZzZXQlMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZteEdlb21ldHJ5JTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGbXhDZWxsJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhDZWxsJTIwaWQlM0QlMjJDS1JNWUl3VmFmYl9OVG9xSWh5dy0xMiUyMiUyMGVkZ2UlM0QlMjIxJTIyJTIwcGFyZW50JTNEJTIyMSUyMiUyMHNvdXJjZSUzRCUyMkNLUk1ZSXdWYWZiX05Ub3FJaHl3LTclMjIlMjBzdHlsZSUzRCUyMmVuZEFycm93JTNEY2xhc3NpYyUzQmh0bWwlM0QxJTNCcm91bmRlZCUzRDAlM0JleGl0WCUzRDAuNSUzQmV4aXRZJTNEMSUzQmV4aXREeCUzRDAlM0JleGl0RHklM0QwJTNCZW50cnlYJTNEMC41JTNCZW50cnlZJTNEMCUzQmVudHJ5RHglM0QwJTNCZW50cnlEeSUzRDAlM0IlMjIlMjB0YXJnZXQlM0QlMjJDS1JNWUl3VmFmYl9OVG9xSWh5dy04JTIyJTIwdmFsdWUlM0QlMjIlMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteEdlb21ldHJ5JTIwaGVpZ2h0JTNEJTIyNTAlMjIlMjByZWxhdGl2ZSUzRCUyMjElMjIlMjB3aWR0aCUzRCUyMjUwJTIyJTIwYXMlM0QlMjJnZW9tZXRyeSUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyMzMwJTIyJTIweSUzRCUyMjcyNSUyMiUyMGFzJTNEJTIyc291cmNlUG9pbnQlMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteFBvaW50JTIweCUzRCUyMjM4MCUyMiUyMHklM0QlMjI2NzUlMjIlMjBhcyUzRCUyMnRhcmdldFBvaW50JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGbXhHZW9tZXRyeSUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14Q2VsbCUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214Q2VsbCUyMGlkJTNEJTIyQ0tSTVlJd1ZhZmJfTlRvcUloeXctMTMlMjIlMjBlZGdlJTNEJTIyMSUyMiUyMHBhcmVudCUzRCUyMjElMjIlMjBzb3VyY2UlM0QlMjJDS1JNWUl3VmFmYl9OVG9xSWh5dy04JTIyJTIwc3R5bGUlM0QlMjJlbmRBcnJvdyUzRGNsYXNzaWMlM0JodG1sJTNEMSUzQnJvdW5kZWQlM0QwJTNCZXhpdFglM0QwJTNCZXhpdFklM0QwLjUlM0JleGl0RHglM0QwJTNCZXhpdER5JTNEMCUzQmVudHJ5WCUzRDAuNSUzQmVudHJ5WSUzRDAlM0JlbnRyeUR4JTNEMCUzQmVudHJ5RHklM0QwJTNCJTIyJTIwdGFyZ2V0JTNEJTIyQ0tSTVlJd1ZhZmJfTlRvcUloeXctOSUyMiUyMHZhbHVlJTNEJTIyJTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhHZW9tZXRyeSUyMGhlaWdodCUzRCUyMjUwJTIyJTIwcmVsYXRpdmUlM0QlMjIxJTIyJTIwd2lkdGglM0QlMjI1MCUyMiUyMGFzJTNEJTIyZ2VvbWV0cnklMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NBcnJheSUyMGFzJTNEJTIycG9pbnRzJTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhQb2ludCUyMHglM0QlMjIxMTAlMjIlMjB5JTNEJTIyNzY1JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGQXJyYXklM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteFBvaW50JTIweCUzRCUyMjEwMCUyMiUyMHklM0QlMjI2OTUlMjIlMjBhcyUzRCUyMnNvdXJjZVBvaW50JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhQb2ludCUyMHglM0QlMjIxNTAlMjIlMjB5JTNEJTIyNjQ1JTIyJTIwYXMlM0QlMjJ0YXJnZXRQb2ludCUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14R2VvbWV0cnklM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZteENlbGwlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteENlbGwlMjBpZCUzRCUyMkNLUk1ZSXdWYWZiX05Ub3FJaHl3LTIxJTIyJTIwY29ubmVjdGFibGUlM0QlMjIwJTIyJTIwcGFyZW50JTNEJTIyQ0tSTVlJd1ZhZmJfTlRvcUloeXctMTMlMjIlMjBzdHlsZSUzRCUyMmVkZ2VMYWJlbCUzQmh0bWwlM0QxJTNCYWxpZ24lM0RjZW50ZXIlM0J2ZXJ0aWNhbEFsaWduJTNEbWlkZGxlJTNCcmVzaXphYmxlJTNEMCUzQnBvaW50cyUzRCU1QiU1RCUzQiUyMiUyMHZhbHVlJTNEJTIyJUQ3JTlCJUQ3JTlGJTIyJTIwdmVydGV4JTNEJTIyMSUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214R2VvbWV0cnklMjByZWxhdGl2ZSUzRCUyMjElMjIlMjB4JTNEJTIyLTAuNDQ0NCUyMiUyMHklM0QlMjIyJTIyJTIwYXMlM0QlMjJnZW9tZXRyeSUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB5JTNEJTIyLTEyJTIyJTIwYXMlM0QlMjJvZmZzZXQlMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZteEdlb21ldHJ5JTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGbXhDZWxsJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhDZWxsJTIwaWQlM0QlMjJDS1JNWUl3VmFmYl9OVG9xSWh5dy0xNCUyMiUyMHBhcmVudCUzRCUyMjElMjIlMjBzdHlsZSUzRCUyMnJvdW5kZWQlM0QxJTNCd2hpdGVTcGFjZSUzRHdyYXAlM0JodG1sJTNEMSUzQiUyMiUyMHZhbHVlJTNEJTIyJUQ3JTk0JUQ3JTlFJUQ3JUE5JUQ3JTlBJTIwJUQ3JTlDJUQ3JTkwJUQ3JTk5JUQ3JUE4JUQ3JTk1JUQ3JUEyJTIwJUQ3JTk0JUQ3JTkxJUQ3JTkwJTIwJUQ3JTlDJUQ3JUE0JUQ3JTk5JTIwJUQ3JTk5JUQ3JTk1JUQ3JTlFJUQ3JTlGJTIwJUQ3JTk0JUQ3JUE0JUQ3JUEyJUQ3JTk5JUQ3JTlDJUQ3JTk1JUQ3JUFBJTIwJUQ3JUE5JUQ3JTlDJTIwJUQ3JTk0JUQ3JTlFJUQ3JTkxJUQ3JUE3JUQ3JUE4JTIyJTIwdmVydGV4JTNEJTIyMSUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214R2VvbWV0cnklMjBoZWlnaHQlM0QlMjI2MCUyMiUyMHdpZHRoJTNEJTIyMTIwJTIyJTIweCUzRCUyMjQ3MCUyMiUyMHklM0QlMjI5NDUlMjIlMjBhcyUzRCUyMmdlb21ldHJ5JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGbXhDZWxsJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhDZWxsJTIwaWQlM0QlMjJDS1JNWUl3VmFmYl9OVG9xSWh5dy0xNSUyMiUyMGVkZ2UlM0QlMjIxJTIyJTIwcGFyZW50JTNEJTIyMSUyMiUyMHNvdXJjZSUzRCUyMkNLUk1ZSXdWYWZiX05Ub3FJaHl3LTklMjIlMjBzdHlsZSUzRCUyMmVuZEFycm93JTNEY2xhc3NpYyUzQmh0bWwlM0QxJTNCcm91bmRlZCUzRDAlM0JleGl0WCUzRDAuNSUzQmV4aXRZJTNEMSUzQmV4aXREeCUzRDAlM0JleGl0RHklM0QwJTNCZW50cnlYJTNEMC41JTNCZW50cnlZJTNEMCUzQmVudHJ5RHglM0QwJTNCZW50cnlEeSUzRDAlM0IlMjIlMjB0YXJnZXQlM0QlMjJDS1JNWUl3VmFmYl9OVG9xSWh5dy0xNCUyMiUyMHZhbHVlJTNEJTIyJTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhHZW9tZXRyeSUyMGhlaWdodCUzRCUyMjUwJTIyJTIwcmVsYXRpdmUlM0QlMjIxJTIyJTIwd2lkdGglM0QlMjI1MCUyMiUyMGFzJTNEJTIyZ2VvbWV0cnklMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NBcnJheSUyMGFzJTNEJTIycG9pbnRzJTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhQb2ludCUyMHglM0QlMjIxMTAlMjIlMjB5JTNEJTIyOTI1JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhQb2ludCUyMHglM0QlMjI1MzAlMjIlMjB5JTNEJTIyOTI1JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGQXJyYXklM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteFBvaW50JTIweCUzRCUyMjIyMCUyMiUyMHklM0QlMjI5MzUlMjIlMjBhcyUzRCUyMnNvdXJjZVBvaW50JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhQb2ludCUyMHglM0QlMjIyNzAlMjIlMjB5JTNEJTIyODg1JTIyJTIwYXMlM0QlMjJ0YXJnZXRQb2ludCUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14R2VvbWV0cnklM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZteENlbGwlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteENlbGwlMjBpZCUzRCUyMkNLUk1ZSXdWYWZiX05Ub3FJaHl3LTE2JTIyJTIwZWRnZSUzRCUyMjElMjIlMjBwYXJlbnQlM0QlMjIxJTIyJTIwc291cmNlJTNEJTIyQ0tSTVlJd1ZhZmJfTlRvcUloeXctOCUyMiUyMHN0eWxlJTNEJTIyZW5kQXJyb3clM0RjbGFzc2ljJTNCaHRtbCUzRDElM0Jyb3VuZGVkJTNEMCUzQmV4aXRYJTNEMSUzQmV4aXRZJTNEMC41JTNCZXhpdER4JTNEMCUzQmV4aXREeSUzRDAlM0JlbnRyeVglM0QwLjUlM0JlbnRyeVklM0QwJTNCZW50cnlEeCUzRDAlM0JlbnRyeUR5JTNEMCUzQiUyMiUyMHRhcmdldCUzRCUyMkNLUk1ZSXdWYWZiX05Ub3FJaHl3LTE0JTIyJTIwdmFsdWUlM0QlMjIlMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteEdlb21ldHJ5JTIwaGVpZ2h0JTNEJTIyNTAlMjIlMjByZWxhdGl2ZSUzRCUyMjElMjIlMjB3aWR0aCUzRCUyMjUwJTIyJTIwYXMlM0QlMjJnZW9tZXRyeSUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ0FycmF5JTIwYXMlM0QlMjJwb2ludHMlMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteFBvaW50JTIweCUzRCUyMjUzMCUyMiUyMHklM0QlMjI3NjUlMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZBcnJheSUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyMzgwJTIyJTIweSUzRCUyMjgzNSUyMiUyMGFzJTNEJTIyc291cmNlUG9pbnQlMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteFBvaW50JTIweCUzRCUyMjQzMCUyMiUyMHklM0QlMjI3ODUlMjIlMjBhcyUzRCUyMnRhcmdldFBvaW50JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGbXhHZW9tZXRyeSUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14Q2VsbCUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214Q2VsbCUyMGlkJTNEJTIyQ0tSTVlJd1ZhZmJfTlRvcUloeXctMTklMjIlMjBjb25uZWN0YWJsZSUzRCUyMjAlMjIlMjBwYXJlbnQlM0QlMjJDS1JNWUl3VmFmYl9OVG9xSWh5dy0xNiUyMiUyMHN0eWxlJTNEJTIyZWRnZUxhYmVsJTNCaHRtbCUzRDElM0JhbGlnbiUzRGNlbnRlciUzQnZlcnRpY2FsQWxpZ24lM0RtaWRkbGUlM0JyZXNpemFibGUlM0QwJTNCcG9pbnRzJTNEJTVCJTVEJTNCJTIyJTIwdmFsdWUlM0QlMjIlRDclOUMlRDclOTAlMjIlMjB2ZXJ0ZXglM0QlMjIxJTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhHZW9tZXRyeSUyMHJlbGF0aXZlJTNEJTIyMSUyMiUyMHglM0QlMjItMC44NTkzJTIyJTIweSUzRCUyMi0xJTIyJTIwYXMlM0QlMjJnZW9tZXRyeSUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyMjUlMjIlMjB5JTNEJTIyLTExJTIyJTIwYXMlM0QlMjJvZmZzZXQlMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZteEdlb21ldHJ5JTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGbXhDZWxsJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhDZWxsJTIwaWQlM0QlMjJDS1JNWUl3VmFmYl9OVG9xSWh5dy0xOCUyMiUyMGVkZ2UlM0QlMjIxJTIyJTIwcGFyZW50JTNEJTIyMSUyMiUyMHNvdXJjZSUzRCUyMkNLUk1ZSXdWYWZiX05Ub3FJaHl3LTMlMjIlMjBzdHlsZSUzRCUyMmVuZEFycm93JTNEY2xhc3NpYyUzQmh0bWwlM0QxJTNCcm91bmRlZCUzRDAlM0JleGl0WCUzRDElM0JleGl0WSUzRDAuNSUzQmV4aXREeCUzRDAlM0JleGl0RHklM0QwJTNCZW50cnlYJTNEMC41JTNCZW50cnlZJTNEMCUzQmVudHJ5RHglM0QwJTNCZW50cnlEeSUzRDAlM0IlMjIlMjB0YXJnZXQlM0QlMjJDS1JNWUl3VmFmYl9OVG9xSWh5dy0xNCUyMiUyMHZhbHVlJTNEJTIyJTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhHZW9tZXRyeSUyMGhlaWdodCUzRCUyMjUwJTIyJTIwcmVsYXRpdmUlM0QlMjIxJTIyJTIwd2lkdGglM0QlMjI1MCUyMiUyMGFzJTNEJTIyZ2VvbWV0cnklMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NBcnJheSUyMGFzJTNEJTIycG9pbnRzJTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhQb2ludCUyMHglM0QlMjI1MzAlMjIlMjB5JTNEJTIyNTc1JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGQXJyYXklM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteFBvaW50JTIweCUzRCUyMjQ5MCUyMiUyMHklM0QlMjI3MTUlMjIlMjBhcyUzRCUyMnNvdXJjZVBvaW50JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhQb2ludCUyMHglM0QlMjI1NDAlMjIlMjB5JTNEJTIyNjY1JTIyJTIwYXMlM0QlMjJ0YXJnZXRQb2ludCUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14R2VvbWV0cnklM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZteENlbGwlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteENlbGwlMjBpZCUzRCUyMkNLUk1ZSXdWYWZiX05Ub3FJaHl3LTIwJTIyJTIwY29ubmVjdGFibGUlM0QlMjIwJTIyJTIwcGFyZW50JTNEJTIyQ0tSTVlJd1ZhZmJfTlRvcUloeXctMTglMjIlMjBzdHlsZSUzRCUyMmVkZ2VMYWJlbCUzQmh0bWwlM0QxJTNCYWxpZ24lM0RjZW50ZXIlM0J2ZXJ0aWNhbEFsaWduJTNEbWlkZGxlJTNCcmVzaXphYmxlJTNEMCUzQnBvaW50cyUzRCU1QiU1RCUzQiUyMiUyMHZhbHVlJTNEJTIyJUQ3JTlDJUQ3JTkwJTIyJTIwdmVydGV4JTNEJTIyMSUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214R2VvbWV0cnklMjByZWxhdGl2ZSUzRCUyMjElMjIlMjB4JTNEJTIyLTAuOTA2NyUyMiUyMHklM0QlMjIxJTIyJTIwYXMlM0QlMjJnZW9tZXRyeSUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB5JTNEJTIyLTklMjIlMjBhcyUzRCUyMm9mZnNldCUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14R2VvbWV0cnklM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZteENlbGwlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteENlbGwlMjBpZCUzRCUyMkNLUk1ZSXdWYWZiX05Ub3FJaHl3LTIyJTIyJTIwcGFyZW50JTNEJTIyMSUyMiUyMHN0eWxlJTNEJTIycm91bmRlZCUzRDElM0J3aGl0ZVNwYWNlJTNEd3JhcCUzQmh0bWwlM0QxJTNCJTIyJTIwdmFsdWUlM0QlMjIlRDclQTIlRDclOTMlRDclOUIlRDclOUYlMjAlRDclOUIlRDclOTklMjAlRDclOTQlRDclQUElRDclOTUlRDclQTglMjAlRDclOUMlRDclOUUlRDclQUElRDclQTclRDclOUYlMjAlRDclQTAlRDclQTAlRDclOTglRDclQTklMjAlRDclQTIlRDclOUMlMjAlRDclOTklRDclOTMlRDclOTklMjAlRDclOTQlRDclOUUlRDclOTElRDclQTclRDclQTglMjIlMjB2ZXJ0ZXglM0QlMjIxJTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhHZW9tZXRyeSUyMGhlaWdodCUzRCUyMjYwJTIyJTIwd2lkdGglM0QlMjIxMjAlMjIlMjB4JTNEJTIyMzUwJTIyJTIweSUzRCUyMjQ1MCUyMiUyMGFzJTNEJTIyZ2VvbWV0cnklMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZteENlbGwlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteENlbGwlMjBpZCUzRCUyMkNLUk1ZSXdWYWZiX05Ub3FJaHl3LTIzJTIyJTIwZWRnZSUzRCUyMjElMjIlMjBwYXJlbnQlM0QlMjIxJTIyJTIwc291cmNlJTNEJTIyQ0tSTVlJd1ZhZmJfTlRvcUloeXctMiUyMiUyMHN0eWxlJTNEJTIyZW5kQXJyb3clM0RjbGFzc2ljJTNCaHRtbCUzRDElM0Jyb3VuZGVkJTNEMCUzQmV4aXRYJTNEMC41JTNCZXhpdFklM0QxJTNCZXhpdER4JTNEMCUzQmV4aXREeSUzRDAlM0JlbnRyeVglM0QwLjUlM0JlbnRyeVklM0QwJTNCZW50cnlEeCUzRDAlM0JlbnRyeUR5JTNEMCUzQiUyMiUyMHRhcmdldCUzRCUyMkNLUk1ZSXdWYWZiX05Ub3FJaHl3LTIyJTIyJTIwdmFsdWUlM0QlMjIlMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteEdlb21ldHJ5JTIwaGVpZ2h0JTNEJTIyNTAlMjIlMjByZWxhdGl2ZSUzRCUyMjElMjIlMjB3aWR0aCUzRCUyMjUwJTIyJTIwYXMlM0QlMjJnZW9tZXRyeSUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyNDcwJTIyJTIweSUzRCUyMjQzMCUyMiUyMGFzJTNEJTIyc291cmNlUG9pbnQlMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteFBvaW50JTIweCUzRCUyMjUyMCUyMiUyMHklM0QlMjIzODAlMjIlMjBhcyUzRCUyMnRhcmdldFBvaW50JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGbXhHZW9tZXRyeSUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14Q2VsbCUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214Q2VsbCUyMGlkJTNEJTIyQ0tSTVlJd1ZhZmJfTlRvcUloeXctMjQlMjIlMjBlZGdlJTNEJTIyMSUyMiUyMHBhcmVudCUzRCUyMjElMjIlMjBzb3VyY2UlM0QlMjJDS1JNWUl3VmFmYl9OVG9xSWh5dy0yMiUyMiUyMHN0eWxlJTNEJTIyZW5kQXJyb3clM0RjbGFzc2ljJTNCaHRtbCUzRDElM0Jyb3VuZGVkJTNEMCUzQmV4aXRYJTNEMC41JTNCZXhpdFklM0QxJTNCZXhpdER4JTNEMCUzQmV4aXREeSUzRDAlM0JlbnRyeVglM0QwLjUlM0JlbnRyeVklM0QwJTNCZW50cnlEeCUzRDAlM0JlbnRyeUR5JTNEMCUzQiUyMiUyMHRhcmdldCUzRCUyMkNLUk1ZSXdWYWZiX05Ub3FJaHl3LTMlMjIlMjB2YWx1ZSUzRCUyMiUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214R2VvbWV0cnklMjBoZWlnaHQlM0QlMjI1MCUyMiUyMHJlbGF0aXZlJTNEJTIyMSUyMiUyMHdpZHRoJTNEJTIyNTAlMjIlMjBhcyUzRCUyMmdlb21ldHJ5JTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhQb2ludCUyMHglM0QlMjI0OTAlMjIlMjB5JTNEJTIyNTMwJTIyJTIwYXMlM0QlMjJzb3VyY2VQb2ludCUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyNTQwJTIyJTIweSUzRCUyMjQ4MCUyMiUyMGFzJTNEJTIydGFyZ2V0UG9pbnQlMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZteEdlb21ldHJ5JTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGbXhDZWxsJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGcm9vdCUzRSUwQSUyMCUyMCUyMCUyMCUzQyUyRm14R3JhcGhNb2RlbCUzRSUwQSUyMCUyMCUzQyUyRmRpYWdyYW0lM0UlMEElM0MlMkZteGZpbGUlM0UlMEHc/+x8AAAQAElEQVR4AeydX6hcVb7n12occCZRkwfpVqSTXGwCvhjRC/pgpw4EhoCmvZIHw4BJ9GVQyYsDyTDT5kRliIgvXvWtSSJ3SGDCnRCFMFzhnKQf7MuNGBECaWWSQKMtBG5yY2QehDP57JNV2alTu05Vnfqz/3yaXmfvvfb68/t9Vpn1rd9atfcvFvyfBCQggQoTuHjx4sLs7OxCq9VaWL9+/UIIwVRCBowNiXFivCr8kdP0FRD4xc3/QP2/BCQggUoSOHDgQNiwYUNm+/79+8Pc3Fy4+e+haWGhdAwYGxLjxIAxbowf56bmEFB0NGes9VQCtSIwMzMT5ufnw81IR2Aia7Va4eY36Vr5WCdnGBsS48R4IUDwb/fu3RxMDSGg6GjIQOumBOpEYOam4MAfJi4mMs5N1SLAuO3cuTMTikXCo1oeaW0/BBQd/VCyjAQkUBoCKSSP4CiNURoyFIEkPKis8IBC/ZOio/5jrIcSqBWB2dnZcOjQoVr5NJgz9SqN8GC5haWyenmmN90IKDq6UTFPAhIoJQGiHLt27cpC8qU0UKOGIoDwaLVagfEN/q/WBBQdtR5enZNAvQhcunQpbN68uatTZlabAPs7jHZUewz7sV7R0Q8ly0hAAqUgwKTEN+JSGKMRIyVAtANROdJGbax0BBQdpRsSDZLAKAnUqy0mJSanenmlNxJoDgFFR3PGWk8lIAEJlJYAYhJRWVoDNWwkBBQdI8FoI1UjoL0SkIAEJDB5AoqOyTO3RwlIQAISkEAjCSg6GjnsRU6bLwEJSEACEhgfAUXH+NjasgQkIAEJSEACOQKKjhyMolPzJSABCUhAAhJYOQFFx8oZ2oIEJCABCUhAAn0QWIHo6KN1i0hAAhKQgAQkIIFbBBQdt0B4kIAEJCABCVSOQMUMVnRUbMA0VwISkIAEJFBVAoqOqo6cdktAAhKQQBEB80tKQNFR0oHRLAlIQAISkEDdCCg66jai+iMBCUigiID5EpgyAUXHlAfA7iUgAQlIQAJNIaDoaMpI66cEJFBEwHwJSGBCBBQdEwJtNxKQgAQkIIGmE1B0NP0ToP8SKCJgvgQkIIERE1B0jBiozUlAApMl8Mknn4QYY5ZWrVoVvvjii8yAH3/8MbRarcB98rgX42K5GGPgmnwKc+Q6xtv3qce977//Pmvnm2++yY4xxuxI+9xPiWv6i3GxjZ07d6ZbmQ0xLubHuHik7A8//JC1FeNiXmcdrvO2vfPOO1mbnX3FGANls5sdf8iPcbH9GBeP5KVinGMLbZJHH+RxnlLehhhj5k+6xxFWMS62nW+LdmJczI+xOzfqm5pDQNHRnLHW09EQsJWSEXj22WfDwsJClr799tvw8ssvt4VHMvXxxx8PTPC/+93vwtmzZ7Oyx44dC88//3xAVHD/xo0bWT5tnTx5Mrz22mvZvQceeCBs3bo1bN++PTtyf926deHDDz9MzWfH1atXh/n5+XYbjzzySFsI5G2kPjasWbMmEz6pzvXr18Ply5cDkz4NzszMhGvXroWXXnopnDlzJnz33Xfh6NGjmW+dfXGPdpj8qZtPR44cCfjz4osvZrZ19pP8SEfqYjvHlPJ8qP/ee++1hQeC5Pe//31mH77B6tVXX82q0jd5JOqRme+Ha1OzCCg6mjXeeiuBWhNAIOzYsSN89tlny/rJpP7YY49lk2VnYe5t2LAhEyjc27JlSxZJYeLmGgFy/vx5TgsTZRERiJrCQrkbCInXX389pHa5fuqpp8KmTZsCkz6+Pfroo13t5d4rr7wSjh8/nmux+yntdvaDiHrzzTcz0XTq1KnQKTryLaX6qS9YwxwbKFfkd6pH+ymqQnlTswgoOpo13uPz1pYlUCICaeLu1yS+rTO5J4HABMlEnK//9NNPhzSx5vN7nV+9ejUTCbT/3HPPhTTZvv/++wFBQT+d9REqqRz3EDgc84n7rVtLRym/l1BIZbodiax8/PHH4aeffgqILcqkI+dFKdnZjXXym+UV2k9tPPjgg2HNzQhPuvbYPAKKjuaNuR5LQAIDEmD54qGHHhqwVvfiLIEwYacliO6lFnP/8pe/BCbqxavx/EU0HDx4MBNHv/71r7MlpG5iqKj3bmIHYTFuu4vsMb/cBBQd4x0fW5eABGpAgKWEbpPrMK4xybPvYbmJnajL119/PVbRQcQEAYRvRHE++OCD8NFHH2V7Wfr1jaUn9ppgL3WImhAloj2uTRLIE1B05Gl4LgEJ1JIAIf577rknfPnllwNP4kzMbOjkmzvtxBjDtm3bhuaE6EiVmaiZoGNc/IUH7Z4+fTo888wzgU2xRAwoy1JKjDEwoXPdb0rt0y7Cols9+sA37rHhlb7Yo4Hf5C2X2G/y1ltvZVxjjIE9G24WXY5ac+9PR3Q0l7eeS0ACYyawd+/ewK8miCTwiw4mUvL4BQXf6tM3cO6fOHEi26TJxHnu3Lk79mzQBnXz5VI7tMX9Xq7QD23Sdr4c9WiHPMpgE+3lE3azhwT7KMN1up9sIo9z2iFxTtucp0Td1H7qk3upbN438km0Qdvc47pbon6+DNfJvnw+beX7hQU+9Wq7W3/m1YeAoqM+Y6knEpCABCQggb4JTKOgomMa1O1TAhJoFAG/4TdquHW2BwFFRw843pKABCQggaYR0N9xElB0jJOubUtAAhKQgAQk0Cag6Gij8EQCEpCABIoImC+BURBQdIyCom1IQAISkIAEJLAsAUXHsogsIAEJSKCIgPkSkMAgBBQdg9CyrAQkIAEJSEACQxNQdAyNzooSkEARAfMlIAEJdCOg6OhGxTwJSEACEpCABEZOQNExcqQ2KIEiAuZLQAISaDYBRUezx1/vJSABCUhAAhMjoOiYGGo7KiJgvgQkIAEJNIOAoqMZ46yXEpCABCQggakTUHRMfQiKDDBfAhKQgAQkUC8Cio56jafeSEACEpCABEpLoHKio7QkNUwCEpCABCQggZ4EFB098XhTAhIoE4FWqxXm5+fLZJK2jIjApUuXwvr160fUms2MmcDQzSs6hkZnRQlIQAISGBUBxGTrpqgcVXu2U04Cio5yjotWSUACXQjs3LkzHDlypMsds6pO4PTp09WPdFR9ECZgv6JjApDtQgISGA0BvgnzjZg0mhZtpQwEWFo5fPhw2L9/fxnM0YYxElB0jBGuTUtAAqMlwJo/E5PRjtFyHXNryza/e/fuMDs7u2w5C1SfgKKj+mOoBxJoFAGiHXwzZqJqlOM1dXZmZibzDDGZnfin1gQUHbUeXp2TQP0IEO04dOhQQHgwYXGspJcNN5pxY/zAMDc3x8HUAAKKjgYMsi5KoG4EkvAg6sHERdRjfn4+zN9MdfO1Tv4gNBijAwcOBMat1WoFBUedRnh5XxQdyzOyhAQkMDkCffeE8CAkz6TFORMZKcYYYjTFWD4GCA3GiEG+ePGiG0cB0bCk6GjYgOuuBOpGAMGRxAcCZGFhIdQlMVZ18QU/EBqMEeOFb6bmEVB0NG/M9biKBLRZAhKQQA0IKDpqMIi6IAEJSEACEqgCAUVHFUZJG4sImC8BCUhAAhUioOio0GBpqgQkIAEJSKDKBBQdVR69ItvNl4AEJCABCZSQgKKjhIOiSRKQgAQkIIE6EmiS6Kjj+OmTBCQgAQlIoDIEFB2VGSoNlYAEJCABCVSbwC9Cte3XeglIQAISkIAEKkLASEdFBkozJSABCUigvgSa4pmioykjrZ8SkIAEJCCBKRNQdEx5AOxeAhKQgASKCJhfNwKKjrqNqP5IQAISkIAESkpA0VHSgdEsCUhAAkUEzJdAVQkoOqo6ctotAQlIQAISqBgBRUfFBkxzJSCBIgLmS0ACZSeg6Cj7CGmfBCQgAQlIoCYEFB01GUjdkEARAfMlIAEJlIWAoqMsI6EdEpCABCQggZoTUHTUfIB1r4iA+RIoH4EDBw70NGq5+z0re1MCJSCg6CjBIGiCBCQgAQisW7cu3H///aFTXHAdY6SISQKVJqDoqPTwjd54W5SABKZHoNVqhStXroSDBw+GtWvXZobEGMPbb7+dne/fvz87+kcCVSWg6KjqyGm3BCRQOwLr168P7777brjrrrvC1atX2/79/PPPYXZ2tn3tiQSqSkDR0dfIWUgCEpDAZAhs37493H333Us6M8qxBIkZFSSg6KjgoGmyBCRQXwJEO/bu3RtWr17ddtIoRxuFJxUnsCLRUXHfNV8CEpBAKQl0RjuMcpRymDRqCAKKjiGgWUUCEpDAOAmkaAd9GOWAgqkHgUrdUnRUarg0VgISaAoBoh34apQDCqa6EFB01GUk9UMCDSVw6dKl7LkWMzMzYcOGDSHGWIuELwxpjPXxB58YJ547gm9jTTZeSgKKjlIOi0ZJQAL9EGDyYiKjLBGBubm5sLCwYCohA8aGxDgxXowb48e5qTkEFB3NGWs9lUCtCPCNeX5+Ply8eDEwkfFgLfZC1MrJ0TsztRYZGxLjxHghQDBm9+7dHEwNIaDoaMhA66YE6kRg5uZSCv4wcTGRcW6qFgHGbefOnYGjwqNaY7cSaxUdK6FnXQlIYOIEUkgewTGyzm1oKgQQHAgPOld4QKH+SdFR/zHWQwnUisDs7Gw4dOhQrXxqsjMID5ZbWCprMoem+K7oaMpI66cEBidQuhpEOXbt2pWF5EtnnAYNTQDh0Wq1sl8hBf9XawKKjloPr85JoF4E+Hns5s2b6+WU3mQEWGYx2pGhqPUfRUeth1fnxkLARqdGgEmJb8RTM8COx0aAaAeicmwd2HApCCg6SjEMGiEBCfRDgEmJyamfspaRgATKR0DRUb4xqapF2i0BCUhgaAKISUTl0A1YsRIEFB2VGCaNlIAEJCABCVSfgKJj3GNo+xKQgAQkIAEJZAQUHRkG/0hAAhKQgAQkMG4C0xId4/bL9iUgAQlIQAISKBkBRUfJBkRzJCABCUhAApMhMPleFB2TZ26PEpCABCQggUYSUHQ0cth1WgISkIAEigiYPz4Cio7xsbVlCUhAAhKQgARyBBQdORieSkACEpBAEQHzJbByAoqOlTO0BQlIoGQEeHlYjDHEGAPvavnxxx8zCzly/cknnwRSjItlVq1aFb744ousDH+oz33Ov//++6wNjlynlNqKcbGNGBeP1KN+jIvX9EdZ6tEHfVGGa9rctGlT1ne+Towxs48ynYm6McZAO7THfeq+8847WTvkx7jYN2W5T0plyItx8X6Mt4/kU46Evdgd4+37MS6y/E//6T9lXGNcvKYsdbAFX/CJ87wdMRb7Q11TcwgoOpoz1noqgcYQOHLkSFhYWMjS66+/Hp555pmQJscE4dlnn83uU+7YsWPh+eefD0yY3H/kkUfC+fPnOQ3fffdduHr1anae/7N69erAC+ion9KLL76YFSnq//HHHw/09dprr7X7WrNmTXjwwQezegcPHsxsok/K5IVAVuDmn2Q37bz//vs3c27/n/Zv3LiRtXH9+vXw3nvvjr7pZAAAEABJREFULREvqT42U2bz5s3h5MmTgfzUUqdv3Kfcp59+Gv7n//yfWfvUL2Kbt4Ny1MefxDf147F5BBQdzRtzPZZAowjMzMwEJvYLFy4U+k2Zxx57LBMYFMqLDq5Xkmg73z8C48qVK2Hfvn3h7NmzWdP33HNPoM/s4uafBx54IHzwwQeZaOgUSzdvt/9/+fLlJWIq3UQ4IAqOHz+eskZ+7PStqAPKbdiwoe1vUTnz609A0VH/MdZDCUyBQLm6JFJB9KCXVfkyCINr1661J/T169cHhEFnfZYsUjSCJYWvvvoqPPHEE53FskgJ/fNNn4jKTz/9FD7++OOwbdu2gDBAIGzZsiW8+eabodVqZf1iw5qbUZAljQ2YkYTJkZvRn7179w5Ye/nieW6pNCzSUgt5+Ldu3TpOTQ0noOho+AdA9yUggaUEmPCZTFl+QCzcd999gYlzacnFHKIRiIe33norEKVYzF36l7bWrl2bRVRYSvn1r3/dFilpSYIlG/qibFG/iB0Ey9IezJFAuQkoOso9PlpXMwK6Uw0CRDWIMrAkw76I7du39zQccUKBblEO8lNCSNAu7b/66quBJQciHul+OhIpeOGFF0K3ftkwSjna4pj65twkgbITUHSUfYS0TwISGAsBJu8YF3+dgQg4ffp0ttzBcgmRBqIM7EVgWSC/ybKbMQgAIiPd7nXm0R7tkxA0LKnQZ94exAsbRTv7ZXnm6NGjYc+ePdlyDwKGvjv78FoCZSWg6CjryDTKLp2VwPgIMLmzZMEEnj9nfwO/rOhMWBJj5JD9SoO9ENlFjz8sjZw7d67r0kq+T2zIt0c9fm1Cfqc95HV2ydIN/VCPdk+cOBE4p03qd5anDXynbOc9rsnnPuW4Lkrcpxzl82W4Jp/72IFt2Jg/T+WxkXLp2mMzCSg6mjnuei0BCRQQYGJEiDBJFhQxWwISGJKAomNIcJOoZh8SkEC5CSBMECnltlLrJFAeAoqO8oyFlkhAAhKQgARqTaCCoqPW46FzEpCABCQggdoSUHTUdmh1TAISkIAEJDAmAkM2q+gYEpzVJCCBZhLI/7SVn7oWUeABXjHG7OVovcoV1TdfAnUkoOio46jqkwQkMBYCiIePPvooe6Io703hAV88yKuzM4QJjx/nwV29ynXW87ryBHRgGQKKjmUAeVsCEpBAInD+/Pns3Sg8i2Ljxo3hb/7mb8Jnn32WbrePlEsPAeOR6jHGTKi0C9w64WFflItxMSKCWLl1645DPmqS3s1CAR6/znWMi/VjXDwijmgrxsVr+qAv6nCkzjfffJP5EmPMjrTFfZMExklA0TFOurYtAQnUigBiIr0NlgdjMZmT1+kkjy/noVlM8DwxlOd+ID46yyFeiIhwn3JEURAMneX4aS5lSLzj5ZlnnsleCocN9EN+Sun1+jwsLOW98sorYceOHVkd+ty6dWv2iHWOlMGPDz/8sLPb0V3bkgRuEVB03ALhQQISkEAvAkQCEAi9yqR7PLvjgw8+CAgN3ir7z//8z9mTQ9P9bkfEAOKgm4jJl+fR7Dz+nPfC5PN7nSchwnIP5XijbYwxpHxE0nL9Us8kgZUSUHSslKD1JSCBRhAgqkBEoB9nWQ45fvx49hj1P/3pT+HJJ58MRDBIMbaXPAKRkM72+pn8ec8LkRHq0hftcs7+kq+++irw7hau8ylfh/ynn36662PbuWeSwLgIKDrGRdZ2JSCB2hFgaSWJghT5IC/vKEKCJQ+iB+QTwWAPBSKECAjLGSSiJtyjzCgS9rD08tZbby0rJhAsDz300Ci6tQ0JDERA0TEQLgtLQAJjJ1DiDhAYCAqEBcsb//f//t/AUkXeZIREEhnkU5Y61OV6XCktnXSLcnT2iQAatz2dfXotAQgoOqBgkoAEJNAHASIV7LtgrwaTO5sveaMqVVnm4BcjnJNPJCPGmO3rQISwsZN7+cRyyKpVq7JnecQYw759+8LHH38caCtfrp9zohcsoaSytBHj4lIO9rLs8tvf/jb88Y9/DNeuXcvswt4YY9i2bVuq5lECYyWg6BgrXhuXwMgI2FBJCCAeWB4hIUKSWUeOHAnc45r9H0Q3KEPiHvmdCcFy48aNbO8H5VIqKk/91Ha+b/Jp69y5c+2lFdpI7aUjfbGX48SJE9nGVuxN9yhPOyYJjJOAomOcdG1bAhKQgAQkIIE2AUVHG4UnlSSg0RJoOAEiFJ1Rj4Yj0f0SE1B0lHhwNE0CEpCABCRQJwKKjjqN5m1fPJOABCQgAQmUjoCio3RDokESkIAEJCCBehJoluio5xjqlQQkIAEJSKASBBQdlRgmjZSABCQgAQlUnwCio/pe6IEEJCABCUhAAqUnoOgo/RBpoAQkIAEJ1J9AMzxUdDRjnPVSArUgwOPEedJnLZzRiTsIXLp0Kaxfv/6OPC/qR0DRUb8x1SMJSEAClSOAmERUdhrudb0IKDrqNZ56I4FaE+AlZjyBs9ZONtS506dPG+lowNgrOhowyLoogboQ4Jsw34hJdfFpOD/qVYullcOHD4f9+/fXyzG9WUJA0bEEiRkSkEBZCbDmz8RktKOsIzScXbt37w6zs7PDVbZWpQgoOio1XBorAQkQ7eCbMRNVJw2vq0dgZmYmMxoxmZ34p9YEFB21Hl6dk0D9CBDtOHToUEB4MGFxrJ+X9feIcWP88HRubo6DqQEEFB0NGGRdbDqB+vmfhAdRDyYuoh7z8/Nh/maqn7f18QihwRgdOHAgMG6tVisoOOozvv14oujoh5JlJCCB0hFAeBCSZ9LinImMFGMMMZpiLB8DhAZjxIfp4sWLbhwFRMOSoqNhA667twl4Vg8CCI4kPhAgCwsLoS6JEaqLL/iB0GCMGC98MzWPgKKjeWOuxxKQgAQkIIGpEFB0TAV7mTvVNglIQAISkMB4CCg6xsPVViUgAQlIQAIS6CCg6OgAUnRpvgQkIAEJSEACKyOg6FgZP2tLQAISkIAEJNAngRWKjj57sZgEJCABCUhAAo0noOho/EdAABKQgAQkUGkCFTJe0VGhwdJUCUhAAhKQQJUJKDqqPHraLgEJSEACRQTMLyEBRUcJB0WTJCABCUhAAnUkoOio46jqkwQkIIEiAuZLYIoEFB1ThG/XEpCABCQggSYRUHQ0abT1VQISKCJgvgQkMAECio4JQLYLCUhAAhKQgARCUHT4KZCABIoJeEcCEpDACAkoOkYI06YkIAEJSEACEigmoOgoZuMdCRQRMF8CEpCABIYgoOgYAppVJCABCUhAAhIYnICiY3Bm1igiYL4EJCABCUigBwFFRw843pKABCQwSQIHDhzo2d1y93tW9qYESkBA0TH+QbAHCUhAAn0RWLduXbj//vtDp7jgOsbYVxsWkkCZCSg6yjw62iYBCTSKQKvVCleuXAkHDx4Ma9euzXyPMYa33347O9+/f3929I8EqkpgeqKjqsS0WwISkMCYCKxfvz68++674a677gpXr15t9/Lzzz+H2dnZ9rUnEqgqAUVHVUdOuyUggVoS2L59e7j77ruX+GaUYwkSM0ZAYNJNKDomTdz+JCABCfQgQLRj7969YfXq1e1SRjnaKDypOAFFR8UHUPMlIIH6EeiMdhjlmPQY29+4CCg6xkXWdiUgAQkMSSBFO6hulAMKproQUHTUZST1QwISqBUBoh04VKYoB/aYJLASAoqOldCzrgQkMHUCly5dyp5rMTMzEzZs2BBijLVI+ALcGOvjDz4xTjx3BN9MzSOg6GjemOuxBGpDgMmLiQyHiAjMzc2FhYWFCSf764c5Y0NinBgvxo3x49zUHAKKjuaMtZ5KoFYE+MY8Pz8fLl68GJjIeLAWeyFq5WSNnGFsSIwT44UAwb3du3dzMDWEgKKjIQOtmxKYNIFx9jdzcymF9pm4mMg4N1WLAOO2c+fOwFHhUa2xW4m1io6V0LOuBCQwcQIpJI/gmHjndjhSAggOhAeNKjygUP+k6Kj/GOthqQhozEoJzM7OhkOHDq20GeuXhADCg+UWlspKYpJmjJGAomOMcG1aAhIYLQGiHLt27cpC8qNt2damSQDh0Wq1sl8hBf9XawKKjloPb3Wc01IJ9EOAn8du3ry5n6KWqRgBllmMdlRs0IYwV9ExBDSrSEAC0yHApMQ34un0bq/jJEC0A1E5zj5se/oEFB3TH4MeFnhLAhLIE2BSYnLK53kuAQlUh4CiozpjpaUSkIAEaksAMYmorK2DOpYRqKToyCz3jwQkIAEJ3EHgiy++CJs2bQrff//9HflFF5SjPPWKypgvgVESUHSMkqZtSUACEqgIAQTHk08+Gb755puKWKyZJSMwlDmKjqGwWUkCEigLgU8++eSOF7ytW7eu/U3/nXfead+jXLKZX0pwj+sff/wxsDk1xsUXq6VynfkxxkA96nQm6sS4WD/GGFatWhVS9IB+YoxZH7RJPvdjLC6f+qHdznPEAvYiFjjGeLtt7CL/wQcfbPtN/+TnEzY8/PDD4b/+1/8afvOb3+Rvtc/pB5YxLtqJzdRLBfAl9R9jDNjKPcoQPaE+59SLcbGNGG+Xo6ypeQQUHc0bcz2WQO0I8DPa69evZy96++CDD8KOHTsCk+LevXuzvO+++y689tpr7YkxD+DVV18NW7dubZf7/e9/nwmG1atXB34tk15mRhtcp8k138azzz6b1U9l33jjjfD666+3bcA2yvMU1ccffzz88MMP4Xe/+104e/ZsVu/YsWPh+eefz8TSI488QtEsnT9/PqRrzsl84IEHMnu3b9+eHekTcfDhhx8G2r5x40bWJvkHDx6kypKUymHDkpu5jLVr17ZtPHPmTPi7v/u7jA1FirhxL6XUD7aQTp48mY0DgiSVqd1Rh3oSUHT0xONNCUigagSeeOKJcPXq1XDhwoW26UzUiJHjx4+38zhh8rt8+XJ48cUXuQyUQ7B89tln2XX+D/deeeWV0NlGvkw6p728DQgYhEG633nkXTKPPfZYQNike4imU6dOpcvsmATIli1bskgG/XADAZJECdcpdctL9wY9IiCwExsH4Zbvh/obNmzIhEw+3/PmEFB0NGes9VQCjSbAksO1a9ey6EMCwQSKOEjX6chkzaTfarXuiI6kST+VS0eiH5SlDnn33HPPip6a+vHHHwfaQKgQraFNbOKY0tNPP52JpHTdeUQYfPXVVwGB0nmvn2v6X79+/ZKi2NGLW74Cyyu3llqy7OXEV1bIP7UmoOio9fDqnAQkkAgwUd53332BiS/lIUTWrFmTLtvHInHRLjDGEyI1jz76aBYNOHLkSNYTYgbBlAQEvjz00EPZvaI/+/bty5aZiFAUlemVD6ennnoqYE/nvpAycuvli/fKQ0DRUZ6x0BIJSGBMBPjWz54OliHyXbBkQjSByAL5lDt69OjQ0QHaGCh1KYxNiI7333+/fZf9GkRkNm7cmOWxxNNLGOHHSqIcWSc3/xBlYS8Gx5uX7f9jY9m4tY3zpFqdUPAAABAASURBVNQEFB2lHh6Nk4AE+iFw+vTpbDkixhj4Fs4vOKjHN/QYF/PY08GGT/LziQmdvRMxLpZ76623sg2Z+TIrOU82JGHDNUsXX375ZWZrt7axib0mMS7+6gP7Pv300yxKk6Ie+ElbMcawbdu2Jc0QwaHMkhsjysBG7IpxPNxGZKbNlIyAoqNkA6I5EpBAGAgBQoJv4/nELzhYVuAbesqnXGqYZQvucc0yAr9K6SyX8vP1OKcu9fKJfNqgDvkcT5w4kYkX+sm3na4RFUQMOsuna9pL9TinzXSvW9t5u2iXOhypU5S4f+7cuczOojJF+dhDH8lGGFAW7rRJ2/lz7pGwM5Xl2tQsAoqOZo233kpAAhKQgASmRkDRMTX0diyBAQlYXAJTJkCUgkjNlM2w+woTUHRUePA0XQISkIAEJFAlAoqOKo2WtnYjYJ4EJCABCVSEgKKjIgOlmRKQgAQkIIGqE1B0VH0Ei+w3XwISkIAEJFAyAoqOkg2I5khAAtMhwNtcY1x8LkaMi0fykjWc81yM/KO9eQgXjz/nSErn+ceip/N0P8YYaId2aTPG22+vpe1+3sraWY4+aC+l1G6Mt/tK99IxXybG7v7iD88FoQ42U4dzkwSGJdA00TEsJ+tJQAI1J8AvM06ePJm9/I1nT1y/fj3wLA0m27zrPHCLB2+Rx+PIOedhX5zz1FDyu6W5ubnw+uuvZ483//zzz7N3wNAn9Xh6KEKC51rwjBH6J2EPT1JFsOTbzJfDzvfee6/9jphkL/W5xwO8OkUJbdE37fPSuFQ27y8P/6JcOnLe6ymo3DdJYDkCio7lCHlfAhJoJAEefoVI4AVnRQB4G21eaPCCNARIt/I8NZSHYiFaqIMgoBzlqcd5Z+rnrazJTh6NTlQCQbNnz56sqfy9LKPHn1Q2+cs1jzp/8803Aw8BQ7woOnoA9FZfBBZFR19FLSQBCUhAAhC4ePFi9gjzjz76KPCuFiZoIhaICe4jLpioyee630S0Y9OmTSFFNqjPxN9PfaIUf/3rX8OlS5eWFOcegmTJjR4ZREx4dPtPP/0UED8UTUfOTRIYhoCiYxhq1pGABBpLgMd7t1qtcPDgwWz5hWtgECHYunVr9n4UrlMissESTLrOi5OUN6pjUdQE4YKAGaQf/MFH7P31r38duvk2SHuWXZ5AE0ooOpowyvooAQmMlMD27dsDUY4UkSBCwd6HLVu2LOkH0UH04cKFC9k9lmR4i2wSK1nmiP6wCfWpp54K6Q21RDfY74G9g3RBPaIjLKdgJy/Ly/s7SFuWlUCegKIjT8NzCUigkQQQD0QD2HfBRLscBJZPXnnllWyJJcYYfvvb34b//b//d9cXpzFp/+EPf8jKxBgzsUIEgT74NQjRiX/6p3/iciQpPaY8xpi9eZcIBfbmG+/HX6IzCCbqUZ/ozo4dO7INsORNLtlTnQgoOuo0mvoiAQkMRQBhwDd7fsWRJm0aYrLlVx6cc8zf45zyJH5xwi9KKNctcY8ylKUf+qMcbZLHPcqQ0htauU+iDHZw3m+iDu2SsLOzHv1jR+d9+qEuSzHpTbapLvnD7FNJ9T1KAAKKDiiYJCABCVSMgOZKoIoEFB1VHDVtloAEJJAjQITCKEQOiKelJaDoKO3QaJgEJDA4AWtIQAJlJqDoKPPoaJsEJCABCUigRgQUHTUaTF2RQBEB8yUgAQmUgYCiowyjoA0SkIAEJCCBBhBQdDRgkHWxiID5EpCABCQwSQKKjknSti8JSKC2BHiKJw/QijGGGBcT7y9JDvMgMK55eilPDo1xsUz+DbCpTHp4V4yLZWJcPKay3fqiLn1x77nnngv0Q16MMfDgM9rkvkkC0ySg6Jgm/ZL2rVkSkMDwBE6ePBl46Nb169fDqVOnQhIKqUUeAMbDwFIZHlPeWSY9vOvFF18Mqb2zZ8+GV199NRMTPLyLn8jSBon3o3BNO9x7+eWXw0svvZR1yX2enrpv377s2j8SmCYBRcc06du3BCRQWwJM/jyC/Pjx44U+Uub1118PvcqkyogV3vKKwEh56YhIQVikdp544oks2rJnz56sCO+EuXbtmo8wz2j4Z5oEFB1907egBCQggeEJHDlyJHR7JDkt8khylkU47ydRlqUcIhupfOc7Y3ip3MaNG9NtjxIoBQFFRymGQSMkIAEJjI4A0ZB77703EEkZXau2JIGVE1ix6Fi5CbYgAQlIQAKjJPDZZ5+Fhx56aJRN2pYERkJA0TESjDYiAQlIYJHAtm3bsv0UMcYw7OZNlk1ijNlejwcffHCx4QH+nj9/PrDcktphj8fVq1cHaMGiFSNQGXMVHZUZKg2VgATKTIClDH5Bwq9F8om9HL3szr+sjbLs+yCPNviVCxtIU33ucy/1xXm6xzn3uebINYl2SNhGPe6bJDAtAoqOaZG3XwlIQAISGC8BWy8dAUVH6YZEgyQgAQlIQAL1JKDoqOe46pUEakmAn4myTFBL5ybnVCl7unTpUli/fn0pbdOo0RFQdIyOpS1JQAISkMCQBBCTiMohq1utIgQUHRUZKM2UgARC4F0ibJIcCwsbnSqB06dPG+mY6ghMpnNFx2Q424sEJDACAnwT5hsxaQTN2URJCLC0cvjw4bB///6SWKQZ4yKg6BgXWduVQD0IlMoL1vyZmIx2lGpYVmzM7t27w+zs7IrbsYHyE1B0lH+MtFACEsgRINrBN2Mmqly2pxUlMDMzk1mOmMxO/FNrAoqOWg+vzo2NgA1PjQDRjkOHDgWEBxMWx6kZY8dDE2DcGD8amJub42BqAAFFRwMGWRclUDcCSXgQ9WDiIuoxPz8f5m+muvlaJ38QGozRgQMHAuPWarWCgqNOI7y8L4qO5RlZon8ClpTAxAggPAjJM2lxzkRGijG2330So+cxlocBQoMx4kNy8eJFN44ComFJ0dGwAdddCdSNAIIjiQ8ECO8ZqUtirOriC34gNBgjxgvfTM0joOiYxJjbhwQkIAEJSEACQdHhh0ACEpCABCQggYkQmKbomIiDdiIBCUhAAhKQQDkIKDrKMQ5aIQEJSEACEpgCgcl2qeiYLG97k4AEJCABCTSWgKKjsUOv4xKQgAQkUETA/PEQUHSMh6utSkACEpCABCTQQUDR0QHESwlIQAISKCJgvgRWRkDRsTJ+1paABCQgAQlIoE8Cio4+QVlMAhKQQBEB8yUggf4IKDr642QpCUhAAhKQgARWSEDRsUKAVpeABIoImC8BCUjgTgKKjjt5eCUBCUhAAhKQwJgIKDrGBNZmJVBEwHwJSEACTSWg6GjqyOu3BCQgAQlIYMIEFB0TBm53RQTMl4AEJCCBuhNQdNR9hPVPAhKQgAQkUBICio6SDESRGeZLQAISkIAE6kJA0VGXkdQPCUhAAhKQQMkJVFR0lJyq5klAAhIYgsCBAwd61lrufs/K3pRACQgoOkowCJogAQlIAALr1q0L999/f+gUF1zHGClikkB5CAxhiaJjCGhWkYAEJDAOAq1WK1y5ciUcPHgwrF27Nusixhjefvvt7Hz//v3Z0T8SqCoBRUdVR067JSCB2hFYv359ePfdd8Ndd90Vrl692vbv559/DrOzs+1rT0pNQON6EFB09IDjLQlIQAKTJrB9+/Zw9913L+nWKMcSJGZUkICio4KDpskSkEB9CRDt2Lt3b1i9enXbyVpEOdreeNJkAoqOJo++vktAAqUk0BntMMpRymHSqCEIKDqGgGYVCUhAAiMi0LWZFO3gplEOKJjqQkDRUZeR1A8JSKBWBIh24JBRDiiY6kJA0VGXkdQPCdSJwAC+XLp0KXuuxczMTNiwYUOIMdYi4QsYYqyPP/jEOPHcEXwzNY+AoqN5Y67HEqgNASYvJjIcIiIwNzcXFhYWTCVkwNiQGCfGi3Fj/Dg3NYeAoqM5Y62n1SegBzkCfGOen58PFy9eDExkPFiLvRC5Ip6WiABjQ2KcGC8ECObt3r2bg6khBBQdDRlo3ZRAnQjM3FxKwR8mLiYyzk3VIsC47dy5M3BUeFRr7FZiraJjJfSsWw4CWtEoAikkj+BolOM1dBbBgfDANYUHFOqfFB31H2M9lECtCMzOzoZDhw7VyqcmO4PwYLmFpbImc2iK74qO+o60nkmgdgSIcuzatSsLydfOuQY7hPBotVrZr5CC/6s1AUVHrYdX5yRQLwL8PHbz5s31ckpvMgIssxjtyFDU+k/zREeth1PnJFBvAkxKfCOut5fN9I5oB6Kymd43x2tFR3PGWk8lUHkCTEpMTpV3RAck0FACSXQ01H3dloAEJCCBMhBATCIqy2CLNoyPgKJjfGxtWQISkIAEJDAAgfoXVXTUf4z1UAISkIAEJFAKAoqOUgyDRkhAAhKQQBEB8+tDQNFRn7HUEwlIQAISkECpCSg6Sj08GicBCUigiID5EqgeAUVH9cZMiyUgAQlIQAKVJKDoqOSwabQEJFBEwHwJSKC8BBQd5R0bLZOABCpA4Mcffww8JfWTTz6pgLWaKIHpElB0TJe/vUtgQgTsRgISkMD0CSg6pj8GWiABCYyRQD4Skc5jjCHG24koBS8ci/F2XowxkIdp1HvuuefCF198keXFGMO6devC999/z+0sbdu2rd0mkQ/qZDdu/qHeqlWr2vfp72Z29n/OU3n6i7G7DRTuvJ/qcS+f6G/Tpk2ZfbQf42Kb77zzTr5Y+5wyqS3KxLhYnnwK5fNiXLyHP3/84x9D4kI5+uU67zv5JgkkAoqORMJjIwnodLMIrF69OvDSuIWFhZDSiy++mEE4cuRIOHnyZOCae9evXw+XL18OTLjUe/nll8NLL72UleX+K6+8Evbt2xe4l2/z7NmzYc2aNVm59Ofxxx8PN27cyPqk3ffeey+kCT2V4djLhnSfvkm0Q96HH37IoTA9++yz7X5PnTqV+VNY+OaNvXv3ZuW/++678Nprr2V2pjz6Jf/RRx8NZ86cCU8//XS47777Ank3q2bHq1evcmqSQFcCio6uWMyUgASaTgAx8frrr4fz589nKJ544oksUrFnz57sesuWLeHatWuh81s9E3CviTe1e/z48aydXn9S2WRDvmy6h5DotCFfLp1THrHz+eefL7E5lckfH3jggfDBBx+E5ex85JFH2ozy9T2XQDcCio5uVBqfJwAJ1JcASxQpysBywFdffRUQFP14zDf8jRs39izKJL1169YsAtKrIFGUfsRCvg1sJ/KS8h588MGwZs2adNk+Elk5d+5cQDi0M2+eUB5BlKIkN7N6/p/y3YRVvlKn6GDZCYGTL+O5BBIBRUci4VECEmgUASZ8IhlvvfXWksm5GwgiGPfee29PMYGYQUy8+uqr3ZqYeh4+IFLuueeevmyhPMsnvUREXpgQkUGE9NW4hRpJQNExwLBbVAISqA+B9G2/3yjHZ599Fh566KGeAPqNcvRsZEw3k8h66qmnegqn1D2bZNnTsX379pTV9YjoIHry5z//ORx51IPeAAAQAElEQVQ9ejSw7NS1oJkSuElA0XETgv+XgASaR4Bv8UyWeM4Ey7IAv0Ap+qaevsUTzYgxZksyqT5tpFRUP90vOv71r38Ny9lQVLdXfrKX6AbLPmwK7VWe5ZsYY0BMsKeDjai9ytMu91nS2bFjR+DItUkC3QiMQHR0a9Y8CUhAAuUmwOSY9j2w94FlEX6dkZ+UmXCPHDmSOcKRaxLlSPxqJb/0kMpkFXr8oY1UN50//PDD2a9laLeXDfl7+HDixImekQvap01Svm6ReZShLIm6neVgBTf6RqAgOhAzlKduZ3mvJZAnoOjI0/BcAhKQgAT6JoDIUGz0jWu8BSvSuqKjIgOlmRKQwHAEiEQQVUjf2vuNRgzXWzVrwQZGsKqmB1pdFQKKjqqMlHZKQAISkMCgBCxfMgKKjpINiOZIQAISkIAE6kpA0VHXkdUvCUhAAkUEzJfAlAgoOqYE3m4lIAEJDEuAJ6luuvVCt2HbsJ4EpkFA0TEN6vYpAQmUkYA2SUACYyag6BgzYJuXgASmTyA9ICvGGFatWhWIFCSreJ9JjDF7mVuMi0eeP1FUhyd7Pvfcc1kbqS4P9eIBY7SZrxfj0v4oQ6KdVquVvcWV/mJc7DvGxSN2/vGPfwypDHXy6Ztvvske4BXjYnnaoM1kG2Xxk2vyuU4JuynPNUdS/pzyqd/kT/KRNrGNxDl+cy/GmPlCOyYJFBFQdBSRMV8CElgkUIO//CSU50mQvv322/Dyyy9nogHX+Akt+SkdPHiQ7JCvc+zYsfD8888HJlh+Vkr9l156KStHvVdeeSXs27cvu87X494bb7wReMcLE3lWoMuf9LwLyvOkVF4sd+bMmfDYY491KR2yp37euHEjUJ6UbMY23pVCG1TkePXqVU7vSPmnpqYnrVKAc44p8aj4f/7nfw4cESFnz55t941f77//fvbeGh6sdvLkyfDee++FXn6mdj02l4Cio7ljr+cSaCQBnqjJ47p5l0q/AGZmZjIBwCROHd7XEmMMe/bs4TJ730jR21hffPHFwMR/4cKFrOw4/uTFAoIif92rP0QUz+fIl6F+uuZpo2+//Xb2xFPy8+1yncpx5LHpvEyOc5MEiggoOorImC+B3gS8W3ECaQJlaYGEO0zCRS8tQzgk0UFZohEbN27k9I7EcgRRgfSNn4l7/fr1d5ThgqgEEz6REa67pX7KYPNXX32VCR/aQAwk37hm6YN2OM8nIjMIBaI02IC9RCzyZTyXwKgJKDpGTdT2JCCByhJgIiYKwntFejmB+Lj33nuzCECvcpO412kzQiJFXRAfiJBOO3gTLKIJP1ja4T5LKByJ4iBSECtcp0Rb6dyjBIYloOgYlpz1uhMwVwIVJcA3fSZrJuTlXGBp5qGHHlqu2Njvd0Y56BDRQVTmz3/+cyiK2hChYSnk448/pkqWEC8IDZafyECsHD9+nNNsL0s+mpJl+kcCQxBQdAwBzSoSkED9CPBN/9KlS23HWHKJcfGXISyRnD59Omzbti37hQbf+pmUWUqJMQaiA0z07coTPEE8IDRSl9jKOdGaoqgNkQwEyUcffdT+1Q512FTLkZQiIDHG7FcyqS1+scIvV2BBOYQPYmWaDLDDVA0Cio7JjJO9SEACJSLAhJqfYDGNb/jnzp3Lfp3BNWX4ZUhnYv8DdTmS0n32ZzCZk5fOaYe8EydOtNslr1fqtGO5svRFHcohlBAdW7duzX7Zgg/kd0vUYQ9Hsh+fOsuRl+6nthAz6Zcz3M+3gy3429mO1xJIBBQdiYRHCUhAAhUngDBAJHCsuCuaX1MC0xUdNYWqWxKQQHUIMEGTqmOxlkqgugQUHdUdOy2XgAQkIAEJrJjAJBtQdEyStn1JQAISkIAEGkxA0dHgwdd1CUhAAhIoImD+OAgoOsZB1TYlIAEJSEACElhCQNGxBIkZEpCABCRQRMB8CayEgKJjJfSsKwEJSEACEpBA3wQUHX2jsqAEJCCBIgLmS0AC/RBQdPRDyTISkEApCPD2Vp56WQpjNGKkBHgEfbe38Y60ExubOgFFx9SHQAMkUF8CeiaBfgkgJhGV/Za3XDUJKDqqOW5aLYFGEti5c2fgfR+NdL7mTvNCPSMdNR/km+4pOm5C8P8SmCwBexuWAN+E+UZMGrYN65WPAEsrhw8fDvv37y+fcVo0UgKKjpHitDEJSGCcBPgmzMRktGOclCff9u7du8Ps7OzkO7bHiRNQdEwcuR0WETBfAv0QINrBN2Mmqn7KW6bcBGZmZjIDEZPZiX9qTUDRUevh1TkJ1I8A0Y5Dhw4FhAcTFsf6eVl/jxg3xg9P5+bmOJgaQEDRUfpB1kAJSKCTQBIeRD2YuIh6zM/Ph/mbqbOs1+UhgNBgjA4cOBAYt1arFRQc5RmfSVii6JgEZfuQgARGTgDhQUieSYtzJjJSjDHEaIqxfAwQGowRH4aLFy+6cRQQDUuVFR0NGyfdlYAECgggOJL4QIAsLCyEuiRcrosv+IHQYIwYL3wzNY+AoqN5Y67HEpCABCQggVEQGLgNRcfAyKwgAQlIQAISkMAwBBQdw1CzjgQkIAEJSKCIgPmFBBQdhWi8IQEJSEACEpDAKAkoOkZJ07YkIAEJSKCIgPkSCIoOPwQSkIAEJCABCUyEgKJjIpjtRAISkEABAbMl0CACio4GDbauSkACEpCABKZJQNExTfr2LQEJFBEwXwISqCEBRUcNB1WXJCABCUhAAmUkoOgo46hokwSKCJgvAQlIoMIEFB0VHjxNl4AEJCABCVSJgKKjSqOlrUUEzJeABCQggQoQUHRUYJA0UQISkIAEJFAHAoqOOoxikQ/mS0AClSJw4MCBnvYud79nZW9KoAQEFB0lGARNkIAEJACBdevWhfvvvz90iguuY4wUMUmg0gSaKDoqPWAaLwEJ1JdAq9UKV65cCQcPHgxr167NHI0xhrfffjs7379/f3b0jwSqSkDRUdWR024JSKB2BNavXx/efffdcNddd4WrV6+2/fv555/D7Oxs+9oTCVSVwG3RUVUPtFsCEpBAjQhs37493H333Us8MsqxBIkZFSSg6KjgoGmyBCRQXwJEO/bu3RtWr17ddtIoRxtF7U/q7qCio+4jrH8SkEDlCHRGO4xyVG4INbiAgKKjAIzZEpCABKZFIEU76N8oBxRMdSGg6KjLSOqHBCRQKwJEO3DIKAcUTHUhoOioy0jqhwQkUBsCly5dCkeOHAlEPHbv3h3m5+e7+mamBKpGQNFRtRHTXglIoLYEEBuIjJmZmczHubm5tvAgX/GRYfFPhQkoOio8eJouAQl0I1C9PJ44umHDhoDYILpx8eLFwLIK5xwRH5s3b86eVEo5ylfPSy2WQAiKDj8FEpCABKZAgKgG4iHGGA4fPhwQF0lsdJqD+Ni1a1dAfJCoq/jopOR1FQgoOqowStoogREQsIlyEEAwIDYQDViE0CAhKrheLiFADh06lAkQytKOSy+QMFWBgKKjCqOkjRKQQOUJIDYQByyh4MzCwkIW3UBEcD1ooh7RESIfnNM2aX5+ftCmLC+BiRFQdEwMtR2Vk4BWSWC8BFJUA7GBOCCqgVgYVa+0SXuID/d9jIqq7YyLgKJjXGRtVwISaCwBohqIjRiX368xKkiID5ZoEB8kbGDpBTtG1YftSGClBBQdKyVY0/q6JQEJDE6AiZ5Jnsme2kQ1SIgBrieVECDu+5gUbfsZhICiYxBalpWABCTQhQBig/0ULKFwe6X7NWhjFAnxkZZeOMdGkvs+RkHXNoYhoOgYiJqFJSABCdwmkKIaiA0mdaIaTPK3S5TjDNuwi2UX932UY0yaaoWio6kjr98SkMBQBIhqIDZinNx+jaEM7VIJ8cFSD+KDhC8sBeFPl+JmSWDkBEYiOkZulQ1KQAISKBkBJmgmZyZpTCOqQWIS57pqCQHivo+qjVr17VV0VH8M9UACEhgjAcQG+yBYQqGbsuzXwJZRJMRHWnrhHF9J7vsYBd2JtlGJzhQdlRgmjZSABCZNIEU1EBtMxkQ1mJwnbcek+sNH/GPZxX0fk6LevH4UHc0bcz2WgAQKCBDVQGzEWL39GgUuDZyN+GDJCPFBgglLSnAZuLEyVNCGUhFQdJRqODRGAhKYBgEmViZVJlf6J6pBYvLluqkJAeK+j6aO/nj8VnSMh6utSkACFSCA2GD/AksomFu3/Rr4VJAGykZ8pKUXzmFGct/HQBgtfJOAouMmBP8vAQk0i0CKaiA2mESJajCpNovC4N7CCk4su7jvY3B+1ghB0eGnQAISaAQBohqIjRh77NdoBImVO4n4YOkJ8UGCLUtT8F1567ZQZwKKjjqPrr5JQAKBCZHJkEkRHEQ1SEyaXJtWRgAB4r6PlTFsUm1FR5NGW18lMByBStZCbLDvgCUUHHC/BhTGlxAfaemFc9iT3PcxPuZVbFnRUcVR02YJSKCQQIpqIDaY/IhqMBkWVvDGSAnAHN4su7jvY6Roa9GYoqMWw6gTUyFgp6UhQFQDsRGj+zXKMiiID5awEB8kxoglLsapLDZqx+QJKDomz9weJSCBERFgImMSYzKjSaIaJCY7rk3lIIAAcd9HOcZi2lYoOqY9AvXrX48kMHYCiA32C7CEQmfu14BC+RPiIy29cM4Yktz3Uf6xG5WFio5RkbQdCUhg7ARSVAOxwaRFVINJbOwd28FICTB2jBvLLnXc97Fz587wySefjJRZXRpTdExqJO1HAhIYigBRDcRGjO7XGApgiSshPlgKQ3yQGGuWyhjvEputaSsgoOhYATyrSmDUBNI3pC+++CKsWrUqxBgDeaPupwrtMQEx+TAJYS9RDRKTFNemehFAgFRt38c777xzx3+f/LeaIhzbtm3L/vuNMQbK1Wu0hvdm2qJjeMutKYEaE3j88cfDjRs3AnsVcLNJ/2ghNljnZwkF32FAKJ5JiWtTvQkwzow3kQ/O+SyQyrjvY+/eveGRRx7JRMWPP/4Yrl27Fh588MFw5MiR7L9dPrsHDx6s94AN6J2iY0BgFpfApAns2bMnfP7554F/1Cbd9yT7S1ENxAaTDVENJp9J2mBf5SHAZ4DxR3yUdd8H/02eOnUqvPnmm+Ef/uEfsqffIjryFM+fP58Jk3xe+c4nZ5GiY3Ks7UkCEuggQFQDsRGj+zU60Hh5iwDigyU1xAeJzwxLbnxubhWZ2gF76PzcuXPh2LFjAQHywAMPkJUlllouX74cENJZhn984ZufAQmUncD7778fnnrqqbB69eqym9q3fUwcTBpMHlQiqkFicuHaJIFuBBAgk9z30c2GfB5RjK1bt4bf/OY34fXXXw/79u3L3w7Hjx/P8uv03+4dDg5xYaRjCGhWkcC4CeQ3ktIXa8ccq54QG6zPp29+rHkTQmcyqbpv2j85Anxe+NwQaeCczxRpmvs+nn322WwZhc2kkGDphSgH56bbBBQdt1l4JoHSEMhvJGVTWmkMG9KQFNVAbDBJENVg4WhHhgAAEABJREFU0hiyOatNlUB5OuezxOcI8TGNfR98GSAlIpyn/16JbiCCECPpvsfg8oofAglIYDwEiGogNmJ0v8Z4CNtqIoD4YGkO8UHis8fSHZ+/VMZjOQgY6SjHOGiFBDICfEuq+jcj/sHnH3v+0ccpohokJgWu65r0qxwEECBl2vdRDirlsULRUZ6x0BIJVJoAYoN1dZZQcKQM+zX49UCMMXtIEw9bY68MtrHe/txzz4V0zZFr8lmT57ko33//fVi3bl1Wl3aol0+UbbVa2eOuuR9j934oE+PiPdrOt5HO6R/7YlwsR3vcS33EuJifr0+dTZs2hW+//Tbk+4hxsSx5/+t//a/s4VWUTe3jG213JvqMcbFujDHzHQaU4x71aYdrjvTNfWyK8XY9+sVuyk0zrV+/PqSlF875bJJY8pimXU3vW9HR9E+A/ktghQRSVAOxwT/uRDX4x36x2en+JWqE+CExOb/88suZ0GC9/b777gvfffddZiDHq1evZufpDz99ZCPgyZMnw3vvvdfzOSn5fvjp5PPPPx+YkOmHSY7+r1+/Hmiv26Sf38NDOfpjou+nPmIg9UE/2Mv+hk8//TTwywoeWPXSSy+FM2fOZP4ePXo0Y5D8zB+pR/+088EHH4QdO3ZkfuPfG2+8kf0SIwkKxvqee+6540FY1KW9Dz/8kEMpEnbyeWTZBf/S55VjKQxsmBGKjoYNuO5KYBQELl26FPhHO8bq7NdARDCJfvbZZxkCniTJTx6zix5/eNjTmjVrepS48xbi67HHHssm+PwdBAQ/q1yuz1SOn1sOU7+zDj+3JiqBsIHBo48+usS2fJ10/sQTTwSE2IULF7IseJ0+fTogKGCIaMPW7OatP1zjI8+rSOLk1q2pHxAfu3btCogPEp/hDRs2ZJ/jqRvXIAMUHQ0abF0dH4EYb4eXY1z+PMZql+Ef69nZ2cDaOZEN/jEfH93RtpwmfSbRdE4PLKUwaXK+0sRkTfSkWztEO/qZkIvKFeV36yvlbd++PZ0OdWQ55YUXXsjq8iwKnsDJk3LJYHklH70ZVKTRxqQTAoTPLunw4cPZElqM0/lvctK+T7s/Rce0R8D+a0OAkHRTEkID0XHgwIFQ1XVyJkeWHhAAiA9ESG0+jDlH/vKXv2TvA8llDXyKgPrbv/3bwPLJiy++GDjfuHHjwO2UpQJRDj63JATztP67LQuPSdqh6JgkbftahoC3q0KAb4pVXydHdBCR+POf/xzY57Bly5aq4O/bTvaVfP311ysWHYiyFAniBWaITpYo+jakJAURyUTpWALjM4wffI5LYl4jzFB0NGKYdVIC4yHAP9x8U2QCIvENkn/U+cd9PD2OrlU2QdIaex3Y68GR65SYsJlo096GlN/tyPJCjIvhedpl78O2bdvCxx9/HGgjxsV75HHvmWeeyTZodmsrn/fXv/61sP5PP/2UL9r1nAhF2o/SarWyZQRs6lr4Zia2YX+MMRMq33zzzc3cxf+nSBD7QthkynILyy6Ld8v7l88kn8cYq7P/qLw0V26ZomPlDMfegh1IoAoEECCskSM+sBfxQfh6fn6ey1Kk9MRIRAKTK7/uILROfjKQZ6VwzeTK/gnu40Ov/R6Up1xnYikitZG/t1x7/FqEMg8//HD2i5d8Xc659/TTT4dz584F7Ey2c0x1sRchdeLEiawMdahLogxl84k87uXTjRs3Am3gHymVp2y6l3ile5SnT/pPedM4JrHB55D+iWqQEMlcm6ZDQNExHe72KoHaEkB8ELJGfHCO8CAx6ZXFaSZQJleOZbFJO0ZDALHB540lFFpknPk88lnk2jRdAhUWHdMFZ+8SkEBvAvwjzz/2iI86Ph+Bb/IIKb719ybh3UkQYAmFqAZig88eUQ0+f5Po2z76J6Do6J+VJSUggSEIMAEQ0kZ8kPgmyuTAJDFEc1aRQJsAnyU+RzG6X6MNZdInA/an6BgQmMUlIIHhCSBAyr7vY3jvrDkpAklsIF7pk6gGCXHLtam8BBQd5R0bLZNAbQkgPgh9E/ngnDV4EssVtXVax1ZMALHB54QlFBor8X4NzDN1IaDo6ALFLAlIYDIEEBxJfNRx38dkKNa/F5ZQiGogNvjMENXgc1N/z+vnoaKjfmOqRxKoHAEmEkLjRD5IfKNlkmGyqZwzGlxMYIA7fAYY/xjdrzEAttIXVXSUfog0UALNIoAAcd9Hs8Y8720SG4hO8olqkBClXJuqTUDRUe3x03oJ1JYA4oMQOpEPzlnLJ9Vw30dtx3AQxxAbjC9LKNRzvwYU6pcUHfUbUz2SQK0IIDiS+HDfR62GNnOGJRSiGogNxpqoBuOd3fRP7QgoOmo3pDokgZoQ6HCDCYkQO5EPEt+MmayYtDqKellyAowd4xaj+zVKPlQjN0/RMXKkNigBCYybAALEfR/jpjz69pPYQCzSOlENEmKSa1P9CSg66j/GelgvAnqTI4D4IBRP5INz9gSQ3PeRg1SCU8QG48ISCua4XwMKzUyKjmaOu15LoFYEEBxJfLjvozxDyxIKUQ3EBmNEVINxKo+FWjJpAoqOSRO3v/EQsFUJ3CTAxEaonsgHiW/YTHpMfjdv+/8JEIA5vGN0v8YEcFeuC0VH5YZMgyUggX4IIEDc99EPqdGUSWIDkUeLRDVIiECuTRKAgKIDCvVNeiaBxhNAfBDSJ/LBOXsLSO77GM1HA7EBT5ZQaNH9GlAwFRFQdBSRMV8CEqgVAQRHEh/u+1j50LKEQlQDsQFbohrwXXnLtlBnAs0UHXUeUX2TgAR6EmCCJORP5IPEN3UmTybRnhW9GWAFpxjdr+HHYTgCio7huFlLAhKoAQEEiPs+lh/IJDYQZ5QmqkFCvHFtkkC/BPKio986lpOABCRQKwKID5YGiHxwzh4FUtP3fSA24MASCgPufg0omFZCQNGxEnrWlYAEakUAwZHER5P3fbCEQlQDsQETohpwqdVgl9aZehum6Kj3+OqdBCQwBAEmWpYOiHyQ+MbPJMxkPERzlaiCj/gXo/s1KjFgFTVS0VHRgdNsCUhgMgQQIHXe95HEBqIKokQ1SIgursuStKMeBBQd9RhHvZCABMZMAPHBEgORD87Z60Cq6r4PxAb2s4QCOvdrQME0bgKKjnETtn0JSKBWBBAcSXxMf9/H4GhZQiGqgdjAF6Ia+DN4S9aQwOAEFB2DM7OGBCQggcCEzRIEkQ8SkQMmcyb1suHBNuyK0f0aZRubptmj6GjaiOuvBBpAYNIuIkDKuO8jiQ3EEEyIapAQS1ybJDBpAoqOSRO3PwlIoLYEEB8sVRD54Jw9E6RJ7/tAbNAvSyjAdr8GFExlIKDoKMMoaIMEJkLATiZFAMGRxMck932whEJUA7GBDUQ1sGNSftuPBJYjoOhYjpD3JSABCQxJgImfpQwiHyQiEIgCxMGQTS6pRpu0F6P7NZbAMaN0BBQdpRsSDZo0AfuTwCQIIEBGue8jiQ1EDPYT1SAhcrg2SaCMBBQdZRwVbZKABGpLAPHBkgeRD87Ze0Hqd98HYoPyLKEAyf0aUDBVhYCioyojNXE77VACEhgnAQRHEh/97PtgCYWoBmKDukQ1qD9OG21bAqMmoOgYNVHbk4AEJDAAAQQESyJEPkhEMhAXiAzOOcbofo0BkFq0xAQUHQMOjsUlIAEJjIsAAqRz3wd9EdUgIU64NkmgqgQUHVUdOe2WgARqSwDxwdKJ+zVqO8SNdWxEoqOx/HRcAhKQgAQkIIE+CSg6+gRlMQlIQAISkECpCVTAOEVHBQZJEyUgAQlIQAJ1IKDoqMMo6oMEJCABCRQRML9EBBQdJRoMTZGABCQgAQnUmYCio86jq28SkIAEigiYL4EpEFB0TAG6XUpAAhKQgASaSEDR0cRR12cJSKCIgPkSkMAYCSg6xgjXpiUgAQlIQAISuE1A0XGbhWcS6JsA78PoVXi5+73qlvKeRklAAhIYAQFFxwgg2kTzCKxbty7cf//9oVNccB1jbB4QPZaABCTQBwFFRx+QLCKBTgKtVitcuXIlHDx4MKxduza7HWMMb7/9dnbOezOyE/9IQAISkECbgKKjjcITCfRPgBdyvfvuu+Guu+4KV69ebVf8+eefw+zsbPvaEwlIQAISuE1A0XGbhWejItCQdrZv3x7uvvvuJd4a5ViCxAwJSEACGQFFR4bBPxIYnADRjr1794bVq1e3KxvlaKPwRAISkMASAoqOJUjGlmHDNSTQGe0wylHDQdYlCUhgZAQUHSNDaUNNJJCiHfhulAMKJglIQALFBKYvOopt844EKkGAaAeGGuWAgkkCEpBAMQFFRzEb74yYwKVLl7LnWszMzIQNGzaEGGMtEr6AKsb6+INPjBPPHcE3kwQkUG8Ck/JO0TEp0g3vh8mLiQwMRATm5ubCwsKCqYQMGBsS48R4MW6MH+cmCUhAAishoOhYCT3r9kWAb8zz8/Ph4sWLgYmMB2uxF6KvyhaaOAHGhsQ4MV4IEIzYvXs3B5MEGkRAV0dNQNExaqK2dweBmZtLKWQwcTGRcW6qFgHGbefOnYGjwqNaY6e1EigbAUVH2UakRvakkDyCo0ZuNdIVBAfCA+cVHlBodtJ7CQxLQNExLDnrLUtgdnY2HDp0aNlyFqgGAYQHyy0slVXDYq2UgATKRkDRUbYRqYk9RDl27dqVheRr4pJu3CSA8Gi1WtmvkIL/6yDgpQQksBwBRcdyhLw/FAF+Hrt58+ah6lqp3ARYZjHaUe4x0joJlJWAoqOsI1Nxu5iU+EZccTc0vwsBoh2Iyi63umaZKQEJSCARUHQkEh5HSoBJiclppI3amAQkIAEJVJqAoqPSw6fx1SVQXcsRk4jK6nqg5RKQwLQIKDqmRd5+JSABCUhAAg0joOho2ICX3V3tk4AEJCCB+hJQdNR3bPVMAhKQgAQkUCoCio5SDUeRMeZLQAISkIAEqk9A0VH9MdQDCUhAAhKQQCUIVFp0VIKwRkpAAhKQgAQkkBFQdGQY/CMBCUhAAhKQwBAEBqqi6BgIl4UlIAEJSEACEhiWgKJjWHLWk4AEJCABCRQRML8rAUVHVyxmSkACEpCABCQwagKKjlETtT0JSEACEigiYH7DCSg6Gv4B0H0JSEACEpDApAgoOiZF2n4kIAEJFBEwXwINIaDoaMhA66YEJCABCUhg2gQUHdMeAfuvHYGdO3cGUt6xL774ImzatCl8//337ex33nknkMggf926dSHGGFqtVvjxxx/JLkzUy/fxySefLOmzsHJ1bmipBCRQMwKKjpoNqO5UgwAi5PPPPw+vvvpqJjB27NgRXnnllbCwsBAQH+QXeYLg2Ldv3x23n3322ewa8ZGd+EcCEpBACQkoOko4KJq0PAEiAUQEYoxZdCDG2I4aUJsoAJNzOo/xdgSB/Bhv14sxhlWrVgWEAOVTYgKnD/oij3q0y/lK0/vvvx9efvnlsHr16nD9+vVw9erVsGXLlqzZ7du3hx+DiAoAABAASURBVPn5+TuiItmNm3/o/9SpU+G//bf/dvPqzv/v2bMn/OEPf8hEzJ13vJKABCRQDgKKjnKMg1YMSeDkyZNZdICJ+9TNyRih0NnUkSNHsjJEEObm5sLevXuza6IK3333XXj00UfDmTNnwuOPP95ZdSzXLKVcvnw5PPHEE1n72MDJgw8+yCFwTLZlGbk/+IIgueeee3K5i6cbN27MTi5cuJAd/SMBCUigbAQUHWUbEe0ZisDq1avD1q1bw/Hjx3vWP3/+fM/7/d4k6hFjzJZCEBFEQ/77f//vfUUZEBlr1qwJSThw/a//+q/9dl1YjqjJfffdF2ivsJA3JCABCUyRgKJjivDtupoEWIZhcica8cEHH2SRiV/+8pfh7/7u77LlkuW8QhSwnJLKEdlYu3ZtuvQoAQlIoLYEFB21HdpbjnnI9kawtPLxxx+viEaKkrAM85//83/O2mIDJ+Ljxo0b7eWZRx55JLuX/4PQSNEMRAaRjnSfa84pk44xxkzMcG2SgAQkUBcCio66jKR+dCXAsge/DCEiwf6PJBy6Fu6S+S//8i+BPRIsobCXokuRJVmIDpZ5iIhwExvee++90Gq1wgMPPJCJCSId7EPhPsssiJDPPvuMy2yJKJXNMvr8Qz/Xrl3L2u+zisUkIAEJTJRAU0XHRCHb2fgIbNu2rf3rlc6fkdIrG0c5zszMZJMxkzKTM3nLJaIY/JKEDZ9PPvlkoK/l6nCfeseOHcs2isYYs70bRFqOHDnC7Ux4cH327Nnsmr0YR48eDR999FHmC5tMP/zww+wewqXz+R7ZjS5/EEdkpw2lnJskIAEJlImAoqNMo6EtfRNgoibywNJGPqWJnSO/UkEAUI7yLIucOHHijn0XRB7OnTvXXhrpNIB2aB8h8Pd///eB684y3a7pl3opddbr/HkrdtAH5ZO9tIvNtMV5PuFbZ5v5n+Hmy3ouAQlIoCwE7hQdZbFKOyRQcwKIiaeeeiqkiEaRuyzrcA9RwrEopZ8KdxMoRXXMl4AEJDBpAoqOSRO3PwncIkC0gnTrsusBsfHWW291vZfPRGx0Rj7y9z2XgASqQ6DOlio66jy6+iYBCUhAAhIoEQFFR4kGQ1MkIAEJSKCIgPl1IKDoqMMo6oMEJCABCUigAgQUHRUYJE2UgAQkUETAfAlUiYCio0qjpa0SkIAEJCCBChNQdFR48DRdAhIoImC+BCRQRgKKjjKOijZJQAISkIAEakhA0VHDQdUlCRQRMF8CEpDANAkoOqZJ374lIAEJSEACDSKg6GjQYOtqEQHzJSABCUhgEgQUHZOgbB8SkIAEJCABCQRFhx+CQgLekIAEJCABCYySgKJjlDRtSwISkIAEJCCBQgKKjkI0RTfMl4AEJCABCUhgGAKKjmGoWUcCEpCABCQggYEJjEx0DNyzFWpNoNVqhfn5+Vr72FTnLl26FNavX99U9/VbAhJYAQFFxwrgWVUCTSSAmGzdFJVN9F2fJVByAqU3T9FR+iGqpoE7d+4MR44cqabxWt2TwOnTp4109CTkTQlIoIiAoqOIjPkrIsA3Yb4Rk1bUkJVLRYCllcOHD4f9+/eXyi6NkUBPAt4sDQFFR2mGol6GsObPxGS0o17junv37jA7O1svp/RGAhKYGAFFx8RQN68joh18M2aiap739fN4ZmYmcwoxmZ34p+oEtF8CEyeg6Jg48uZ0SLTj0KFDAeHBhMWxOd7Xx1PGjfHDo7m5OQ4mCUhAAkMRUHQMhc1K/RJIwoOoBxMXUY/5+fkwfzP124blJk8AocEYHThwIDBurVYrNEZwTB63PUqgMQQUHY0Z6uk5ivAgJM+kxTkTGSnGGGI0xVg+BggNxohPzcWLF904CgiTBCSwYgKKjhUjtIF+CSA4kvhAgCwsLIS6JBjUxRf8QGgwRowXvoUQPEhAAhJYMQFFx4oR2oAEJCABCUhAAv0QUHT0Q8kyEigiYL4EJCABCfRNQNHRNyoLSkACEpCABCSwEgKKjpXQs24RAfMlIAEJSEACSwgoOpYgMUMCEpCABCQggXEQUHSMg2pRm+ZLQAISkIAEGkxA0dHgwdd1CUhAAhKQwCQJlEF0TNJf+5KABCQgAQlIYEoEFB1TAm+3EpCABCQggfIQmIwlio7JcLYXCUhAAhKQQOMJKDoa/xEQgAQkIAEJFBEwf7QEFB2j5WlrEpCABCQgAQkUEFB0FIAxWwISkIAEigiYL4HhCCg6huNmLQlIQAISkIAEBiSg6BgQmMUlIAEJFBEwXwIS6E1A0dGbj3clIAEJSEACEhgRAUXHiEDajAQkUETAfAlIQAKLBBQdixz8KwEJSEACEpDAmAkoOsYM2OYlUETAfAlIQAJNI6DoaNqI6+9ICBw4cKBnO8vd71nZmxKQgARqSkDRUdOBra5b1bB83bp14f777w+d4oLrGGM1nNBKCUhAAhMmoOiYMHC7qweBVqsVrly5Eg4ePBjWrl2bORVjDG+//XZ2vn///uzoHwlIQAISuE1A0XGbRanPNK5cBNavXx/efffdcNddd4WrV6+2jfv555/D7Oxs+9oTCUhAAhK4TUDRcZuFZxIYiMD27dvD3XffvaSOUY4lSMyQgAQkkBGouOjIfPCPBKZCgGjH3r17w+rVq9v9G+Voo/BEAhKQwBICio4lSMyQQP8EOqMdRjn6Z2dJCUigJgQGcEPRMQAsi0qgk0CKdpBvlAMKJglIQALFBBQdxWy8I4G+CBDtoKBRDiiYJCCBWwQ8dCGg6OgCxSwJ9Evg0qVL4ciRI4GIx+7du8P8/Hy/VS0nAQlIoHEEFB2NG3IdHgUBxAYiY2ZmJmtubm6uLTzIV3xkWPwjgaUEzGk0AUVHo4df5wclwBNHN2zYEBAbRDcuXrwYWFbhnCPiY/PmzdmTSilH+UH7sLwEJCCBuhJQdNR1ZPVrZASIaiAeYozh8OHDAXGRxEZnJ4iPXbt2BcQHibqKj05KXnchYJYEGkFA0dGIYdbJYQggGBAbiAbqIzRIiAqul0sIkEOHDmUChLK049ILJEwSkEBTCSg6mjry+l1IALGBOGAJhUILCwtZdAMRwfWgiXpER4h8cE7bpPn5+UGbal55PZaABGpFQNFRq+HUmZUQSFENxAbigKgGYmElbebr0ibtIT7c95En47kEJNAUAoqOpoy0fnYlQFQDsRHj8vs1ujYwRCbigyUaxAcJG1h6wY4+m7OYBCQggUoSUHRUctg0eqUEmOiZ5JnsaYuoBgkxwPWkEgLEfR+Tom0/EpDAtAkoOqY9AvY/OgJ9tITYYD8FSygUX+l+DdoYRUJ8pKUXzrGR5L6PUdC1DQlIoCwEFB1lGQntGCuBFNVAbDCpE9Vgkh9rp0M0jm3YxbKL+z6GAGgVCUig1AQUHaUenpEY19hGiGogNmKc3H6NUcFGfLDUg/gg4QtLQfgzqj5sRwISkMCkCSg6Jk3c/sZOgAmayZlJms6IapCYxLmuWkKAuO+jaqOmvRKQQDcCzRUd3WiYV2kCiA32QbCEgiNl2a+BLaNIiI+09MI5vpLc9zEKurYhAQlMgoCiYxKU7WOsBFJUA7HBZExUg8l5rJ1OsXF8xD+WXdz3McWBsGsJSGBgAp2iY+AGrCCBaRAgqoHYiLF6+zVGxQvxwZIR4oMEE5aU4DKqPmxHAhKQwCgJKDpGSdO2xk6AiZVJlcmVzohqkJh8uW5qQoC476Opo6/f9SNQX48UHfUd21p5hthg/wJLKDhWt/0a+DSKhPhISy+cw4zkvo9R0LUNCUhgpQQUHSslaP2xEkhRDcQGkyhRDSbVsXZag8ZhBSeWXdz3UYMB1YWMgH+qT0DRUf0xrJ0HRDUQGzE2d7/GqAYV8cHSE+KDBFuWpuA7qj5sRwISkEC/BBQd/ZKy3NgJMCEyGTIp0hlRDRKTJtemlRFAgLjvY2UMy1lbqyRQHQKKjuqMVW0tRWyw74AlFJx0vwYUxpcQH2nphXPYk9z3MT7mtiwBCSwSUHQscvDvFAikqAZig8mPqAaT4RRMaWSXMIc3yy513PfRyEHVaQmUnICio+QDVDfziGogNmJ0v0ZZxhbxwRIW4oPEGLHExTiVxUbtkIAE6kFA0VGPcSy9F0xkTGJMZhhLVIPEZMe1aVIEeveDAHHfR29G3pWABIYnoOgYnp01+yCA2GC/AEsoFHe/BhTKnxAfaemFc8aQ5L6P8o+dFkqgzAQUHWUenQrblqIaiA0mLaIaTGJldUm7uhNg7Bg3ll3c99Gd0bC5O3fuDJ988smw1a0ngUoSUHRUctjKaTRRDcRGjO7XKOcIDW8V4oOlMMQHibFmqYzxHr5Va0pAAk0joOgY8Yinby9ffPFFWLVqVYgxBvJG3M2EmuuvGyYgJh8mIWoQ1SAxSXFtqhcBBIj7Pvof03feeeeOfwP49yBFOLZt25b9GxFjDJTrv1VLSqCaBBQdYxq3xx9/PNy4cSOwh4Eu6vgPCmKDdX6WUPARXwnFMylxbao3AcaZ8SbywTmfBZL7Pu4c971794ZHHnkkExU//vhjuHbtWnjwwQfDkSNHsn8f+O/m4MGDd1bySgI1JaDoGGJgB62yZ8+e8Pnnnwf+wRm0bhnLp6gGYoPJhqgGk08ZbdWm8RPgM8D4Iz7c97GUN//dnzp1Krz55pvhH/7hHwJiHdGRL3n+/PlMmOTzPJdAHQkoOuo4qmPwiX8oERsxul9jDHhr0STigyU1xAeJzwxLbnxuauHgkE7Agqrnzp0Lx44dCwiQBx54gKwssdRy+fLlgIjPMvwjgRoTGKHoqDGlFbr2/vvvh6eeeiqsXr16hS1NvjoTB5MGkwe9E9UgMblwbZJANwIIEPd9LJIhirF169bwm9/8Jrz++uth3759izdu/T1+/HiWX8V/H2654EECfRNQdPSNarCC+Y2k1GRdl2NVEmKD9fn07Yt1Z0LoTCZV8UE7p0+AzwufG77tc85nitTUfR/PPvtstozCZlJGh6UXohycmyQwEgIlb0TRMaYBym8kZcPYmLoZebMpqoHYYJIgqsGkMfKObLBRBPgs8TlCfDRt3wdfOEhpwDlP/yYQ3UCAIUbSfY8SqDMBRUedR7dP34hqIDZibMZ+je+//z6sW7cu+6liq9Ua+wZfvtXGGLOfUBMBKxqWVC7G+v58EvHB0hzig8Rnj6U7Pn9FXMyXwAgI2ERJCCg6RjwQfIOpyrcW/sHnH3v+0QcDUQ0SkwLXdUyEs3fs2BFeeeWV7OeKiI9XX311bK7yU2nC59evX882ET7//PMB0dPZYb7cd999F44ePRp6CZTO+lW8RoC476OKI6fNEhiegKJjeHaVrYnYYF2dJRScaNJ+DSb/q1evhi1btuB62L59eyC83U0I8KsCHvDG5M95KxcVQSQQmcgayf1B1FAu3UubCAmjP/HEE1nJs2fPZsf0hzr8pPq9997LNhvzywZ+6cASXSpT5yPiIy29cM5nk8RIVIeVAAAQAElEQVS41NnvUvimERKYMAFFx4SBT7O7FNVAbPCPO1EN/rGfpk2T7psoAn2m5yRwRHSlfMQH0Y8YY+BXBTzgrZ/JHxESYwy//OUvA+KBiBdigigHD4aiz3vuuScQVUKIcJ1SEkKHDx/OlnxirO/ySvK525HPJJ9Hll2atu+jGw/zJFBHAoqOOo5qzieiGoiNGGNgUuMf9SaKjYQEcfGv//qv6fKOI8LhySefDH/605+ypReEwx0FulwkkcItxEtepCAmYM295RLl/u3f/i3rl0jIhx9+WPvllSImiI9du3YFxAeJzzBijc9xUZ0R59ucBCQwJgKNEx0xxva3yRjrf84/1rOzs4G1cyY2/jEf02epEs0S2Vi7dm1XW/lVAYID4RFjf+/MYSmEaAYNxnjnZtEU2eDecgmbeHIt5TZu3Bj+5m/+Jnz22WdcNjohQPjskhDNMdb/v9kY9THGZjBo4n/cjRMdDDLfSJuSEBqIDr4luk4esnde8Bkg4pGOMcZ2fhIRfD7Y75H2dFA2nzqXSBAs1Pnhhx+yBz2xp4N9HCzVpLIp8pGWW1J7iJM1a9aEZFPKL91xCgYR5eBzS0Iww9i0kEXE5FB9DlP4T2rqXTZSdEyd+gQN4JsiSyqEqV0nDyFN8CmKwL6NVqsVEBudw8KvkPLLJf/yL/8SLly4kP36ZH5+vrN4do3Q4F5amkFg8Nhr9newbEKhtKGUcxJ1eGIltnDNWNFX2uxKXtMSIpkoXZP3HzVtzPW3GQQUHc0Y54D44JsiExqJb5D8o84/7g1BkLnJBM/PUT/66KNsmY2lEfZPZDd7/EGAEPlAMLD8wivJbxXveSACQrQDsfPCCy+Ef/zHf8wEDntBNm3a1N63QTkaijEG2uYdHf1sYKVOXRKfST6PMbr/qC5jqh8S6CSg6Ogk0oDr9evXZ3s8EB+4i/ggfM03dK7rnohqIDYIT+MzQqQfn4leUIe6f//3fx+4HqRePmqCDZ0/i6U92ichcvppuw5lktjgc4g/LAmSEMlcmyQggfoQUHTUZywH9gTxkZZeOEd4kJiIB26ss4LXEliGAGKDzxtLKBRFbPF55LPItUkCEqgfAUVH/cZ0YI/4R55/7Il8uO9jYHxWGJAASyhENRAbfPaIavD5G7AZi0tAAhUkoOiY7KCVujcmAELaiA8S30SZHJgkSm24xpWeAJ8lPkcxul+j9IOlgRIYIwFFxxjhVrlpBAjPRkB84Afig1C4Sy/QMPVLIIkNPj/UIapBQtxybZKABJpFoByio1nMK+Ut4oPQN+KDc4QHSfFRqWGcuLGIDT4nLKHQufs1oGCSgAQUHX4G+iKA4Ejiw30ffSFrZCGWUIhqIDb4zBDV4HPTSBg6LYGKEZiEuYqOSVCuUR9MJITGiXyQ+EbLJMNkUyM3dWUAAnwGGP8Y3a8xADaLSqCRBBQdjRz20TiNAHHfx2hYVrGVJDYQndhPVIOEKOXaJIF6ENCLURJQdIySZkPbQnwQQifywTlr+ST3fdTzA4HYYHxZQsFD92tAwSQBCfRDQNHRDyXL9EUAwZHEh/s++kJWqUIsoRDVQGww1kQ1GO9KOaGxIyNgQxIYhoCiYxhq1ulJgAmJEDuRDxLfjJmsmLR6VvRm6QgwdoxbjO7XKN3gaJAEKkhA0VHBQauSyQgQ931UacQWbU1iA7FIDlENEmKSa1MRAfMlIIFeBBQdveh4b2QEEB+E4ol8cM6eAJL7PkaGeCQNITYYF5ZQaND9GlAwSUACoyKg6BgVSdvpiwCCI4kP9330hWwihVhCIaqB2GCMiGowTqPq3HYkIAEJQEDRAQXTxAkwsRGqJ/JB4hs2kx6T38SNaWiHMId3jO7XaOhHQLclMHECio6JI7fDTgIIkGbu++gkMZnrJDYQefRIVIOECOTaJAEJSGBcBBQd4yJruwMTQHwQ0ifywTl7C0ju+xgYZdcKiA14soRCAfdrQMEkAQlMkoCiY5K07asvAgiOJD7c99EXsp6FWEIhqoHYgC1RDfj2rORNCUhAAmMgoOgYA1SbHA0BJkhC/kQ+SHxTZ/JkEh1ND/VtBVZwitH9GvUdZT2TQPUIKDoqM2bNNhQB4r6P5T8DSWwgzihNVIOEeOPaJAEJSGCaBBQd06Rv3wMTQHywNEDkg3P2KJCavu8DsQEHllCA6n4NKJgkIIGyEai86CgbUO2ZDAEERxIfTd73wRIKUQ3EBkyIasBlMqNgLxKQgAQGI6DoGIyXpUtGgImWpQMiHyS+8TMJMxmXzNSRmYOP+Bej+zVGBtWGJCCBlRDou66io29UFiw7AQRInfd9JLGBqGIsiGqQEF1cmyQgAQmUnYCio+wjpH0DE0B8sMRA5INz9jqQqrrvA7GB/SyhAMP9GlAwSaACBDRxCQFFxxIkZtSFAIIjiY8q7vtgCYWoBmIDX4hq4E9dxkc/JCCB5hFQdDRvzBvnMRM2SxBEPkhEDpjMmdTLBgPbsCtG92uUbWy0Z2QEbKjBBBQdDR78JrqOACnjvo8kNhBDjAtRDRJiiWuTBCQggToQUHTUYRT1YWACiA+WKoh8cM6eCdKk930gNuiXJRSccL8GFBqadFsCDSCg6GjAIOtiMQEERxIfk9z3wRIKUQ3EBjYQ1cCOYku9IwEJSKD6BBQd1R9DPRgBASZ+ljKIfJCIQCAKEAcjaD5rgjZpL0b3a2RA+vtjKQlIoEYEFB01GkxdGQ0BBMgo930ksYGIwUKiGiREDtcmCUhAAk0hoOhoykjr58AEEB8seRD54Jy9F6R+930gNijPEgqdj3S/Bg2aJCABCVSMgKKjYgOmuZMngOBI4qOffR8soRDVQGxQl6gG9SdvuT1KQAISKBcBRUe5xkNrVkZgrLURECyJEPkgEclAXCAyOOcYo/s1xjoINi4BCVSagKKj0sOn8dMigADp3PeBLUQ1SIgTrk0SkIAEJHCbgKLjNov6nunZ2AggPlg6cb/G2BDbsAQkUCMCio4aDaauSEACEpCABMpMoMmio8zjom0SkIAEJCCB2hFQdNRuSHVIAhKQgAQkUE4CS0VHOe3UKglIQAISkIAEKk5A0VHxAdR8CUhAAhKoH4G6eqToqOvI6pcEJCABCUigZAQUHSUbEM2RgAQkIIEiAuZXnYCio+ojqP0SkIAEJCCBihBQdFRkoDRTAhKQQBEB8yVQFQK1Fx28D6PXYCx3v1dd70lAAhKQgAQk0D+B2ouOdevWhfvvvz90iguuY4z9k7KkBCRQMQKaKwEJlI1A7UVHq9UKV65cCQcPHgxr167N+McYw9tvv52d896M7MQ/EpCABCQgAQmMlUDtRQcv5Hr33XfDXXfdFa5evdqG+fPPP4fZ2dn2tScSaAoB/ZSABCQwLQK1Fx2A3b59e7j77rs5vSMZ5bgDhxcSkIAEJCCBsRJohOgg2rF3796wevXqNkyjHG0UnmQE/CMBCUhAAuMm0AjRAcTOaIdRDqiYJCABCUhAApMj0BjRkaIdoDXKAYX+kqUkIAEJSEACoyLQGNEBMKIdHI1yQMEkAQlIQAISmCyBrqLj0qVL2XMtZmZmwoYNG0KMsRYJX8Ab40r9KUd9/CExTjx3BN9MEpCABCQggbISWCI6mLyYyDCYiMDc3FxYWFgwlZABY0NinBgvxo3x49wkAQlIQAISKBuBO0QH35jn5+fDxYsXAxMZD9ZiL0S/RltusgQYGxLjxHghQLBg9+7dHEwSkIAEJCCBUhFoi46Zm0spWMbExUTGualaBBi3nTt3Bo4Kj2qNndZKQAISGBGBUjeTiY4UkkdwlNraEhn3ySefBCIMP/74Y4msCgHBgfDAKIUHFEwSkIAEJFAWApnomJ2dDYcOHSqLTdqxQgIID5ZbWCpbYVNWl4AEJFAPAnpRCgK/IMqxa9eu7BtyN4v4Jt9qtZb8eoW8P/7xj2HTpk3h+++/D3zz5xv2F198EVatWpWVf+edd7ImOVKetsjgmsR5Svl6MS7+OoR2yKcMx9RX/px2YoztqAN90FeMi23EuHjEPtpJKZUjn1Rke75c/px2Tp8+He65557M1xhjxoD8fMJW/IgxhnXr1mWsuN9pNwy5H+NtXyiHbTEu+hBjzNjSJvd6pfXr1wc4HDhwIPg/CUhAAhKQQBkI/IKfx27evLnQFh4dzjfm9AuWkydPBsp/+umn4T/8h//QrseekGvXroWXXnopnDlzJnz33Xfh6NGjgQny1Vdfzcql5Zvz589n1/k/jz/+ePjhhx/C7373u3D27Nns1zLHjh0Lzz//fHuizpdP5zze/Pr169kl7Xfai90vvvhidr/oTy/bi+o8++yzmY20T+rWB0Li5ZdfznhQ5oMPPgg7duwIiJdOux944IFw+fLlrM2tW7eGZ555JivX2c8bb7wRXn/99exekW0pHyHF2KVrjxKQgAQ6CHgpgYkS+AWTEt+IV9ork/1TTz2VRT4QEEyijz76aCY+uPfee++FP/zhD5mAQJxs2bJl2S4RA4899ljWRq/CtE+UoFeZXveoX2R7r3rpHiICwZCu0xHxBAN4kIc/a9asCRcuXOAyexdMN7sRaflyWeFbfxA3V69ebbdxK7vrgWgHorLrTTMlIAEJSEACEyaQRTqYnIbpl8n03LlzAYGR6qenfqbrdKQsE/uDDz4Y+PbPdbrXz5Hyqa/8ebe6fMNnWYJ7RFq++uqr8MQTT3DZMxXZniohThBpRB9SHkdEBEKgs49uER3KEQWiXlGin/vuuy8TW/iBKETYUJ7lnGHHi/omCUigDwIWkYAExkLgF6Ns9S9/+UtAVBS1yTd4vql3Ts5F5Veaz0TNUsRbb711hzDq1u5ytnerQ16vPh555BGK3JGIYPRiRGHaJBq0XDnK9kqIEyMdvQh5TwISkIAEJklgZKKD/Qtff/11T9HBN/g9e/aE//E//sdEfEx7PZYTOf3YXmRwUZSD8vRLlIVoC9fsOSHSsXHjRi4L04cffhj6KVfYgDckMB4CtioBCUhgRQRGJjpYMuBbPNawHBBjDB9//DGXIf1SI8aYLXOwoZJlg+zmrT+UYengyy+/7ClcbhVvH6gX4+2+2jdunmATk/fN057/p1yR7T0r3rpJRAHbb122Dyw7sY/lt7/9bfYLl9deey3bXIv46rQb4cP+jhhjOHXqVGCjLuXajXkiAQlIQAISqDiBgUUH+xnY19A5IbLP4sSJE9kyBvf5tQaJ8vxSg/N8Ij/PLpVhQyaTNffogzZpm+tuKdWj7c42qZf2gXSrm/IoRz/022k7NpDX2XZnXcqlvPyRtm/cuJH9KiXvW6fd9M19/KC/1B795q/Jx1bazffj+RQJ2LUEJCABCfRFYGDR0VerFpKABCQgAQlIQAIdBGopOo4cORKIEHT4esclEQMiCMuVu6OSF4MQsKwEJCABCUjgDgK1FB13eOiFBCQgAQlIQAKlIKDomPQw2J8EJCABCUigoQRqN23CdQAAEABJREFUKTp4zgW/oOn8hUxDx1i3JSABCUhAAqUgUBbRUQoYGiEBCUhAAhKQwPgIrEh0EEkgosCL2jjGeOebVLnPI8l5OFZ60yrPp+h0J0UmYlx8myp1UhnaiHExP8Y720/tUobyPOuCN9HyrA+ut23blj0fI8aYvXE12ZnKUyaf6DfGxb6SnamPGBfzU136Ss/ViPFOu2gzX49ylO/Mj3GxzXSf/uEID8piA3mcmyQgAQlIQALjJTD+1lckOpJ5CAp+CcIzJnjw15NPPpm92I0XnPE475deeimcOXMme5fI0aNHAxNyqssx/ZKE+iQeH56fbDdv3hx4uij3aH/Hjh2BiZlnVRw7dizw0K00qfOQr4cffjgke6jDi9fIp6+ixATPPcrTFw/oQmDQR3rOBvm8uI58yq5duzbQNnWwK/mNLbxfBp/TvbzNqT3u8WCyDRs20FzgSaScpCPnsOBokoAEJCABCVSdwEhERx4CP0FttVrZZIyY4CVvRB+YvHkAFm9dZaLN1+k85/0sPCiLybvzHo8W5ymjPH6ce7yf5MqVK2Hfvn1Zn+R1Ph2U/qjDvW4JAfP5558HHtHOfezmnS3Hjx/nsp3y+fTBk0jTzeQ3T2FFiOAnPnMf8YXoSTaT1y3RPlGPN998MyCaED6Kjm6kzJOABCQwOQL2NDoCIxcdmMZEef78eU6ztH379uw4yB9EAmKhVx1EyfPPPx9++umn7JHrLKcgFpi88/UQD1u3bg1EZPL56ZwIRrcXoyF8ECSpXDqSzzmCChGUoiT4TX7ed65JyR9sbt0UZRzJR6QgNBBktMM1/iBUuJ+OnJskIAEJSEACVSYwFtHBpJsm4GHf3toPVEQJSxwcDx48GH79619n73bJ12UpBJHAG27z+fnzzqhFuocY6BQw6R7H9Chzjlwnv5Pv5KVEpIOoTLrmyDITy03YzjX1OccffEEo9eqfOiYJSEAC0yFgrxIYnMDIRQeTPEsDRAD4Nv/1118P9AK3QVxgcmYyRzQgKtgbQaQg30aKcvSavLlH1OL999/PqhLdYO/GIBEaohTJb3z/6quv2ntX5ubmApGOjRs3Zu2nP9jPMgxRDvpEHCFYuGaPyEcffZTtjUnlPUpAAhKQgASqTGAkooONkSwZxBizTZ1/+tOfshe/MakiCgCU7neKAu6tJKVoBMIBocB+CIRPvk0m8vx1t/MUrYgxBkQMUQb2aXQrS17e5xhjQCAkvxENRW+XpW5KRDbSOUdYpWgIfcMsbUDlvkkCEig/AS2UgASKCYxEdLBXgm/5/BqDb+tMunTJRsoTJ05kAiTdpwwTKveLEvXPnTsXqE9Z6iIqKN95j/eskE+iPGKAOlyTuJ+uaYO20jX384my2EdKIiR/n3Pq0sYvf/nLwJGypLzflEu2dLvHfRJ90Cfn2AYr6nFN4h59cI9rkwQkIAEJSKDKBEYiOqoMQNslIIFJELAPCUhAAiGsSHSkb/1+E+//o0SkhugFx/5rWVICEpCABCRQfQIrEh3Vd18PJDBdAvYuAQlIoEkEFB1NGm19lYAEJCABCUyRgKJjivDtuoiA+RKQgAQkUEcCfYkO3oPCcyjqCGAYn+QxDDXrSEACEpBA0wn0JTqaDqks/muHBCQgAQlIoMoE+hYdvFAtxhhijIEHcvEcDV7kxqO8eZrmc889lz2BM3+eB8MDu2JcrM9zPajHfcq3Wq3A/XQe42K5GGMgqpDKpT7Ii3HRDp56yn3qx3i7Xr4P7qeUL5cvk/rmPrZxL8bi9jp5JDtSP93aoG3uc4xxsW36oSz5HLmOcfEefubLxninz9QxSUACEpCABKpCoC/RwUOqeMgViaeM8rhxfiab3rLKC9N4zDdP0+S828vT+Hkt9Unffvtt4NXvTLJ5ULTJz0kpQ6Ivrpl4uUcdXpNPHe6/8sorYd++fVyGfPvce+ONNwIvf0NMZAVu/cmXK7KDB3T98MMP4Xe/+104e/ZsoL1jx44FXi6HuOjG41bz7QNt8KAy6pJOnjyZPa2V+kU2UAebeO8K5emHBjdv3hzgSjs8Ht2nlELFJAEJSEACVSPQl+jIO4UQQGDk85iYL1682M7KP867nZk74RkVTJyfffZZLnfpKeUQFrw/hbu80yTGGPbs2cNl2LJlS7h27VroFBbcfPHFFwN2Xrhwgcuuifb7sYPKvO31scceC/jPdUpc00+6LjpSf8OGDZmIyZfptAGWlKN8vlw6hwH99fIrlfUoAQlIQAISmAiBPjsZWHQgFHhJ2a9+9atsUmcSfOGFF8I//uM/Zo8773cSxj7ePUIEg2gG3/4RD61bSy3cJ3W+N4W+O1+cRjmiIdSlDa55f0qKxHDdK2FHr/u97iUeiId8OaI4LD8R2SAfP1mW4rxbWokN3dozTwISkIAEJFA2AgOJDibSo0ePhoMHDwYmUV6KRkSBZQSWBnAO0YEw6JyEubfSRNv33ntv1vdK2xpF/TyPUbRHGyxRESni3CQBCUhAApUmoPEdBAYSHZ3f6lnemJubyzaQ0i7f6l977bWwfft2Lkee6P+hhx4aebvDNog94xBY7IlhD8ewdllPAhKQgAQkUEYCA4kOHMgvdxDd+PDDDwNLLDHGwLd0NjqyVELZUSeWIOifpZQYY9Yv+xtG3Q/PJGF55ssvv8x86tU+9vS67z0JSEACEugg4GVjCQwkOvbu3RtIeVoIDH5VkRLX+ftF57Rz5MiRO26zZJP2d6QbtJfKceSalPqjPPXIS+fUJe/EiRMBYcR1UUp2UJ76tEMe7V++fDnbp0Jd7ne2RzkS9zsT/fKz4vwyU7K/syxtcI/8znrYg130z33ao13KcW2SgAQkIAEJVIXAQKKjKk5ppwQkIIEKEtBkCdSegKKj9kOsgxKQgAQkIIFyEFB0lGMctEICEigiYL4EJFAbAoqO2gyljkhAAhKQgATKTUDRUe7x0ToJFBEwXwISkEDlCCg6KjdkGiwBCUhAAhKoJoHKiQ4ec87jznlWB08Ezb+VNcbbb2HlDa0xLr6tNcbFI3kME208d+utuOTFeLse91NejIv16I863DOVnIDmSUACEpBAaQlUTnTkSfKsCh7BzjM1SDwmnZelUYbnXvCmVh7Tzj2e8MlzN3jwF8+8ePnll8NLL71E0ewtsrxYbt++fdk1dalDoh6ZPASNo0kCEpCABCQggeEIVFp0DOIyQoNX3fNUU+qlp6gu98baVO/UqVNd32ZLWxVImigBCUhAAhKYOoFfsHTAEy+nbskQBvCuF+znSPWPP/448CZXntrJ9XKJ96Z0e2MtyytERFJ9Hu9exZew8Q6Xft+0m3z1KAEJSEACEhgXgdpEOtjfkd6A2w8slmLuvffeQCSjn/JVLIOYRJRV0XZtloAEJCCB+hH4Bd/q2cNQddcQEUQu+o1y8IbYMr2xdhz8T58+HYx0jIOsbUpAAhKQwDAE2ssrfCu+1UAlD2mvBsaz3MIyy7Zt28IjjzxC1pJEee7xK5gYY2CPx9WrV5eUq2oGSyuHDx8O+/fvr6oL2i0BCUhAAjUj8Au+CTMxVT3akX9TK9EOfqnCr0/IT2PGG1uTnxy5JlGOND8/ny23cC9fj1/J8IbZKi3F7N69O8zOzibXPUpAAhKQQKUI1NPYbE8H6/58M2aiqqebzfJqZmYmcxgxmZ34RwISkIAEJFACApnoINpx6NChgPBgwuJYAtu6mkC0gYgEEQoiGpxz7Fq4YZmMG+OH23NzcxxMEpCABGpFQGeqTSATHbiQhAdRDyYuoh5M6CTum8pJAKHBGB04cCAwboyfgqOcY6VVEpCABJpOoC06AIHwICTPpMU5ExkpxsXHgcfoMcZyMUBoMEaM38WLF904CgiTBBpHQIclUA0Cd4iOZDKCI4kPBAibLOuS8LEuvuAHQoMxYrzwzSQBCUhAAhIoK4GuoqOsxmqXBCQggUEIWFYCEigXAUVHucZDayQgAQlIQAK1JaDoqO3Q6pgEigiYLwEJSGA6BBQd0+FurxKQgAQkIIHGEVB0NG7IdbiIgPkSkIAEJDBeAoqO8fK1dQlIQAISkIAEbhFQdNwC4aGIgPkSkIAEJCCB0RBQdIyGo61IQAISkIAEJLAMAUXHMoCKbpsvAQlIQAISkMBgBBQdg/GytAQkIAEJSEACQxIYsegY0gqrSUACEpCABCRQewKKjtoPsQ5KQAISkECjCJTYWUVHiQdH0yQgAQlIQAJ1IqDoqNNo6osEJCABCRQRML8EBBQdJRgETZCABCQgAQk0gYCiowmjrI8SkIAEigiYL4EJElB0TBC2XUlAAhKQgASaTEDR0eTR13cJSKCIgPkSkMAYCCg6xgDVJiUgAQlIQAISWEpA0bGUiTkSkEARAfMlIAEJrICAomMF8KwqAQlIQAISkED/BBQd/bOypASKCJgvAQlIQAJ9EFB09AHJIhKQgAQkIAEJrJyAomPlDG2hiID5EpCABCQggRwBRUcOhqcSkIAEJCABCYyPgKJjfGyLWjZfAhKQgAQk0EgCio5GDrtOS0ACEpCABCZPoDyiY/K+26MEJCABCUhAAhMkoOiYIGy7koAEJCABCZSZwLhtU3SMm7DtS0ACEpCABCSQEVB0ZBj8IwEJSEACEigiYP6oCCg6RkXSdiQgAQlIQAIS6ElA0dETjzclIAEJSKCIgPkSGJSAomNQYpaXgAQkIAEJSGAoAoqOobBZSQISkEARAfMlIIEiAoqOIjLmS0ACEpCABCQwUgKKjpHitDEJSKCIgPkSkIAEFB1+BiQgAQlIQAISmAiBRoqOGGOI0RSjDGKcNgP7j1EGMcogxuYxmMgsX7JOGic6FhYWgkkGfgb8DPgZ8DMw7c9AyfTARMxpnOiYCFU7WTEBG5CABCQggfoRUHTUb0z1SAISkIAEJFBKAoqOUg5LkVHmS0ACEpCABKpLQNFR3bHTcglIQAISkEClCNRCdFSKuMZKQAISkIAEGkpA0dHQgddtCUhAAhKQwAgJ9NWUoqMvTBaSgAQkIAEJSGClBBQdKyVofQlIQAISkEARAfPvIKDouAOHFxKQgAQkIAEJjIuAomNcZG1XAhKQgASKCDQy/8CBAz39Xu5+z8oVuanoqMhAaaYEJCABCVSbwLp168K///f/PnSKC65jjNV2rk/rFR19grKYBCQggbETsINaE2i1WuH//b//F2ZnZ8O/+3f/LvM1xphdc7F//34OtU6KjloPr85JQAISkEBZCKxfvz68+uqrmTk///xzdkx/ECLpvM5HRUedR1ffJFAPAnohgdoQ+C//5b+Eu+++e4k/TYhy4LSiAwomCUhAAhKQwAQIEO14+eWX7+ipKVEOnFZ0QMEkgSoS0GYJSKCSBDqjHU2JcjBYig4omCQgAQlIQAITIpCPdjQpygFeRQcUTHUioC8SkIAESk+AaAdGNinKgb+KDiiYJCABCUiglAQuXbqUPddiZmYmbNiwIcsubucAAA3ZSURBVMQYa5HwBeAx1scffGKceO4IvnVLio5uVOqYp08SkIAEKkaAyYuJDLOJCMzNzYWFhQVTCRkwNiTGifFi3Bg/zvNJ0ZGn4bkEJCABCZSCAN+Y5+fnw8WLFwMTGQ/WYi9EKYzTiCUEGBsS48R4IUAotHv3bg7t1HTR0QbhiQQkIAEJlIPAzM2lFCxh4mIi49xULQKM286dOwPHvPBQdFRrHLVWAhKQQK0JpJA8gqPWjjbAOQQHwgNXk/DoLjooYZKABCQgAQlMmMDs7Gw4dOjQhHu1u3ERQHiw3MJSGX0oOqBgkoAEJCCBqRMgyrFr164sJD91Y0pgQF1MQHi0Wq3sV0iKjuD/JCABCUigDAT4eezmzZvLYIo2jJgAyyxEOxQdIwZrcxKQgAQkMBwBJiW+Efeu7d0qEiDagahUdFRx9LRZAhKQQA0JMCkxOdXQNV26RUDRcQuEBwlIQAJVJqDtEigzAcQkolLRUeZR0jYJSEACEpBAjQgoOmo0mLoiAQl0EvBaAhIoEwFFR5lGQ1skIAEJSEACNSag6Kjx4OqaBIoImC8BCUhgGgQUHdOgbp8SkIAEJCCBBhJQdDRw0HW5iID5EpCABCQwTgKKjnHStW0JSEACEpCABNoEFB1tFJ4UETBfAhKQQFUI8Ljtd955pyrmDmXnjz/+GHhy6yeffDJU/WlWUnRMk759S0ACEpCABBpEQNEx9GBbUQISkIAEJkkgRTG+//77sG7duhBjzBLn5CVb9u3bl+XHGLNy+XuUIUKwatWq8MUXX3CZHTdt2hS+/fbbLIIQ42K79JcVuPmHstSJcfFejDHQzs1bS/5Pfr4u5+TRBv1gD3kxLrZF1ILoBQ3ly3BOn9TlHvWo/+WXX3IZtm3b1vYz3wY3aY+8GBf7oD/ySZynaBBHEvkcSfRHmdR/jDGQT5nOdmOMgbLc6ycpOvqhZBkJSEACEigVgbVr14azZ8+GhYWF8MEHH4Qnn3wyMCkfOXIkyyP/u+++Cxs2bFhi97PPPhveeOON8PrrrwcmUQqsX78+/OpXvwq8dI66169fD5cvX25Pto8//ni4ceNGu+2TJ0+G1157LeuT+vn04IMPhmvXrmVt0z7t5O9zfiRnJ3Y888wzWXnupUSfx44du6OfNWvWhIcffrhtJ7bCgfxUj+Pq1avvKPPII4+0xQHnlCGdP38+pGvOyZuZmcnsf+mll8KZM2cCHI8ePZqJs852uQczhAp1l0u/WK7AoPctLwEJSEACEhgngXvuuScgElIfiIhWqxU+/vjjlJUdmRCvXr2anXf+YaI9ffp0+PDDD8Nnn30W7rvvvsCEmspxjhhIE3HKT0cmZgQNE37KS0dEB/0iXFJeryNtIRouXLiwpBhtXblyJRC9SX3hf75gLz9TuRdffDETUQizlMc5giFdc4QLvj/11FOBqArC54EHHgiPPvpoJj4ok0/ce+WVV8Lx48fz2YXnio5CNN6QgAQkIIEyESA6sHfv3kwcMCk+8cQT7UgEk2WnrYgJJksmxvw9lg1eeOGFLIvJ/M033wx79uzJrjv/EKUgWkEdJmEmasowMbOsw3lnQhSsWbOmPUlzjnjoLJe/RqQgHpjkz507F7CZvp5//vnw008/ZYKK5RSEEH3n6zLhb926NeNyK7/rIfXBTfzGJgQDog0f8ZV7KW3fvj2dto+UQ+DlIxvd2LcrdJwoOjqAeCkBCUhAAuUngPhgaYEj1hKRyE9+iASWBA4ePMjtOxKT+9/+7d8GIhFEADjfuHHjHWVWcoEoIHJCP/Rx9erVwAQ/aJvUZxmJI378+te/DgitfDtM/oiFV199NZ/d83zLli3tyEXih51USu3/5S9/Gcpm2uiVFB296HhPAhKQgARKT4BNjiwTpAkTg7tGObhxMyFQiFIgDpjML168GObm5m7eGd3/EUD0w5IPfRG5GLR1xAZREiIniAqWc2gv306/UY58HQQW7ebbIvKR7CTC8vXXXys68tA8l4AEJCCBZhJIIf4YF3+Z8dFHH4U//elP2ZJEnggTf/46f57uIQbYiMpyy//5P/8nMPHGuNguyxns+2CTJ0sc+frLndM+Ezm2IWyWK190H3sQR6T33nsvsBREdCNfnr7y18ud0xZRIGyLcdFX6rB8xTGJHc5ZSokxZss7XK80GelYKUHrS0ACEqg2gcpZz6RJZIPlFRLLC4iHvCMsG5DyeemcfFK6Zk8Dv0z5j//xP2abLWkzn+jr6aefDmmvRarHJE3ddM2RKAFCgV+2MHnnbcvv16BsSsmfzra4po9UjvrYSX7K437+OuUvd4QXtiU/aSfVoZ8TJ05kIg7fUxn66WYr+fn6qZ1uR0VHNyrmSUACEpCABIYgkCZzJnTOh2ii1lUUHbUeXp2TgASGJmBFCdSIAAKISA1RjGm6peiYJn37loAEJCABCTSIgKKjQYOtqxIYAQGbkIAEJDA0AUXH0OisKAEJSEACEpDAIAQUHYPQsqwEigiYLwEJ1J4ADxzLP5V03A5Pur9x+0P7ig4omCQgAQlIQAISGDsBRcfYETe6A52XgAQkMBQBvuWvWrUqxLj48Kr8A7E4j3ExP8bFIw+x+vbbb+94uBdPKk2dc07K1+Wc+519xRhDusf9fPrmm2+yJ3XGuNgvbZJ27tyZFaNet/Ps5q0/3I9xsX6Mi0fybt0OnNMm1936w94UceH8ueeeCzwwjfLUI5Gf5wefVAYbY1zsN8YYKEd56ucT5alHeZ4/wjn2cIwxBo6UoU7e5vw59/JJ0ZGn4bkEJCABCZSCAD/t5EFYPJiK94LwNE4mP4zjYVTkp8TbV3ms969+9av2w714MBdP3Ex1eMfKqVOnwu9///vsRWzU4ZzJNN8XbZ48eTLwcC/u0V9KneXSk0bzTwTl0efpmvNUN3/kQVr0gU30h3881wOxkC9X1F++DO904d0utEF+vs/f/OY3ma/0wQvheLIqIqGT3xtvvBF4kRz3aKNb4ie3tMFL4DjSJg9B4y293coX5Sk6isiMM9+2JSABCUigbwI8BZNJkfeM9FuJSZI3qKZJmGsmyR07dmRP2uT9I+vXr88m5c42Z2ZmAu85QZh03stfp7ZTHpM2wiZdc0wChPOilPzrbK+zfLf7vJcFwYXIypdHjJCf8nh3C+fd3jGD+EG4XLhwgSKFiRfFxRgD5SmEAOlmE/eKkqKjiIz5EpCABCRQKgJEA5jYWQrILym8//774amnngpM3p0G5yfFe++9NzBxdpahvbRcwT3aQaBwXpSIgnz11Vft9nh5GgKAeukR6/m+i9rpNz/fHxEQHvSFkMLW9Ebb1BZCh3vYgvgg2kM5rinDdX5pBLsRYNxbLvE4eNruVY5ITmLQWa5MoqPTNq8lIAEJSEACPQkwgSJG0jf5osKIlX/7t3/L9mMUlRkkn5e5ETVBADzxxBPh0UcfDURGmHBph/6uXbvWFiXkrSTl++tsB5GBwEl9IjQoc+TIkcAyCMspnfe4P2gimvLQQw8NWu2O8oqOO3B4IQEJSEACVSLAZMseA77J97KbpQNEAN/qe5Xr514+6kB5vvkjOoi4cE1irwNLFizjcL2S1NlfZ1tJdLB0UtQnoqzo3p3tFV+xvEVfxSWWv6PoWJ6RJSQgAQlIoKQEEB3JNJZJ+CVGjIu/zCA6wLIHv6bgWzrLEGy4ZJkB8fFP//RPqerAR/ZLpIgClREZRFxiXOybvR2ffvrpkiUfBAT9b9u2LQwygXf2R58pYQeC4IUXXghsuEWApX5iXLQH27rZk9pY7piPlLDhNcYY8GG5ep33FR2dRLyWgAQkIIHSEWCJgNesM6HmjWMJIe0fYKkj/eKFZYWUKEN9jkQlmIC5R1nqkNIeidQ2ZamTrvNH2sAWjikfu8ijXRLn5KX76Uid1H+ym3v0RZ+cc8zfow7tceR+SggB9mb89re/DWfOnAnJH+5TNvXTaQ990V6yj+OJEycCHKjbLeXLYBttkrCV8hzJ57xXUnT0ouM9CUhAAhKQQEkJIAQQD3mxUVJT22YpOtooPJGABCQggcEITKc038j5Zs6kOx0LmtErfBE1REZG5bGiY1QkbUcCEpCABCQggZ4EFB098XhTAhKQwOAErCEBCXQnoOjozsVcCUhAAhKQgARGTEDRMWKgNicBCRQRMF8CEmg6AUVH0z8B+i8BCUhAAhKYEAFFx4RA240EigiYLwEJSKApBBQdTRlp/ZSABCQgAQlMmYCiY8oDYPdFBMyXgASaRoCna/JciKb53QR/L126FNavXx8UHU0YbX2UgAQkIAEJTJEAYhJRqeiY4iAM07V1JCABCdSVAC9m4x0edfWvyX6dPn3aSEeTPwD6LgEJSKBsBPgmzDdiUtls057hCbC0cvjw4bB///66LK8MD8OaEpCABCRQDgKs+TMxGe0ox3iMyordu3eH2dnZrDmXVzIM/pGABCQggTIQINrBN2MmqjLYow0DEOhSdGZmJstFTHKi6ICCSQISkIAESkGAaMehQ4cCwoMJi2MpDNOIgQgwbowflebm5jhkSdGRYfCPBCQgAQmUhUASHkQ9mLiIeszPz4f5m6ksNg5gR2OKIjQYowMHDgTGrdVqhbzgAISiAwomCUhAAhIoFQGEByF5Ji3OmchIMcYQoynG8jFAaDBGfJAuXryYbRzlPJ8UHXkanktAAhKQwGQI9NkLgiOJDwTIwsJCMJWTAUKDMWK8iob3/wMAAP//WMrDQAAAAAZJREFUAwD+86bgPw4+ZAAAAABJRU5ErkJggg==)
# ## **:תרשים טיפול - אירוע סיום ארוחת צהריים**
# ![EndLunchEvent.drawio.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAb4AAAJXCAYAAADsLjE+AAAtlnRFWHRteGZpbGUAJTNDbXhmaWxlJTIwaG9zdCUzRCUyMmFwcC5kaWFncmFtcy5uZXQlMjIlMjBhZ2VudCUzRCUyMk1vemlsbGElMkY1LjAlMjAoV2luZG93cyUyME5UJTIwMTAuMCUzQiUyMFdpbjY0JTNCJTIweDY0KSUyMEFwcGxlV2ViS2l0JTJGNTM3LjM2JTIwKEtIVE1MJTJDJTIwbGlrZSUyMEdlY2tvKSUyMENocm9tZSUyRjE0My4wLjAuMCUyMFNhZmFyaSUyRjUzNy4zNiUyMiUyMHZlcnNpb24lM0QlMjIyOS4yLjklMjIlMjBzY2FsZSUzRCUyMjElMjIlMjBib3JkZXIlM0QlMjIwJTIyJTNFJTBBJTIwJTIwJTNDZGlhZ3JhbSUyMG5hbWUlM0QlMjJQYWdlLTElMjIlMjBpZCUzRCUyMkhPcVhzT3NIZHNjN0ItRmdRdk53JTIyJTNFJTBBJTIwJTIwJTIwJTIwJTNDbXhHcmFwaE1vZGVsJTIwZHglM0QlMjI5ODMlMjIlMjBkeSUzRCUyMjUxOCUyMiUyMGdyaWQlM0QlMjIxJTIyJTIwZ3JpZFNpemUlM0QlMjIxMCUyMiUyMGd1aWRlcyUzRCUyMjElMjIlMjB0b29sdGlwcyUzRCUyMjElMjIlMjBjb25uZWN0JTNEJTIyMSUyMiUyMGFycm93cyUzRCUyMjElMjIlMjBmb2xkJTNEJTIyMSUyMiUyMHBhZ2UlM0QlMjIxJTIyJTIwcGFnZVNjYWxlJTNEJTIyMSUyMiUyMHBhZ2VXaWR0aCUzRCUyMjg1MCUyMiUyMHBhZ2VIZWlnaHQlM0QlMjIxMTAwJTIyJTIwbWF0aCUzRCUyMjAlMjIlMjBzaGFkb3clM0QlMjIwJTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTNDcm9vdCUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214Q2VsbCUyMGlkJTNEJTIyMCUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214Q2VsbCUyMGlkJTNEJTIyMSUyMiUyMHBhcmVudCUzRCUyMjAlMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteENlbGwlMjBpZCUzRCUyMndtdlVIUGp6M09QUGpFUVMwWHZqLTElMjIlMjBwYXJlbnQlM0QlMjIxJTIyJTIwc3R5bGUlM0QlMjJyb3VuZGVkJTNEMSUzQndoaXRlU3BhY2UlM0R3cmFwJTNCaHRtbCUzRDElM0IlMjIlMjB2YWx1ZSUzRCUyMiVENyU5NCVENyU5NSVENyVBMSVENyVBMyUyMCVENyU5MCVENyVBQSUyMCVENyU5NCVENyU5NCVENyU5QiVENyVBMCVENyVBMSVENyU5NCUyMCVENyU5RSVENyU5NCVENyU5MyVENyU5NSVENyU5QiVENyU5RiUyMCVENyU5QyVENyU5RSVENyVBOSVENyVBQSVENyVBMCVENyU5NCUyMCVENyU5NCVENyVBMSVENyU5NSVENyU5QiVENyU5RCUyMCVENyU5MCVENyVBQSUyMCVENyU5QiVENyU5QyVENyU5QyUyMCVENyU5NCVENyU5QiVENyVBMCVENyVBMSVENyU5NSVENyVBQSUyMCVENyU5NCVENyVBNCVENyU5MCVENyVBOCVENyVBNyUyMiUyMHZlcnRleCUzRCUyMjElMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteEdlb21ldHJ5JTIwaGVpZ2h0JTNEJTIyNjAlMjIlMjB3aWR0aCUzRCUyMjEyMCUyMiUyMHglM0QlMjIzNjUlMjIlMjB5JTNEJTIyMTg3JTIyJTIwYXMlM0QlMjJnZW9tZXRyeSUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14Q2VsbCUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214Q2VsbCUyMGlkJTNEJTIyc2d3UVNVSXBySGZQcTN2X0lOLU4tMSUyMiUyMHBhcmVudCUzRCUyMjElMjIlMjBzdHlsZSUzRCUyMnJvdW5kZWQlM0QxJTNCd2hpdGVTcGFjZSUzRHdyYXAlM0JodG1sJTNEMSUzQiUyMiUyMHZhbHVlJTNEJTIyJUQ3JTkzJUQ3JTkyJUQ3JTk1JUQ3JTlEJTIwdX5VKDAlMkMxKSUyNmx0JTNCYnIlMjZndCUzQiVENyVBOSVENyU5RSVENyVBMSVENyU5RSVENyU5QyUyMCVENyU5NCVENyU5MCVENyU5RCUyMCVENyU5NCVENyU5RSVENyU5MSVENyVBNyVENyVBOCUyMCVENyU5NCVENyU5OSVENyU5NCUyMCVENyU5RSVENyVBOCVENyU5NSVENyVBNiVENyU5NCUyMCVENyU5RSVENyU5NCVENyU5RSVENyVBMCVENyU5NCUyMCVENyVBOSVENyVBOCVENyU5QiVENyVBOSUzRiUyMiUyMHZlcnRleCUzRCUyMjElMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteEdlb21ldHJ5JTIwaGVpZ2h0JTNEJTIyNjAlMjIlMjB3aWR0aCUzRCUyMjEyMCUyMiUyMHglM0QlMjIzNjUlMjIlMjB5JTNEJTIyMjg3JTIyJTIwYXMlM0QlMjJnZW9tZXRyeSUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14Q2VsbCUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214Q2VsbCUyMGlkJTNEJTIyc2d3UVNVSXBySGZQcTN2X0lOLU4tMiUyMiUyMHBhcmVudCUzRCUyMjElMjIlMjBzdHlsZSUzRCUyMnJob21idXMlM0J3aGl0ZVNwYWNlJTNEd3JhcCUzQmh0bWwlM0QxJTNCJTIyJTIwdmFsdWUlM0QlMjIwJTI2YW1wJTNCbHQlM0J1JTI2YW1wJTNCbHQlM0IwLjklMjZsdCUzQmJyJTI2Z3QlM0IlRDclOTQlRDclOTAlRDclOUQlMjAlRDclOTQlRDclOUUlRDclOTElRDclQTclRDclQTglMjAlRDclOTQlRDclOTklRDclOTQlMjZsdCUzQmRpdiUyNmd0JTNCJUQ3JTlFJUQ3JUE4JUQ3JTk1JUQ3JUE2JUQ3JTk0JTIwJUQ3JTlFJUQ3JTk0JUQ3JTlFJUQ3JUEwJUQ3JTk0JTNGJTI2bHQlM0IlMkZkaXYlMjZndCUzQiUyMiUyMHZlcnRleCUzRCUyMjElMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteEdlb21ldHJ5JTIwaGVpZ2h0JTNEJTIyODAlMjIlMjB3aWR0aCUzRCUyMjE4MCUyMiUyMHglM0QlMjIzMzUlMjIlMjB5JTNEJTIyMzg3JTIyJTIwYXMlM0QlMjJnZW9tZXRyeSUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14Q2VsbCUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214Q2VsbCUyMGlkJTNEJTIyc2d3UVNVSXBySGZQcTN2X0lOLU4tNSUyMiUyMHBhcmVudCUzRCUyMjElMjIlMjBzdHlsZSUzRCUyMnJvdW5kZWQlM0QxJTNCd2hpdGVTcGFjZSUzRHdyYXAlM0JodG1sJTNEMSUzQiUyMiUyMHZhbHVlJTNEJTIyJUQ3JTk0JUQ3JTk1JUQ3JUE4JUQ3JTkzJTIwJUQ3JTkzJUQ3JTk5JUQ3JUE4JUQ3JTk1JUQ3JTkyJTIwJUQ3JUE0JUQ3JTkwJUQ3JUE4JUQ3JUE3JTIwJUQ3JUEyJUQ3JTkxJUQ3JTk1JUQ3JUE4JTI2YW1wJTNCbmJzcCUzQiUyMCVENyU5RSVENyU5MSVENyVBNyVENyVBOCUyMCVENyU5MS0wLjglMjIlMjB2ZXJ0ZXglM0QlMjIxJTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhHZW9tZXRyeSUyMGhlaWdodCUzRCUyMjYwJTIyJTIwd2lkdGglM0QlMjIxMjAlMjIlMjB4JTNEJTIyMjE1JTIyJTIweSUzRCUyMjQ3NyUyMiUyMGFzJTNEJTIyZ2VvbWV0cnklMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZteENlbGwlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteENlbGwlMjBpZCUzRCUyMnNnd1FTVUlwckhmUHEzdl9JTi1OLTYlMjIlMjBwYXJlbnQlM0QlMjIxJTIyJTIwc3R5bGUlM0QlMjJyb3VuZGVkJTNEMSUzQndoaXRlU3BhY2UlM0R3cmFwJTNCaHRtbCUzRDElM0IlMjIlMjB2YWx1ZSUzRCUyMiVENyU5NCVENyU5RSVENyVBOSVENyU5QSUyMCVENyU5QyVENyU5MCVENyU5OSVENyVBOCVENyU5NSVENyVBMiUyMCVENyU5NCVENyU5MSVENyU5MCUyMCVENyU5QyVENyVBNCVENyU5OSUyMCVENyU5OSVENyU5NSVENyU5RSVENyU5RiUyMCVENyU5NCVENyVBNCVENyVBMiVENyU5OSVENyU5QyVENyU5NSVENyVBQSUyMCVENyVBOSVENyU5QyUyMCVENyU5NCVENyU5RSVENyU5MSVENyVBNyVENyVBOCUyMiUyMHZlcnRleCUzRCUyMjElMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteEdlb21ldHJ5JTIwaGVpZ2h0JTNEJTIyNjAlMjIlMjB3aWR0aCUzRCUyMjEyMCUyMiUyMHglM0QlMjI1NDAlMjIlMjB5JTNEJTIyNjI3JTIyJTIwYXMlM0QlMjJnZW9tZXRyeSUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14Q2VsbCUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214Q2VsbCUyMGlkJTNEJTIyc2d3UVNVSXBySGZQcTN2X0lOLU4tNyUyMiUyMGVkZ2UlM0QlMjIxJTIyJTIwcGFyZW50JTNEJTIyMSUyMiUyMHNvdXJjZSUzRCUyMndtdlVIUGp6M09QUGpFUVMwWHZqLTElMjIlMjBzdHlsZSUzRCUyMmVuZEFycm93JTNEY2xhc3NpYyUzQmh0bWwlM0QxJTNCcm91bmRlZCUzRDAlM0JleGl0WCUzRDAuNSUzQmV4aXRZJTNEMSUzQmV4aXREeCUzRDAlM0JleGl0RHklM0QwJTNCZW50cnlYJTNEMC41JTNCZW50cnlZJTNEMCUzQmVudHJ5RHglM0QwJTNCZW50cnlEeSUzRDAlM0IlMjIlMjB0YXJnZXQlM0QlMjJzZ3dRU1VJcHJIZlBxM3ZfSU4tTi0xJTIyJTIwdmFsdWUlM0QlMjIlMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteEdlb21ldHJ5JTIwaGVpZ2h0JTNEJTIyNTAlMjIlMjByZWxhdGl2ZSUzRCUyMjElMjIlMjB3aWR0aCUzRCUyMjUwJTIyJTIwYXMlM0QlMjJnZW9tZXRyeSUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyNTcwJTIyJTIweSUzRCUyMjI3NyUyMiUyMGFzJTNEJTIyc291cmNlUG9pbnQlMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteFBvaW50JTIweCUzRCUyMjYyMCUyMiUyMHklM0QlMjIyMjclMjIlMjBhcyUzRCUyMnRhcmdldFBvaW50JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGbXhHZW9tZXRyeSUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14Q2VsbCUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214Q2VsbCUyMGlkJTNEJTIyc2d3UVNVSXBySGZQcTN2X0lOLU4tOCUyMiUyMGVkZ2UlM0QlMjIxJTIyJTIwcGFyZW50JTNEJTIyMSUyMiUyMHNvdXJjZSUzRCUyMnNnd1FTVUlwckhmUHEzdl9JTi1OLTElMjIlMjBzdHlsZSUzRCUyMmVuZEFycm93JTNEY2xhc3NpYyUzQmh0bWwlM0QxJTNCcm91bmRlZCUzRDAlM0JleGl0WCUzRDAuNSUzQmV4aXRZJTNEMSUzQmV4aXREeCUzRDAlM0JleGl0RHklM0QwJTNCZW50cnlYJTNEMC41JTNCZW50cnlZJTNEMCUzQmVudHJ5RHglM0QwJTNCZW50cnlEeSUzRDAlM0IlMjIlMjB0YXJnZXQlM0QlMjJzZ3dRU1VJcHJIZlBxM3ZfSU4tTi0yJTIyJTIwdmFsdWUlM0QlMjIlMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteEdlb21ldHJ5JTIwaGVpZ2h0JTNEJTIyNTAlMjIlMjByZWxhdGl2ZSUzRCUyMjElMjIlMjB3aWR0aCUzRCUyMjUwJTIyJTIwYXMlM0QlMjJnZW9tZXRyeSUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyNTMwJTIyJTIweSUzRCUyMjM4NyUyMiUyMGFzJTNEJTIyc291cmNlUG9pbnQlMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteFBvaW50JTIweCUzRCUyMjU4MCUyMiUyMHklM0QlMjIzMzclMjIlMjBhcyUzRCUyMnRhcmdldFBvaW50JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGbXhHZW9tZXRyeSUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14Q2VsbCUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214Q2VsbCUyMGlkJTNEJTIyc2d3UVNVSXBySGZQcTN2X0lOLU4tOSUyMiUyMGVkZ2UlM0QlMjIxJTIyJTIwcGFyZW50JTNEJTIyMSUyMiUyMHNvdXJjZSUzRCUyMnNnd1FTVUlwckhmUHEzdl9JTi1OLTIlMjIlMjBzdHlsZSUzRCUyMmVuZEFycm93JTNEY2xhc3NpYyUzQmh0bWwlM0QxJTNCcm91bmRlZCUzRDAlM0JleGl0WCUzRDAlM0JleGl0WSUzRDAuNSUzQmV4aXREeCUzRDAlM0JleGl0RHklM0QwJTNCZW50cnlYJTNEMC41JTNCZW50cnlZJTNEMCUzQmVudHJ5RHglM0QwJTNCZW50cnlEeSUzRDAlM0IlMjIlMjB0YXJnZXQlM0QlMjJzZ3dRU1VJcHJIZlBxM3ZfSU4tTi01JTIyJTIwdmFsdWUlM0QlMjIlMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteEdlb21ldHJ5JTIwaGVpZ2h0JTNEJTIyNTAlMjIlMjByZWxhdGl2ZSUzRCUyMjElMjIlMjB3aWR0aCUzRCUyMjUwJTIyJTIwYXMlM0QlMjJnZW9tZXRyeSUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ0FycmF5JTIwYXMlM0QlMjJwb2ludHMlMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteFBvaW50JTIweCUzRCUyMjI3NSUyMiUyMHklM0QlMjI0MjclMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZBcnJheSUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyMTkwJTIyJTIweSUzRCUyMjQ0NyUyMiUyMGFzJTNEJTIyc291cmNlUG9pbnQlMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteFBvaW50JTIweCUzRCUyMjI0MCUyMiUyMHklM0QlMjIzOTclMjIlMjBhcyUzRCUyMnRhcmdldFBvaW50JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGbXhHZW9tZXRyeSUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14Q2VsbCUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214Q2VsbCUyMGlkJTNEJTIyc2d3UVNVSXBySGZQcTN2X0lOLU4tMTIlMjIlMjBjb25uZWN0YWJsZSUzRCUyMjAlMjIlMjBwYXJlbnQlM0QlMjJzZ3dRU1VJcHJIZlBxM3ZfSU4tTi05JTIyJTIwc3R5bGUlM0QlMjJlZGdlTGFiZWwlM0JodG1sJTNEMSUzQmFsaWduJTNEY2VudGVyJTNCdmVydGljYWxBbGlnbiUzRG1pZGRsZSUzQnJlc2l6YWJsZSUzRDAlM0Jwb2ludHMlM0QlNUIlNUQlM0IlMjIlMjB2YWx1ZSUzRCUyMiVENyU5QyVENyU5MCUyMiUyMHZlcnRleCUzRCUyMjElMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteEdlb21ldHJ5JTIwcmVsYXRpdmUlM0QlMjIxJTIyJTIweCUzRCUyMi0wLjY4NTMlMjIlMjB5JTNEJTIyMiUyMiUyMGFzJTNEJTIyZ2VvbWV0cnklMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteFBvaW50JTIweCUzRCUyMi04JTIyJTIweSUzRCUyMi0xMiUyMiUyMGFzJTNEJTIyb2Zmc2V0JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGbXhHZW9tZXRyeSUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14Q2VsbCUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214Q2VsbCUyMGlkJTNEJTIyc2d3UVNVSXBySGZQcTN2X0lOLU4tMTAlMjIlMjBlZGdlJTNEJTIyMSUyMiUyMHBhcmVudCUzRCUyMjElMjIlMjBzb3VyY2UlM0QlMjJzZ3dRU1VJcHJIZlBxM3ZfSU4tTi01JTIyJTIwc3R5bGUlM0QlMjJlbmRBcnJvdyUzRGNsYXNzaWMlM0JodG1sJTNEMSUzQnJvdW5kZWQlM0QwJTNCZXhpdFglM0QwLjUlM0JleGl0WSUzRDElM0JleGl0RHglM0QwJTNCZXhpdER5JTNEMCUzQmVudHJ5WCUzRDAuNSUzQmVudHJ5WSUzRDAlM0JlbnRyeUR4JTNEMCUzQmVudHJ5RHklM0QwJTNCJTIyJTIwdGFyZ2V0JTNEJTIyc2d3UVNVSXBySGZQcTN2X0lOLU4tNiUyMiUyMHZhbHVlJTNEJTIyJTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhHZW9tZXRyeSUyMGhlaWdodCUzRCUyMjUwJTIyJTIwcmVsYXRpdmUlM0QlMjIxJTIyJTIwd2lkdGglM0QlMjI1MCUyMiUyMGFzJTNEJTIyZ2VvbWV0cnklMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NBcnJheSUyMGFzJTNEJTIycG9pbnRzJTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhQb2ludCUyMHglM0QlMjIyNzUlMjIlMjB5JTNEJTIyNTc3JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhQb2ludCUyMHglM0QlMjI2MDAlMjIlMjB5JTNEJTIyNTc3JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGQXJyYXklM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteFBvaW50JTIweCUzRCUyMjQyMCUyMiUyMHklM0QlMjI1MzclMjIlMjBhcyUzRCUyMnNvdXJjZVBvaW50JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhQb2ludCUyMHglM0QlMjI0NzAlMjIlMjB5JTNEJTIyNDg3JTIyJTIwYXMlM0QlMjJ0YXJnZXRQb2ludCUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14R2VvbWV0cnklM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZteENlbGwlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteENlbGwlMjBpZCUzRCUyMnNnd1FTVUlwckhmUHEzdl9JTi1OLTExJTIyJTIwZWRnZSUzRCUyMjElMjIlMjBwYXJlbnQlM0QlMjIxJTIyJTIwc291cmNlJTNEJTIyc2d3UVNVSXBySGZQcTN2X0lOLU4tMiUyMiUyMHN0eWxlJTNEJTIyZW5kQXJyb3clM0RjbGFzc2ljJTNCaHRtbCUzRDElM0Jyb3VuZGVkJTNEMCUzQmV4aXRYJTNEMSUzQmV4aXRZJTNEMC41JTNCZXhpdER4JTNEMCUzQmV4aXREeSUzRDAlM0JlbnRyeVglM0QwLjUlM0JlbnRyeVklM0QwJTNCZW50cnlEeCUzRDAlM0JlbnRyeUR5JTNEMCUzQiUyMiUyMHRhcmdldCUzRCUyMnNnd1FTVUlwckhmUHEzdl9JTi1OLTYlMjIlMjB2YWx1ZSUzRCUyMiUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214R2VvbWV0cnklMjBoZWlnaHQlM0QlMjI1MCUyMiUyMHJlbGF0aXZlJTNEJTIyMSUyMiUyMHdpZHRoJTNEJTIyNTAlMjIlMjBhcyUzRCUyMmdlb21ldHJ5JTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDQXJyYXklMjBhcyUzRCUyMnBvaW50cyUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyNjAwJTIyJTIweSUzRCUyMjQyNyUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRkFycmF5JTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhQb2ludCUyMHglM0QlMjI2NDAlMjIlMjB5JTNEJTIyNTE3JTIyJTIwYXMlM0QlMjJzb3VyY2VQb2ludCUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyNjkwJTIyJTIweSUzRCUyMjQ2NyUyMiUyMGFzJTNEJTIydGFyZ2V0UG9pbnQlMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZteEdlb21ldHJ5JTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGbXhDZWxsJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhDZWxsJTIwaWQlM0QlMjJzZ3dRU1VJcHJIZlBxM3ZfSU4tTi0xMyUyMiUyMGNvbm5lY3RhYmxlJTNEJTIyMCUyMiUyMHBhcmVudCUzRCUyMnNnd1FTVUlwckhmUHEzdl9JTi1OLTExJTIyJTIwc3R5bGUlM0QlMjJlZGdlTGFiZWwlM0JodG1sJTNEMSUzQmFsaWduJTNEY2VudGVyJTNCdmVydGljYWxBbGlnbiUzRG1pZGRsZSUzQnJlc2l6YWJsZSUzRDAlM0Jwb2ludHMlM0QlNUIlNUQlM0IlMjIlMjB2YWx1ZSUzRCUyMiVENyU5QiVENyU5RiUyMiUyMHZlcnRleCUzRCUyMjElMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteEdlb21ldHJ5JTIwcmVsYXRpdmUlM0QlMjIxJTIyJTIweCUzRCUyMi0wLjc1ODUlMjIlMjB5JTNEJTIyLTElMjIlMjBhcyUzRCUyMmdlb21ldHJ5JTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhQb2ludCUyMHglM0QlMjItOSUyMiUyMHklM0QlMjItMTElMjIlMjBhcyUzRCUyMm9mZnNldCUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14R2VvbWV0cnklM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZteENlbGwlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteENlbGwlMjBpZCUzRCUyMnNnd1FTVUlwckhmUHEzdl9JTi1OLTE0JTIyJTIwcGFyZW50JTNEJTIyMSUyMiUyMHN0eWxlJTNEJTIycm91bmRlZCUzRDElM0J3aGl0ZVNwYWNlJTNEd3JhcCUzQmh0bWwlM0QxJTNCJTIyJTIwdmFsdWUlM0QlMjIlRDclOTQlRDclOUIlRDclQTAlRDclQTElMjAlRDclOUMlRDclOTklRDclOTUlRDclOUUlRDclOUYlMjAlRDclOTAlRDclQUElMjAlRDclQTElRDclOTklRDclOTUlRDclOUQlMjAlRDclOTAlRDclQTglRDclOTUlRDclOTclRDclQUElMjAlRDclOTQlRDclQTYlRDclOTQlRDclQTglRDclOTklRDclOTklRDclOUQlMjAlRDclOTQlRDclOTElRDclOTAlRDclOTQlMjIlMjB2ZXJ0ZXglM0QlMjIxJTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhHZW9tZXRyeSUyMGhlaWdodCUzRCUyMjYwJTIyJTIwd2lkdGglM0QlMjIxMjAlMjIlMjB4JTNEJTIyMzY1JTIyJTIweSUzRCUyMjkwJTIyJTIwYXMlM0QlMjJnZW9tZXRyeSUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14Q2VsbCUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214Q2VsbCUyMGlkJTNEJTIyc2d3UVNVSXBySGZQcTN2X0lOLU4tMTUlMjIlMjBlZGdlJTNEJTIyMSUyMiUyMHBhcmVudCUzRCUyMjElMjIlMjBzb3VyY2UlM0QlMjJzZ3dRU1VJcHJIZlBxM3ZfSU4tTi0xNCUyMiUyMHN0eWxlJTNEJTIyZW5kQXJyb3clM0RjbGFzc2ljJTNCaHRtbCUzRDElM0Jyb3VuZGVkJTNEMCUzQmV4aXRYJTNEMC41JTNCZXhpdFklM0QxJTNCZXhpdER4JTNEMCUzQmV4aXREeSUzRDAlM0JlbnRyeVglM0QwLjUlM0JlbnRyeVklM0QwJTNCZW50cnlEeCUzRDAlM0JlbnRyeUR5JTNEMCUzQiUyMiUyMHRhcmdldCUzRCUyMndtdlVIUGp6M09QUGpFUVMwWHZqLTElMjIlMjB2YWx1ZSUzRCUyMiUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214R2VvbWV0cnklMjBoZWlnaHQlM0QlMjI1MCUyMiUyMHJlbGF0aXZlJTNEJTIyMSUyMiUyMHdpZHRoJTNEJTIyNTAlMjIlMjBhcyUzRCUyMmdlb21ldHJ5JTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhQb2ludCUyMHglM0QlMjI1MzAlMjIlMjB5JTNEJTIyMjEwJTIyJTIwYXMlM0QlMjJzb3VyY2VQb2ludCUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyNTgwJTIyJTIweSUzRCUyMjE2MCUyMiUyMGFzJTNEJTIydGFyZ2V0UG9pbnQlMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZteEdlb21ldHJ5JTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGbXhDZWxsJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGcm9vdCUzRSUwQSUyMCUyMCUyMCUyMCUzQyUyRm14R3JhcGhNb2RlbCUzRSUwQSUyMCUyMCUzQyUyRmRpYWdyYW0lM0UlMEElM0MlMkZt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)
# ## **:תרשים טיפול - אירוע סיום אטרקציה**
# ![EndAttractionEvent.drawio.png](data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAaUAAAJGCAYAAAD76pofAAA0Q3RFWHRteGZpbGUAJTNDbXhmaWxlJTIwaG9zdCUzRCUyMmFwcC5kaWFncmFtcy5uZXQlMjIlMjBhZ2VudCUzRCUyMk1vemlsbGElMkY1LjAlMjAoV2luZG93cyUyME5UJTIwMTAuMCUzQiUyMFdpbjY0JTNCJTIweDY0KSUyMEFwcGxlV2ViS2l0JTJGNTM3LjM2JTIwKEtIVE1MJTJDJTIwbGlrZSUyMEdlY2tvKSUyMENocm9tZSUyRjE0My4wLjAuMCUyMFNhZmFyaSUyRjUzNy4zNiUyMiUyMHZlcnNpb24lM0QlMjIyOS4yLjklMjIlMjBzY2FsZSUzRCUyMjElMjIlMjBib3JkZXIlM0QlMjIwJTIyJTNFJTBBJTIwJTIwJTNDZGlhZ3JhbSUyMG5hbWUlM0QlMjJQYWdlLTElMjIlMjBpZCUzRCUyMjZiay01QVhkOHNVbkVockYxclZaJTIyJTNFJTBBJTIwJTIwJTIwJTIwJTNDbXhHcmFwaE1vZGVsJTIwZHglM0QlMjI5ODMlMjIlMjBkeSUzRCUyMjUxOCUyMiUyMGdyaWQlM0QlMjIxJTIyJTIwZ3JpZFNpemUlM0QlMjIxMCUyMiUyMGd1aWRlcyUzRCUyMjElMjIlMjB0b29sdGlwcyUzRCUyMjElMjIlMjBjb25uZWN0JTNEJTIyMSUyMiUyMGFycm93cyUzRCUyMjElMjIlMjBmb2xkJTNEJTIyMSUyMiUyMHBhZ2UlM0QlMjIxJTIyJTIwcGFnZVNjYWxlJTNEJTIyMSUyMiUyMHBhZ2VXaWR0aCUzRCUyMjg1MCUyMiUyMHBhZ2VIZWlnaHQlM0QlMjIxMTAwJTIyJTIwbWF0aCUzRCUyMjAlMjIlMjBzaGFkb3clM0QlMjIwJTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTNDcm9vdCUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214Q2VsbCUyMGlkJTNEJTIyMCUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214Q2VsbCUyMGlkJTNEJTIyMSUyMiUyMHBhcmVudCUzRCUyMjAlMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteENlbGwlMjBpZCUzRCUyMmVNZ3BQdThPcW13eTd1aktuN0gzLTElMjIlMjBwYXJlbnQlM0QlMjIxJTIyJTIwc3R5bGUlM0QlMjJyb3VuZGVkJTNEMSUzQndoaXRlU3BhY2UlM0R3cmFwJTNCaHRtbCUzRDElM0IlMjIlMjB2YWx1ZSUzRCUyMiVENyVBMSVENyU5RSVENyU5RiUyMCVENyU5MCVENyVBQSUyMCVENyU5NCVENyU5MCVENyU5OCVENyVBOCVENyVBNyVENyVBNiVENyU5OSVENyU5NCUyMCVENyU5QiVENyU5MSVENyU5NSVENyVBNiVENyVBMiVENyU5NCUyMCVENyVBMiVENyU5MSVENyU5NSVENyVBOCUyMCVENyU5RSVENyU5MSVENyVBNyVENyVBOCUyMCVENyU5NiVENyU5NCUyMiUyMHZlcnRleCUzRCUyMjElMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteEdlb21ldHJ5JTIwaGVpZ2h0JTNEJTIyNjAlMjIlMjB3aWR0aCUzRCUyMjEyMCUyMiUyMHglM0QlMjIzNjUlMjIlMjB5JTNEJTIyMTg5JTIyJTIwYXMlM0QlMjJnZW9tZXRyeSUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14Q2VsbCUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214Q2VsbCUyMGlkJTNEJTIySzJmUW9OZHFwbWVveXhsNUVobGYtMiUyMiUyMHBhcmVudCUzRCUyMjElMjIlMjBzdHlsZSUzRCUyMnJvdW5kZWQlM0QxJTNCd2hpdGVTcGFjZSUzRHdyYXAlM0JodG1sJTNEMSUzQiUyMiUyMHZhbHVlJTNEJTIyJUQ3JTkzJUQ3JTkyJUQ3JTk1JUQ3JTlEJTIwdX5VKDAlMkMxKSUyNmx0JTNCYnIlMjZndCUzQiVENyVBOSVENyU5RSVENyVBMSVENyU5RSVENyU5QyUyMCVENyU5NCVENyU5MCVENyU5RCUyMCVENyU5QyVENyU5RSVENyU5MSVENyVBNyVENyVBOCUyMCVENyU5NCVENyU5OSVENyU5OSVENyVBQSVENyU5NCUyMCVENyU5NyVENyU5NSVENyU5NSVENyU5OSVENyU5NCUyMCVENyU5OCVENyU5NSVENyU5MSVENyU5NCUyMCVENyU5MSVENyU5RSVENyVBQSVENyVBNyVENyU5RiUyMiUyMHZlcnRleCUzRCUyMjElMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteEdlb21ldHJ5JTIwaGVpZ2h0JTNEJTIyNjAlMjIlMjB3aWR0aCUzRCUyMjEyMCUyMiUyMHglM0QlMjIzNjUlMjIlMjB5JTNEJTIyMjg5JTIyJTIwYXMlM0QlMjJnZW9tZXRyeSUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14Q2VsbCUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214Q2VsbCUyMGlkJTNEJTIySzJmUW9OZHFwbWVveXhsNUVobGYtMyUyMiUyMHBhcmVudCUzRCUyMjElMjIlMjBzdHlsZSUzRCUyMnJob21idXMlM0J3aGl0ZVNwYWNlJTNEd3JhcCUzQmh0bWwlM0QxJTNCJTIyJTIwdmFsdWUlM0QlMjIwJTI2YW1wJTNCbHQlM0J1JTI2YW1wJTNCbHQlM0IwLjUlMjZsdCUzQmJyJTI2Z3QlM0IlRDclOTQlRDclOTAlRDclOUQlMjAlRDclOUMlRDclOUUlRDclOTElRDclQTclRDclQTglMjAlRDclOTQlRDclOTklRDclOTklRDclQUElRDclOTQlMjAlRDclOTclRDclOTUlRDclOTUlRDclOTklRDclOTQlMjZsdCUzQmRpdiUyNmd0JTNCJUQ3JTk4JUQ3JTk1JUQ3JTkxJUQ3JTk0JTIwJUQ3JTkxJUQ3JTlFJUQ3JUFBJUQ3JUE3JUQ3JTlGJTNGJTI2bHQlM0IlMkZkaXYlMjZndCUzQiUyMiUyMHZlcnRleCUzRCUyMjElMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteEdlb21ldHJ5JTIwaGVpZ2h0JTNEJTIyODAlMjIlMjB3aWR0aCUzRCUyMjE4MCUyMiUyMHglM0QlMjIzMzUlMjIlMjB5JTNEJTIyMzk5JTIyJTIwYXMlM0QlMjJnZW9tZXRyeSUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14Q2VsbCUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214Q2VsbCUyMGlkJTNEJTIySzJmUW9OZHFwbWVveXhsNUVobGYtNCUyMiUyMGVkZ2UlM0QlMjIxJTIyJTIwcGFyZW50JTNEJTIyMSUyMiUyMHNvdXJjZSUzRCUyMmVNZ3BQdThPcW13eTd1aktuN0gzLTElMjIlMjBzdHlsZSUzRCUyMmVuZEFycm93JTNEY2xhc3NpYyUzQmh0bWwlM0QxJTNCcm91bmRlZCUzRDAlM0JleGl0WCUzRDAuNSUzQmV4aXRZJTNEMSUzQmV4aXREeCUzRDAlM0JleGl0RHklM0QwJTNCZW50cnlYJTNEMC41JTNCZW50cnlZJTNEMCUzQmVudHJ5RHglM0QwJTNCZW50cnlEeSUzRDAlM0IlMjIlMjB0YXJnZXQlM0QlMjJLMmZRb05kcXBtZW95eGw1RWhsZi0yJTIyJTIwdmFsdWUlM0QlMjIlMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteEdlb21ldHJ5JTIwaGVpZ2h0JTNEJTIyNTAlMjIlMjByZWxhdGl2ZSUzRCUyMjElMjIlMjB3aWR0aCUzRCUyMjUwJTIyJTIwYXMlM0QlMjJnZW9tZXRyeSUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyNTYwJTIyJTIweSUzRCUyMjMxOSUyMiUyMGFzJTNEJTIyc291cmNlUG9pbnQlMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteFBvaW50JTIweCUzRCUyMjYxMCUyMiUyMHklM0QlMjIyNjklMjIlMjBhcyUzRCUyMnRhcmdldFBvaW50JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGbXhHZW9tZXRyeSUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14Q2VsbCUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214Q2VsbCUyMGlkJTNEJTIySzJmUW9OZHFwbWVveXhsNUVobGYtNSUyMiUyMGVkZ2UlM0QlMjIxJTIyJTIwcGFyZW50JTNEJTIyMSUyMiUyMHNvdXJjZSUzRCUyMksyZlFvTmRxcG1lb3l4bDVFaGxmLTIlMjIlMjBzdHlsZSUzRCUyMmVuZEFycm93JTNEY2xhc3NpYyUzQmh0bWwlM0QxJTNCcm91bmRlZCUzRDAlM0JleGl0WCUzRDAuNSUzQmV4aXRZJTNEMSUzQmV4aXREeCUzRDAlM0JleGl0RHklM0QwJTNCZW50cnlYJTNEMC41JTNCZW50cnlZJTNEMCUzQmVudHJ5RHglM0QwJTNCZW50cnlEeSUzRDAlM0IlMjIlMjB0YXJnZXQlM0QlMjJLMmZRb05kcXBtZW95eGw1RWhsZi0zJTIyJTIwdmFsdWUlM0QlMjIlMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteEdlb21ldHJ5JTIwaGVpZ2h0JTNEJTIyNTAlMjIlMjByZWxhdGl2ZSUzRCUyMjElMjIlMjB3aWR0aCUzRCUyMjUwJTIyJTIwYXMlM0QlMjJnZW9tZXRyeSUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyNTQwJTIyJTIweSUzRCUyMjQxOSUyMiUyMGFzJTNEJTIyc291cmNlUG9pbnQlMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteFBvaW50JTIweCUzRCUyMjU5MCUyMiUyMHklM0QlMjIzNjklMjIlMjBhcyUzRCUyMnRhcmdldFBvaW50JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGbXhHZW9tZXRyeSUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14Q2VsbCUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214Q2VsbCUyMGlkJTNEJTIySzJmUW9OZHFwbWVveXhsNUVobGYtNiUyMiUyMHBhcmVudCUzRCUyMjElMjIlMjBzdHlsZSUzRCUyMnJvdW5kZWQlM0QxJTNCd2hpdGVTcGFjZSUzRHdyYXAlM0JodG1sJTNEMSUzQiUyMiUyMHZhbHVlJTNEJTIyJUQ3JTk0JUQ3JUEyJUQ3JTlDJUQ3JTk0JTIwJUQ3JTkzJUQ3JTk5JUQ3JUE4JUQ3JTk1JUQ3JTkyJTIwJUQ3JUE0JUQ3JTkwJUQ3JUE4JUQ3JUE3JTIwJUQ3JUEyJUQ3JTkxJUQ3JTk1JUQ3JUE4JTIwJUQ3JTlFJUQ3JTkxJUQ3JUE3JUQ3JUE4JTIwJUQ3JUEyJUQ3JTlDJTIwJUQ3JUE0JUQ3JTk5JTIwJUQ3JTk0JUQ3JUE0JUQ3JTk1JUQ3JUE3JUQ3JUEwJUQ3JUE2JUQ3JTk5JUQ3JTk0JTIyJTIwdmVydGV4JTNEJTIyMSUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214R2VvbWV0cnklMjBoZWlnaHQlM0QlMjI2MCUyMiUyMHdpZHRoJTNEJTIyMTIwJTIyJTIweCUzRCUyMjIxNSUyMiUyMHklM0QlMjI0ODklMjIlMjBhcyUzRCUyMmdlb21ldHJ5JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGbXhDZWxsJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhDZWxsJTIwaWQlM0QlMjJLMmZRb05kcXBtZW95eGw1RWhsZi03JTIyJTIwcGFyZW50JTNEJTIyMSUyMiUyMHN0eWxlJTNEJTIycm91bmRlZCUzRDElM0J3aGl0ZVNwYWNlJTNEd3JhcCUzQmh0bWwlM0QxJTNCJTIyJTIwdmFsdWUlM0QlMjIlRDclOTQlRDclOTUlRDclQTglRDclOTMlMjAlRDclOTMlRDclOTklRDclQTglRDclOTUlRDclOTIlMjAlRDclQTQlRDclOTAlRDclQTglRDclQTclMjAlRDclQTIlRDclOTElRDclOTUlRDclQTglMjZhbXAlM0JuYnNwJTNCJTIwJUQ3JTlFJUQ3JTkxJUQ3JUE3JUQ3JUE4JTIwJUQ3JTkxLTAuMSUyMiUyMHZlcnRleCUzRCUyMjElMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteEdlb21ldHJ5JTIwaGVpZ2h0JTNEJTIyNjAlMjIlMjB3aWR0aCUzRCUyMjEyMCUyMiUyMHglM0QlMjI1MTUlMjIlMjB5JTNEJTIyNDg5JTIyJTIwYXMlM0QlMjJnZW9tZXRyeSUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14Q2VsbCUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214Q2VsbCUyMGlkJTNEJTIySzJmUW9OZHFwbWVveXhsNUVobGYtOCUyMiUyMGVkZ2UlM0QlMjIxJTIyJTIwcGFyZW50JTNEJTIyMSUyMiUyMHNvdXJjZSUzRCUyMksyZlFvTmRxcG1lb3l4bDVFaGxmLTMlMjIlMjBzdHlsZSUzRCUyMmVuZEFycm93JTNEY2xhc3NpYyUzQmh0bWwlM0QxJTNCcm91bmRlZCUzRDAlM0JleGl0WCUzRDAlM0JleGl0WSUzRDAuNSUzQmV4aXREeCUzRDAlM0JleGl0RHklM0QwJTNCZW50cnlYJTNEMC41JTNCZW50cnlZJTNEMCUzQmVudHJ5RHglM0QwJTNCZW50cnlEeSUzRDAlM0IlMjIlMjB0YXJnZXQlM0QlMjJLMmZRb05kcXBtZW95eGw1RWhsZi02JTIyJTIwdmFsdWUlM0QlMjIlMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteEdlb21ldHJ5JTIwaGVpZ2h0JTNEJTIyNTAlMjIlMjByZWxhdGl2ZSUzRCUyMjElMjIlMjB3aWR0aCUzRCUyMjUwJTIyJTIwYXMlM0QlMjJnZW9tZXRyeSUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ0FycmF5JTIwYXMlM0QlMjJwb2ludHMlMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteFBvaW50JTIweCUzRCUyMjI3NSUyMiUyMHklM0QlMjI0MzklMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZBcnJheSUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyMjUwJTIyJTIweSUzRCUyMjQ3OSUyMiUyMGFzJTNEJTIyc291cmNlUG9pbnQlMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteFBvaW50JTIweCUzRCUyMjMwMCUyMiUyMHklM0QlMjI0MjklMjIlMjBhcyUzRCUyMnRhcmdldFBvaW50JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGbXhHZW9tZXRyeSUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14Q2VsbCUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214Q2VsbCUyMGlkJTNEJTIySzJmUW9OZHFwbWVveXhsNUVobGYtMTAlMjIlMjBjb25uZWN0YWJsZSUzRCUyMjAlMjIlMjBwYXJlbnQlM0QlMjJLMmZRb05kcXBtZW95eGw1RWhsZi04JTIyJTIwc3R5bGUlM0QlMjJlZGdlTGFiZWwlM0JodG1sJTNEMSUzQmFsaWduJTNEY2VudGVyJTNCdmVydGljYWxBbGlnbiUzRG1pZGRsZSUzQnJlc2l6YWJsZSUzRDAlM0Jwb2ludHMlM0QlNUIlNUQlM0IlMjIlMjB2YWx1ZSUzRCUyMiVENyU5QiVENyU5RiUyMiUyMHZlcnRleCUzRCUyMjElMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteEdlb21ldHJ5JTIwcmVsYXRpdmUlM0QlMjIxJTIyJTIweCUzRCUyMi0wLjQ5NzIlMjIlMjB5JTNEJTIyLTMlMjIlMjBhcyUzRCUyMmdlb21ldHJ5JTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhQb2ludCUyMHklM0QlMjItNyUyMiUyMGFzJTNEJTIyb2Zmc2V0JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGbXhHZW9tZXRyeSUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14Q2VsbCUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214Q2VsbCUyMGlkJTNEJTIySzJmUW9OZHFwbWVveXhsNUVobGYtOSUyMiUyMGVkZ2UlM0QlMjIxJTIyJTIwcGFyZW50JTNEJTIyMSUyMiUyMHNvdXJjZSUzRCUyMksyZlFvTmRxcG1lb3l4bDVFaGxmLTMlMjIlMjBzdHlsZSUzRCUyMmVuZEFycm93JTNEY2xhc3NpYyUzQmh0bWwlM0QxJTNCcm91bmRlZCUzRDAlM0JleGl0WCUzRDElM0JleGl0WSUzRDAuNSUzQmV4aXREeCUzRDAlM0JleGl0RHklM0QwJTNCZW50cnlYJTNEMC41JTNCZW50cnlZJTNEMCUzQmVudHJ5RHglM0QwJTNCZW50cnlEeSUzRDAlM0IlMjIlMjB0YXJnZXQlM0QlMjJLMmZRb05kcXBtZW95eGw1RWhsZi03JTIyJTIwdmFsdWUlM0QlMjIlMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteEdlb21ldHJ5JTIwaGVpZ2h0JTNEJTIyNTAlMjIlMjByZWxhdGl2ZSUzRCUyMjElMjIlMjB3aWR0aCUzRCUyMjUwJTIyJTIwYXMlM0QlMjJnZW9tZXRyeSUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ0FycmF5JTIwYXMlM0QlMjJwb2ludHMlMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteFBvaW50JTIweCUzRCUyMjU3NSUyMiUyMHklM0QlMjI0MzklMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZBcnJheSUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyNjUwJTIyJTIweSUzRCUyMjQ0OSUyMiUyMGFzJTNEJTIyc291cmNlUG9pbnQlMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteFBvaW50JTIweCUzRCUyMjcwMCUyMiUyMHklM0QlMjIzOTklMjIlMjBhcyUzRCUyMnRhcmdldFBvaW50JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGbXhHZW9tZXRyeSUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14Q2VsbCUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214Q2VsbCUyMGlkJTNEJTIySzJmUW9OZHFwbWVveXhsNUVobGYtMTElMjIlMjBjb25uZWN0YWJsZSUzRCUyMjAlMjIlMjBwYXJlbnQlM0QlMjJLMmZRb05kcXBtZW95eGw1RWhsZi05JTIyJTIwc3R5bGUlM0QlMjJlZGdlTGFiZWwlM0JodG1sJTNEMSUzQmFsaWduJTNEY2VudGVyJTNCdmVydGljYWxBbGlnbiUzRG1pZGRsZSUzQnJlc2l6YWJsZSUzRDAlM0Jwb2ludHMlM0QlNUIlNUQlM0IlMjIlMjB2YWx1ZSUzRCUyMiVENyU5QyVENyU5MCUyMiUyMHZlcnRleCUzRCUyMjElMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteEdlb21ldHJ5JTIwcmVsYXRpdmUlM0QlMjIxJTIyJTIweCUzRCUyMi0wLjU0OTglMjIlMjB5JTNEJTIyMSUyMiUyMGFzJTNEJTIyZ2VvbWV0cnklMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteFBvaW50JTIweSUzRCUyMi05JTIyJTIwYXMlM0QlMjJvZmZzZXQlMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZteEdlb21ldHJ5JTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGbXhDZWxsJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhDZWxsJTIwaWQlM0QlMjJLMmZRb05kcXBtZW95eGw1RWhsZi0xMiUyMiUyMHBhcmVudCUzRCUyMjElMjIlMjBzdHlsZSUzRCUyMnJvdW5kZWQlM0QxJTNCd2hpdGVTcGFjZSUzRHdyYXAlM0JodG1sJTNEMSUzQiUyMiUyMHZhbHVlJTNEJTIyJUQ3JTk0JUQ3JTlCJUQ3JUEwJUQ3JUExJTIwJUQ3JTlDJUQ3JTk5JUQ3JTk1JUQ3JTlFJUQ3JTlGJTIwJUQ3JTkwJUQ3JUFBJTIwJUQ3JUExJUQ3JTk5JUQ3JTk1JUQ3JTlEJTIwJUQ3JTk0JUQ3JTkwJUQ3JTk4JUQ3JUE4JUQ3JUE3JUQ3JUE2JUQ3JTk5JUQ3JTk0JTIwJUQ3JTk0JUQ3JUEwJUQ3JTk1JUQ3JTlCJUQ3JTk3JUQ3JTk5JUQ3JUFBJTIwJUQ3JTk0JUQ3JTkxJUQ3JTkwJUQ3JTk0JTIyJTIwdmVydGV4JTNEJTIyMSUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214R2VvbWV0cnklMjBoZWlnaHQlM0QlMjI2MCUyMiUyMHdpZHRoJTNEJTIyMTIwJTIyJTIweCUzRCUyMjM2NSUyMiUyMHklM0QlMjIxMDAlMjIlMjBhcyUzRCUyMmdlb21ldHJ5JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGbXhDZWxsJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhDZWxsJTIwaWQlM0QlMjJLMmZRb05kcXBtZW95eGw1RWhsZi0xMyUyMiUyMGVkZ2UlM0QlMjIxJTIyJTIwcGFyZW50JTNEJTIyMSUyMiUyMHNvdXJjZSUzRCUyMksyZlFvTmRxcG1lb3l4bDVFaGxmLTEyJTIyJTIwc3R5bGUlM0QlMjJlbmRBcnJvdyUzRGNsYXNzaWMlM0JodG1sJTNEMSUzQnJvdW5kZWQlM0QwJTNCZXhpdFglM0QwLjUlM0JleGl0WSUzRDElM0JleGl0RHglM0QwJTNCZXhpdER5JTNEMCUzQmVudHJ5WCUzRDAuNSUzQmVudHJ5WSUzRDAlM0JlbnRyeUR4JTNEMCUzQmVudHJ5RHklM0QwJTNCJTIyJTIwdGFyZ2V0JTNEJTIyZU1ncFB1OE9xbXd5N3VqS243SDMtMSUyMiUyMHZhbHVlJTNEJTIyJTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhHZW9tZXRyeSUyMGhlaWdodCUzRCUyMjUwJTIyJTIwcmVsYXRpdmUlM0QlMjIxJTIyJTIwd2lkdGglM0QlMjI1MCUyMiUyMGFzJTNEJTIyZ2VvbWV0cnklMjIlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteFBvaW50JTIweCUzRCUyMjU2MCUyMiUyMHklM0QlMjIyMDAlMjIlMjBhcyUzRCUyMnNvdXJjZVBvaW50JTIyJTIwJTJGJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhQb2ludCUyMHglM0QlMjI2MTAlMjIlMjB5JTNEJTIyMTUwJTIyJTIwYXMlM0QlMjJ0YXJnZXRQb2ludCUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRm14R2VvbWV0cnklM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZteENlbGwlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteENlbGwlMjBpZCUzRCUyMksyZlFvTmRxcG1lb3l4bDVFaGxmLTE0JTIyJTIwcGFyZW50JTNEJTIyMSUyMiUyMHN0eWxlJTNEJTIycm91bmRlZCUzRDElM0J3aGl0ZVNwYWNlJTNEd3JhcCUzQmh0bWwlM0QxJTNCJTIyJTIwdmFsdWUlM0QlMjIlRDclOTQlRDclOUUlRDclQTklRDclOUElMjAlRDclOUMlRDclOTAlRDclOTklRDclQTglRDclOTUlRDclQTIlMjAlRDclOTQlRDclOTElRDclOTAlMjAlRDclOUMlRDclQTQlRDclOTklMjAlRDclOTklRDclOTUlRDclOUUlRDclOUYlMjAlRDclOTQlRDclQTQlRDclQTIlRDclOTklRDclOUMlRDclOTUlRDclQUElMjAlRDclQTklRDclOUMlMjAlRDclOTQlRDclOUUlRDclOTElRDclQTclRDclQTglMjIlMjB2ZXJ0ZXglM0QlMjIxJTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhHZW9tZXRyeSUyMGhlaWdodCUzRCUyMjYwJTIyJTIwd2lkdGglM0QlMjIxMjAlMjIlMjB4JTNEJTIyMzY1JTIyJTIweSUzRCUyMjYyMCUyMiUyMGFzJTNEJTIyZ2VvbWV0cnklMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZteENlbGwlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0NteENlbGwlMjBpZCUzRCUyMksyZlFvTmRxcG1lb3l4bDVFaGxmLTE1JTIyJTIwZWRnZSUzRCUyMjElMjIlMjBwYXJlbnQlM0QlMjIxJTIyJTIwc291cmNlJTNEJTIySzJmUW9OZHFwbWVveXhsNUVobGYtNiUyMiUyMHN0eWxlJTNEJTIyZW5kQXJyb3clM0RjbGFzc2ljJTNCaHRtbCUzRDElM0Jyb3VuZGVkJTNEMCUzQmV4aXRYJTNEMC40NjUlM0JleGl0WSUzRDAuOTk4JTNCZXhpdER4JTNEMCUzQmV4aXREeSUzRDAlM0JleGl0UGVyaW1ldGVyJTNEMCUzQmVudHJ5WCUzRDAuNSUzQmVudHJ5WSUzRDAlM0JlbnRyeUR4JTNEMCUzQmVudHJ5RHklM0QwJTNCJTIyJTIwdGFyZ2V0JTNEJTIySzJmUW9OZHFwbWVveXhsNUVobGYtMTQlMjIlMjB2YWx1ZSUzRCUyMiUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214R2VvbWV0cnklMjBoZWlnaHQlM0QlMjI1MCUyMiUyMHJlbGF0aXZlJTNEJTIyMSUyMiUyMHdpZHRoJTNEJTIyNTAlMjIlMjBhcyUzRCUyMmdlb21ldHJ5JTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDQXJyYXklMjBhcyUzRCUyMnBvaW50cyUyMiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyMjcxJTIyJTIweSUzRCUyMjU4MCUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyNDI1JTIyJTIweSUzRCUyMjU4MCUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQyUyRkFycmF5JTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhQb2ludCUyMHglM0QlMjIyNjAlMjIlMjB5JTNEJTIyNjIwJTIyJTIwYXMlM0QlMjJzb3VyY2VQb2ludCUyMiUyMCUyRiUzRSUwQSUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUyMCUzQ214UG9pbnQlMjB4JTNEJTIyMzEwJTIyJTIweSUzRCUyMjU3MCUyMiUyMGFzJTNEJTIydGFyZ2V0UG9pbnQlMjIlMjAlMkYlM0UlMEElMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlMjAlM0MlMkZteEdlb21ldHJ5JTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDJTJGbXhDZWxsJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhDZWxsJTIwaWQlM0QlMjJLMmZRb05kcXBtZW95eGw1RWhsZi0xNiUyMiUyMGVkZ2UlM0QlMjIxJTIyJTIwcGFyZW50JTNEJTIyMSUyMiUyMHNvdXJjZSUzRCUyMksyZlFvTmRxcG1lb3l4bDVFaGxmLTclMjIlMjBzdHlsZSUzRCUyMmVuZEFycm93JTNEY2xhc3NpYyUzQmh0bWwlM0QxJTNCcm91bmRlZCUzRDAlM0JleGl0WCUzRDAuNSUzQmV4aXRZJTNEMSUzQmV4aXREeCUzRDAlM0JleGl0RHklM0QwJTNCZW50cnlYJTNEMC41JTNCZW50cnlZJTNEMCUzQmVudHJ5RHglM0QwJTNCZW50cnlEeSUzRDAlM0IlMjIlMjB0YXJnZXQlM0QlMjJLMmZRb05kcXBtZW95eGw1RWhsZi0xNCUyMiUyMHZhbHVlJTNEJTIyJTIyJTNFJTBBJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTIwJTNDbXhHZW9tZXRyeSUyMGhlaWdodCUzRCUyMjUwJTIy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KngHyT0REAERqBMBiVKdelttrR4BtUgEKkZAolSxDlVzREAERKDMBCRKZe49+S4CIiACFSNwg3MVa5GaIwIiIAIiUFoCmimVtuvkuAiIgAhUj4BEqXp9qhY55wRBBESgnAQkSuXsN3ktAiIgApUkIFGqZLeqUSIgAtUjUI8WSZTq0c9qpQiIgAiUgoBEqRTdJCdFQAREoB4EJEr16OfQSh1FQAREoNAEJEqF7h45JwIiIAL1IiBRqld/q7UiUD0CalGlCEiUKtWdaowIiIAIlJuARKnc/SfvRUAERKBSBCRKvjv1jwiIgAiIQBEISJSK0AvyQQREQAREwBOotCiNjY25c+fO+YbqHxGoGwG1VwTKSKDSolTGDpHPIiACIlBnAhKlOve+2i4CIiACBSPQXpQK5qzcEQEREAERqDYBiVK1+1etEwEREIFSEZAolaq75OwQCMiECIhAgQlIlArcOXJNBERABOpGQKJUtx5Xe0VABKpHoEItkihVqDPVFBEQAREoO4FaiJKZOTMFMzEwEwMzMTArB4OyC0w//ldelObn551CNwyUZ5DPCX98g5RXWX3+Wn0G+GzVKVRelOrUmWqrCIiACJSdgESp7D0o/0VABFoSUEL5CEiUytdn8lgEREAEKktAolTZrlXDREAERKB8BCRKnfpM6SIgAiIgArkRkCjlhloViYAIiIAIdCIgUepESOkiUD0CapEIFJaARKmwXSPHREAERKB+BCRK9etztVgEREAECkugb1EqbIvkmAiIgAiIQGkJSJRK23VyXAREQASqR0CiVL0+VYv6JqCCIiACoyYgURp1D6h+ERABERCBJgGJUhOFTkRABESgegTK1iKJUtl6TP6KgAiIQIUJSJQq3LlqmgiIgAiUjYBEqWw9Ngp/VacIiIAI5ERAopQTaFUjAiIgAiLQmYBEqTMj5RABEageAbWooAQkSgXtGLklAiIgAnUkIFGqY6+rzSIgAiJQUAISpQE6RkVFQAREQASGS0CiNFyesiYCIiACIjAAAYnSAPBUVASqR0AtEoHREpAojZa/ahcBERABEYgISJQiGDoVAREQAREYLYEsRGm0LVLtIiACIiACpSUgUSpt18lxERABEageAYlS9fpULcqCgGyKgAjkQkCilAtmVVI1Anv37m3bpE7pbQsrUQRqTECiVOPOV9P7J7Bu3Tq3atUq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)
#

# %% [markdown] id="-6yxvtvy5GB-"
# ## **בחירת מדדים:**
#
# **הרווח הכולל של הפרק**
#
# הרווח הכולל של הפארק הוא מדד הביצוע המרכזי הבוחן את הצלחתו הכלכלית של המודל. הרווח נגזר ממכירת כרטיסים, רכישה חד-פעמית בדוכני המזון, ורכישת תמונות ביציאה. מאחר שרכישת תמונות מותנית בשביעות הרצון של המבקר, מדד זה משמש כבבואה לרמת ההנאה שהופקה מהביקור.
#
# **זמן המתנה ממוצע בתור**
#
# זמן המתנה ממוצע בתור (כלל-פארקי) מהווה אינדיקציה ליעילות התפעולית ולעומס על תשתיות הפארק. במודל הנוכחי, זמן ההמתנה הוא גורם שמשפיע מאוד על חוויית המשתמש; המתנה ממושכת שוחקת את הנאת המבקר ומורידה את הדירוג הסובייקטיבי שהוא מעניק לפארק, מה שמשפיע ישירות על נכונותו להשקיע ברכישת מזכרות ותמונות בסיום היום.
#
# **אחוז נטישה מתורי האטרקציות**
#
# אחוז נטישה מתורי אטרקציות הוא מדד התנהגותי המזהה כשלים נקודתיים וצווארי בקבוק. נטישה היא "הכרזת אי-אמון" של המבקר במתקן מסוים, והיא משמשת כמשתנה המוריד את דירוג הפארק באופן משמעותי. מעקב אחר אחוז הנטישה מאפשר להבין לא רק כמה זמן אנשים המתינו, אלא באיזה אחוז ההמתנה הפכה לבלתי נסבלת עבורם.
#
# ---
#
# השילוב בין המדדים מאפשר לבחון את השרשרת הסיבתית בסימולציה: תפעול, חוויה,  רווח. בעוד שהכנסות מכרטיסים ואוכל הן יציבות יחסית, הרווח ממכירת תמונות הוא "רווח מבוסס רגש" . אם הסימולציה תראה זמני המתנה גבוהים ואחוזי נטישה גדלים, נוכל לצפות לירידה בדירוג הפארק ובהתאמה לירידה ברווח הכולל (למרות שכמות המבקרים לא השתנתה). המודל מאפשר לנו למצוא את "נקודת הקריסה" של החוויה – זו שמעבר לה, העומס כבר לא משתלם כלכלית כי הוא פוגע במכירות הקצה ביציאה.

# %% [markdown] id="PVu8GCez0H_5"
# # Visitiors Classes
#

# %% id="d60Wp5I30RJe" colab={"base_uri": "https://localhost:8080/", "height": 106} outputId="7479fceb-8771-4132-fceb-e6bcbd8d1c71"
class Visitor:
  def __init__(self, rank, age, group):
    self.rank = 10.0 # דירוג התחלתי
    self.group = group
    self.age = age #צריך להבין איפה מג'נרטים ברנדומליות גיל
    self.activity_diary = []

  def add_rank(self, adrenaline_level):
    # נוסחת הציון
    GS = self.group.amount_of_members
    score = (((GS - 1) / 5 )* 0.3) + (((adrenaline_level - 1) / 4) * 0.7)
    self.rank = min(10,self.rank + score)

  def decrease_rank(self, penalty):
    self.rank = max(0, self.rank - penalty)

  #implements the success of an activity during the day
  def enter_successful(self, activity):
    for act in self.activity_diary:
      if act[0] == activity:
        act[1] = True
        break

  #implements the failiure when abandoning a queue for an activity
  def failed_attempt(self, activity):
    for act in self.activity_diary:
      if act[0] == activity:
        act[2] = True
       # self.group.last_abandoned = act[0]
        break


class Group:
  def __init__(self, amount_of_members, max_wait_time):
    self.has_express = random.random() < 0.25
    self.amount_of_members = amount_of_members
    self.members = []
    self.pictures_cost = 0
   # self.last_abandoned = None
    self.max_wait_time = max_wait_time
    self.decided_on_lunch = False

  def increase_rank(self, adrenaline_level):
    for Visitor in self.members:
        Visitor.add_rank(adrenaline_level)

  def decrease_rank(self, decrease_value):
    for Visitor in self.members:
        Visitor.decrease_rank(decrease_value)

  # Generates the activity diary of the group in the beginning, Forces the inheritors to write this function
  def generate_activity_diary(self):
    pass

  # Returns the next possible activities that the group needs to proceed to, Forces the inheritors to write this function
  def get_candidate_activities(self, last_activity_tried):
    pass

  # Checkes if the first 'count' activities have already completed
  def is_phase_finished(self, count):
    return all(act[1] for act in self.members[0].activity_diary[:count])

  # Finds the minimal age on the group
  def find_min_age(self):
    minAge = self.members[0].age
    for i in range(1, len(self.members)):
      if (self.members[i].age < minAge):
        minAge = self.members[i].age
    return minAge

  def purchase_pictures(self):
    avg_rank = sum(m.rank for m in group.members) / len(group.members)

    # Charge according to average rank
    if avg_rank >= 8.5:
      # 10 photos and a video
      self.pictures_cost = 120
    elif 7.5 <= avg_rank < 8.5:
      # 10 photos
      self.pictures_cost = 100
    elif 6.0 <= avg_rank < 7.5:
      # 1 photo
      self.pictures_cost = 20
    else:
      #Not buying photos
      self.pictures_cost = 0

  def units_for(self, activity):

    # אם מדובר בבריכת ילדים, רק הילדים נכנסים
    if activity== "Kids Pool":
      kidsNum=0
      for member in self.members:
        if member.age < 4:
          kidsNum+=1
      return kidsNum 
    
    # כברירת מחדל לכל שאר המתקנים - כל הקבוצה נכנסת
    return self.amount_of_members


class SingleVisitor(Group):
  def __init__(self, age = 35):
    super().__init__(amount_of_members=1, max_wait_time=30)

  @staticmethod
  def CreateSingleVisitor():
    new_single_visitor = SingleVisitor()
    # Using reasonable upper bound of 30, choosing continous uniform distribution
    single_visitor_age = Algorithm.sample_continuous_uniform(18, 30)
    single_visitor = Visitor(rank=10, age= single_visitor_age, group=new_single_visitor)
    new_single_visitor.members.append(single_visitor)

  def get_candidate_activities(self, last_activity_tried):
    # Dividing the diary to 2 phases - before finishing the rides for ages 12 or more and after it
    phase1 = self.members[0].activity_diary[:3]
    phase2 = self.members[0].activity_diary[3:]

    # Wer'e on phase 1
    if not all(act[1] for act in phase1):
      candidates = [act[0] for act in phase1 if not act[1] and act[0] != last_activity_tried]

    # Wer'e on phase 2
    else:
      candidates = [act[0] for act in phase2 if not act[1] and act[0] != last_activity_tried]

    return candidates

#זמני הגעה וכמות משתתפים יג'ונרטו באירועים?
class Family(Group):
  def __init__(self):
    self.split = False
    super().__init__(2 + Algorithm.sample_number_of_children(1,5),15)
    self.split_groups = []
    self.leave_time = Algorithm.sample_family_leaving_time()
    # Decide if family will split (60% chance)
    self.will_split = random.random() < 0.6


  @staticmethod
  def CreateFamily(): # creates family, decides ages...
    new_family = Family()
    num_children = new_family.amount_of_members - 2
    # Create 2 parents
    for i in range(2):
      parent_age = 35  # Assuming all parents are 35
      parent = Visitor(rank=10, age=parent_age, group=new_family)
      new_family.members.append(parent)
    # Create children
    for i in range(num_children):
      child_age = Algorithm.sample_continuous_uniform(2, 18)  # age uniform continuous [2,18]
      child = Visitor(rank=10, age=child_age, group=new_family)
      new_family.members.append(child)

    return new_family


  def generate_activity_diary(self):#including spilt
    # Generate for family only the rides for all ages
    self.members[0].activity_diary = [["Lazy River", False, False], ["Big Tube Slide", False, False]]
    random.shuffle(self.members[0].activity_diary)

    # Makes sure all of the family has the same diary
    for i in range(1, len(self.members) - 1):
      self.members[i].activity_diary = copy.deepcopy(self.members[0].activity_diary)

  def generate_final_activity_diary(self):
    if self.split:
      for group in self.split_groups:
        group.generate_activity_diary()
    else:
      minAge = self.find_min_age
      if minAge >= 12:
        for member in self.members:
          member.activity_diary.extend(["Waves Pool", False, False], ["Small Tube Slide", False, False])
      if minAge >= 14:
        for member in self.members:
          member.activity_diary.extend(["Single Water Slide", False, False])
      if minAge >= 6:
        for member in self.members:
          member.activity_diary.extend(["Snorkeling Tour", False, False])
      if minAge <= 4:
        for member in self.members:
          member.activity_diary.extend(["Kids Pool", False, False])
      # לשים לב שצריך איפשהו בטיפול אירוע לשנות את הסדר של המתקנים לפי אורך תור

  def get_candidate_activities(self, last_activity_tried):
    # Phase 1 - before the decision on splitting
    phase1 = self.members[0].activity_diary[:2]

    # We're still on phase 1
    if not all(act[1] for act in phase1):
      return [act[0] for act in phase1 if not act[1] and act[0] != last_activity_tried]

    # We're after the decision on splitting
    phase2 = self.members[0].activity_diary[2:]
    return [act[0] for act in phase2 if not act[1] and act[0] != last_activity_tried]


    def decide_on_split(self):
    # If already split, return existing split groups
      if self.split:
        return self.split_groups

      if self.will_split:
        self.split = True
        # Separate parents and children
        parents = [m for m in self.members if m.age >= 18]
        children = [m for m in self.members if m.age < 18]

        # Categorize children by age
        young_children = [c for c in children if c.age < 8]  # # Must have parent or 12+
        middle_children = [c for c in children if 8 <= c.age < 12]  # Independent
        older_children = [c for c in children if c.age >= 12]  # Can supervise

        # Decide number of groups (2 or 3, equal probability)
        if random.random() < 0.5:
          num_groups = 2
        else:
          num_groups = 3

        if num_groups == 3 and self.can_split_into_three(self,  young_children, older_children, middle_children):
          self.split_groups = self.split_into_three_groups(parents, young_children, middle_children, older_children)

        else:
          self.split_groups = self.split_into_two_groups(parents, young_children, middle_children, older_children)

        # Generates the rest of the activities (after the family splitted)
        self.generate_final_activity_diary()
        return self.split_groups

      else: # Family doesn't split
          # Generates the rest of the activities (after the family decdided not to split)
          self.generate_final_activity_diary()
          return [self]

def distribute_children(self, children, num_groups, start_index=0):
  groups = [[] for _ in range(num_groups)]

  for i, child in enumerate(children[start_index:], start=start_index):
    group_index = i % num_groups
    groups[group_index].append(child)

  return groups

  #Split family to 2 groups
  def split_into_two_groups(self, parents, young, middle, older):

    # Start with parents
    group1_members = [parents[0]]
    group2_members = [parents[1]]

    # Distribute young children
    if young:
      # Distribute young children half and half
      young_groups = self.distribute_children(young, num_groups=2)
      group1_members.extend(young_groups[0])
      group2_members.extend(young_groups[1])

    # Distribute older children
    if older:
      older_groups = self.distribute_children(older, num_groups=2)
      group1_members.extend(older_groups[0])
      group2_members.extend(older_groups[1])

    # Distribute middle children
    middle_groups = self.distribute_children(middle, num_groups=2)
    group1_members.extend(middle_groups[0])
    group2_members.extend(middle_groups[1])

    # Create SplitedFamily objects
    return self.create_splitted_family([group1_members, group2_members])

  #Splitting a family to 3
  def split_into_three_groups(self, parents, young, middle, older):

    group1_members = [parents[0]]
    group2_members = [parents[1]]
    group3_members = [] # Group 3 is led by an older/middle child

    # Track remaining older and middle children
    remaining_older = older
    remaining_middle = middle


    if older:
      # Priority 1: An older child leads
      group3_members = [older[0]]
      remaining_older = older[1:]
    elif middle:
      # Priority 2: A middle child leads (if no older kids exist)
      group3_members = [middle[0]]
      remaining_middle = middle[1:]

    # Distribute young children
    if young:
      young_groups = self.distribute_children(young, num_groups=3)
      group1_members.extend(young_groups[0])
      group2_members.extend(young_groups[1])
      group3_members.extend(young_groups[2])

    # Distribute remaining older children
    if remaining_older:
      older_groups = self.distribute_children(remaining_older, num_groups=3)
      group1_members.extend(older_groups[0])
      group2_members.extend(older_groups[1])
      group3_members.extend(older_groups[2])

    # Distribute middle children
    if remaining_middle:
      middle_groups = self.distribute_children(middle, num_groups=3)
      group1_members.extend(middle_groups[0])
      group2_members.extend(middle_groups[1])
      group3_members.extend(middle_groups[2])

    # Create SplitedFamily objects
    return self.create_splitted_family([group1_members, group2_members, group3_members])

  #Check if a family can split into 3 according to instructions
  def can_split_into_three(self, young, older, middle):
    return len(older) + len(middle) > 0


  def create_splitted_family(self, sub_groups):
    # Create Splits
    for group in sub_groups:
      split_group = SplittedFamily(len(group), members, self)
      self.split_groups.append(split_group)


class SplittedFamily(Group):
  def __init__(self, amount_of_members, members, original_family):
    super().__init__(amount_of_members,15)
    self.members = copy.deepcopy(members)
    self.original_family = original_family
    self.has_express = original_family.has_express
    self.leave_time = original_family.leave_time

    # Update group reference for each member
    for member in self.members:
      member.group = self

  def generate_activity_diary(self):
    minAge = self.find_min_age()
    if minAge >= 12:
      for member in self.members:
        member.activity_diary.extend(["Waves Pool", False, False], ["Small Tube Slide", False, False])
    if minAge >= 14:
      for member in self.members:
        member.activity_diary.extend(["Single Water Slide", False, False])
    if minAge >= 6:
      for member in self.members:
        member.activity_diary.extend(["Snorkeling Tour", False, False])
    if minAge <= 4:
      for member in self.members:
        member.activity_diary.extend(["Kids Pool", False, False])
    # לשים לב שצריך איפשהו בטיפול אירוע לשנות את הסדר של המתקנים לפי אורך תור

  def get_candidate_activities(self, last_activity_tried):

    # Phase 1 - before the decision on splitting
    phase1 = self.members[0].activity_diary[:2]

    # We're still on phase 1
    if not all(act[1] for act in phase1):
      return [act[0] for act in phase1 if not act[1] and act[0] != last_activity_tried]

    # We're after the decision on splitting
    phase2 = self.members[0].activity_diary[2:]
    return [act[0] for act in phase2 if not act[1] and act[0] != last_activity_tried]



class Teenagers(Group):
  def __init__(self):
    super().__init__(Algorithm.sample_number_of_teenagers(), 20)

  @staticmethod
  def createTeenagers():
    new_teenage_group = Teenagers()
    # Create teenager members
    for i in range(new_teenage_group.amount_of_members):
        # Using reasonable upper bound of 17, choosing continous uniform distribution
        teen_age = Algorithm.sample_continuous_uniform(14, 17)
        teen = Visitor(rank=10, age=teen_age, group=new_teenage_group)
        new_teenage_group.members.append(teen)

  @staticmethod
  def buy_express(Teenagers):
    Teenagers.has_express = True
    return amount_of_members * 50

  def generate_activity_diary(self):
    # Generate for teenagers only the rides with 3 or more waves
    self.members[0].activity_diary = [["Single Water Slide", False, False], ["Small Tube Slide", False, False], ["Waves Pool", False, False], ["Snorkeling Tour", False, False]]
    random.shuffle(self.members[0].activity_diary)

    # Makes sure all the teenagers have the same diary
    for i in range(1, len(self.members) - 1):
      self.members[i].activity_diary = copy.deepcopy(self.members[0].activity_diary)

  def get_candidate_activities(self, last_activity_tried):
    # If we have express we don't care about the last activity
    if self.has_express:
      return [act[0] for act in self.members[0].activity_diary if not act[1]]

    # If we don't have express we care about the last activity
    return [act[0] for act in self.members[0].activity_diary if not act[1] and act[0] != last_activity_tried]



# %% [markdown] id="HbMpWMLK5VoR"
# #Facility Classes

# %% id="vvk6CdXiX7Pr"
class Park:
  def __init__(self):
    self.visitor_groups = []
    self.total_revenue = 0
    self.avg_rank = 0
    self.facilities = {} # a dictionary for all the park's facilities
    self.queues = {} # a dictionary for all the park's queues
    self.opening_hour = datetime(2025, 1, 1, 9, 0)
    self.closing_hour = datetime(2025, 1, 1, 19, 0)
    create_rides_and_food_stands()

  def create_rides_and_food_stands(self):
      # Setting all the park's queues
    self.queues["Park Entrance"] = Queue("Park Entrance")
    self.queues["Lazy River"] = Queue("Lazy River")
    self.queues["Single Water Slide"] = Queue("Single Water Slide")
    self.queues["Big Tube Slide"] = Queue("Big Tube Slide")
    self.queues["Small Tube Slide"] = Queue("Small Tube Slide")
    self.queues["Waves Pool"] = Queue("Waves Pool")
    self.queues["Kids Pool"] = Queue("Kids Pool")
    self.queues["Snorkeling Tour"] = Queue("Snorkeling Tour")
    # Setting all the park's facilities
    self.facilities["Park Entrance"] = ParkEntrance()
    self.facilities["Lazy River"] = LazyRiverAttraction(self.queues["Lazy River"])
    self.facilities["Single Water Slide"] = SingleWaterSlideAttraction(self.queues["Single Water Slide"])
    self.facilities["Big Tube Slide"] = BigTubeSlideAttraction(self.queues["Big Tube Slide"])
    self.facilities["Small Tube Slide"] = SmallTubeSlideAttraction(self.queues["Small Tube Slide"])
    self.facilities["Waves Pool"] = WavesPoolAttraction(self.queues["Waves Pool"])
    self.facilities["Kids Pool"] = KidsPoolAttraction(self.queues["Kids Pool"])
    self.facilities["Snorkeling Tour"] = SnorkelingTourAttraction(self.queues["Snorkeling Tour"])
    self.facilities["Pizza stand"] = PizzaFoodStand()
    self.facilities["Hamburger stand"] = HamburgerFoodStand()
    self.facilities["Salad stand"] = SaladFoodStand()

  


  def is_open(self, current_time):
    # Cheks if the stand is open according to the simulation time
    return self.opening_hour <= current_time <= self.closing_hour

  def cacl_amount_of_visitors(self):
    amount_of_visitors = 0
    for vg in self.visitor_groups:
      for m in vg.members:
        amount_of_visitors += 1

  def calc_avg_ranking(self):
    if not self.visitor_groups:
      self.avg_rank = 0
      return 0
    # Calculate the rating by suming all the rates and dividing by number of visitors
    total_rank = 0
    for vg in self.visitor_groups:
      for m in vg.members:
        total_rank += m.rank
    self.avg_rank = total_rank / self.cacl_amount_of_visitors
    return self.avg_rank

  def calc_total_revenue(self):
    return sum(facility.total_revenue for facility in self.facilities.values()) + \
           sum(v.pictures_cost for v in self.visitor_groups)

           ##need to add teenagers buying express after renegegd

class Facility:
  def __init__(self, name, num_servers, capacity_per_server):
    self.name = name
    self.available_servers = num_servers
    self.capacity_per_server = capacity_per_server
    self.queue = SmartQueue(name) #what? TODO
    self.total_revenue = 0

  def is_idle(self):
    return self.available_servers > 0

  def assign_server(self):
    self.available_servers -= 1

  def release_server(self):
    self.available_servers += 1

class ParkEntrance(Facility):
  def __init__(self, num_servers=3):
    super().__init__("Park Entrance", num_servers=num_servers, capacity_per_server=1)
    self.total_revenue = 0

  def calculate_and_charge(self, group):
    # חישוב מחיר כניסה ועדכון הכנסות
    group_total_cost = 0
    for member in group.members:
      # תמחור לפי גיל
      if member.age >= 14:
        ticket_price = 150
      else:
        ticket_price = 75

    # תוספת אקספרס לקבוצה (המחיר הוא לכל חבר בקבוצה)
    if group.has_express:
      ticket_price += 50

    group_total_cost += ticket_price

    self.total_revenue += group_total_cost
    return group_total_cost



# %% [markdown] id="JI9DXXToYCnC"
# # Rides Classes

# %% id="DrAjmYFN5dM-"
class Attraction(Facility):
    def __init__(self, name, adrenalinLevel, minAge, availableServers, rideCapacity, queue):
        super().__init__(name, availableServers, rideCapacity)
        self.adrenalinLevel = adrenalinLevel
        self.minAge = minAge
        self.queue=queue

    def enter_ride(self, units_to_enter,clock):
      pass

    def exit_ride(self, units_finished):
      pass

    def get_free_capacity(self) -> int:
      #get specific capacity
      pass

    def get_ride_time(self, isFirst = False):
      pass  
    def has_free_capacity(self) -> bool:
      #at least one free spot at least one free server
      pass

    def get_priority_session(self, simulation):
      #find the relevant session TODO
      pass
    def calculate_units_for_entry(self, group) -> int:
      #how much can I fit in the attraction from the group
      pass

class LazyRiverAttraction(Attraction):
    def __init__(self,queue):
      super().__init__("Lazy River", adrenalinLevel=1, minAge=0, availableServers=1, rideCapacity=60, queue=queue)
      self.tubeCapacity = 2
      self.occupiedTubes = 0

    def get_ride_time(self):
      return sample_continuous_uniform(20,30)

    def enter_ride(self, units_to_enter,clock):
      tubes_num=units_to_enter // self.tubeCapacity
      self.occupiedTubes += tubes_num

    def exit_ride(self, units_finished):
      tubes_num=units_finished // self.tubeCapacity
      self.occupiedTubes -= tubes_num


    def get_free_capacity(self) -> int:
      return (self.rideCapacity-self.occupiedTubes)*self.tubeCapacity



    def has_free_capacity(self) -> bool:
      return self.occupiedTubes < self.rideCapacity

    def calculate_units_for_entry(self, group) -> int:
      avail_tubes =self.rideCapacity-self.occupiedTubes
      if avail_tubes*2>group.amount_of_members:
        return group.amount_of_members
      else:
        return avail_tubes*2

      


class SingleWaterSlideAttraction(Attraction):
    def __init__(self, queue):
        super().__init__("Single Water Slide", adrenalinLevel=5, minAge=14, availableServers=2, rideCapacity=1,queue=queue)
        self.gate = timedelta(seconds=30) #new visitor every 30 seconds
        self.ride_time = timedelta(minutes=3) #takes excately 3 minutes
        self.next_entry_time = [open_time, open_time] # 2 servers
        self.last_server_assigned = None


    def enter_ride(self, units_to_enter,clock):
      for server_id, t in enumerate(self.next_entry_time):
            if clock >= t:
                self.last_server_assigned = server_id
                # reserve gate
                self.next_entry_time[server_id] = clock + self.gate
                break


      

    def exit_ride(self, units_finished):
      pass

    def get_free_capacity(self) -> int:
      num = 0
      for t in self.next_entry_time:  # Just iterate over times directly
          if clock >= t:
              num += 1
      return num

    def has_free_capacity(self) -> bool:
      for t in self.next_entry_time:  # Just iterate over times directly
          if clock >= t:
              return True
      return False


    def get_ride_time(self, isFirst = False):
      return self.ride_time

    def calculate_units_for_entry(self, group) -> int:
      return 1
        

    # Calculate how many people can be entered right now
    def calculate_units_for_entry(self, group) -> int:
      num = 0
      for t in self.next_entry_time:  # Just iterate over times directly
          if clock >= t:
              num += 1
      return num



class BigTubeSlideAttraction(Attraction):
    def __init__(self,queue):
      super().__init__("Big Tube Slide", adrenalinLevel=2, minAge=0, availableServers=1, rideCapacity=1, queue=queue)
      self.tubeCapacity = 8

    def get_ride_time(self):
      return sample_normal(4.800664, 1.823101)

    def enter_ride(self, units_to_enter,clock):
      self.assign_server()

    def exit_ride(self, units_finished):
      self.release_server()


    def get_free_capacity(self) -> int:
      return self.tubeCapacity

    def has_free_capacity(self) -> bool:
      return self.availableServers > 0

    def calculate_units_for_entry(self, group) -> int:
      return group.amount_of_members
      

class SmallTubeSlideAttraction(Attraction):
    def __init__(self,queue):
        super().__init__("Small Tube Slide", adrenalinLevel=4, minAge=12, availableServers=1, rideCapacity=1, queue=queue)
        self.tubeCapacity = 3

    def get_ride_time(self):
      return sample_exponential(2.107060)

    def enter_ride(self, units_to_enter,clock):
      self.assign_server()

    def exit_ride(self, units_finished):
      self.release_server()


    def get_free_capacity(self) -> int:
      return self.tubeCapacity

    def has_free_capacity(self) -> bool:
      return self.availableServers > 0

    def calculate_units_for_entry(self, group) -> int:
      if group.amount_of_members <= self.tubeCapacity:
        return group.amount_of_members  
      return self.tubeCapacity

class WavesPoolAttraction(Attraction):
    def __init__(self,queue):
        super().__init__("Waves Pool", adrenalinLevel=3, minAge=12, availableServers=1, rideCapacity=80, queue=queue)
        self.occupied_spots = 0  

    def enter_ride(self, units_to_enter,clock):
      self.occupied_spots += units_to_enter

    def exit_ride(self, units_finished):
      self.occupied_spots -= units_finished

    def get_ride_time(self):
      return generate_wavepool_time()

    def get_free_capacity(self) -> int:
        return self.rideCapacity-self.occupied_spots

    def check_spots(self, guest_size):
        if self.rideCapacity - self.occupied_spots >= guest_size:
            return True
        return False

    def has_free_capacity(self) -> bool:
      return self.occupied_spots < self.rideCapacity

    def calculate_units_for_entry(self, group) -> int:
      #how much can I fit in the attraction from the group
      if self.check_spots(group.size):
        return group.size
      return self.rideCapacity - self.occupied_spots


class KidsPoolAttraction(Attraction):
    def __init__(self,queue):
        super().__init__("Kids Pool", adrenalinLevel=1, minAge=0, availableServers=1, rideCapacity=30, queue=queue)
        self.maxAge = 4
        self.occupied_spots=0

    def get_ride_time(self):
      return generate_kids_pool_time() * 60

    def enter_ride(self, units_to_enter,clock):
      self.occupied_spots += units_to_enter

    def exit_ride(self, units_finished):
      self.occupied_spots -= units_finished


    def get_free_capacity(self) -> int:
        return self.rideCapacity-self.occupied_spots

    def check_spots(self, guest_size):
        if self.rideCapacity - self.occupied_spots >= guest_size:
            return True
        return False

    def has_free_capacity(self) -> bool:
      return self.occupied_spots < self.rideCapacity

    def calculate_units_for_entry(self, group) -> int:
      #how much can I fit in the attraction from the group
      if self.check_spots(group.units_for(self)):
        return group.units_for(self)
      return self.rideCapacity - self.occupied_spots

    

class SnorkelingTourAttraction(Attraction):
    def __init__(self,queue):
        super().__init__("Snorkeling Tour", adrenalinLevel=3, minAge=6, availableServers=2, rideCapacity=30, queue=queue)
        self.guide_break_duration = 30
        # חשוב: guides_ready_time חייב להכיל אובייקט datetime מלא
        self.guides_ready_time = [time(9,0)] * 2

    def get_ride_time(self):
      return sample_normal(30, 10)

    def is_idle(self, current_time):
        # בדיקת הפסקת צהריים
        if 13 <= current_time.hour < 14:
            return False
        # בדיקה אם יש מדריך שסיים הפסקה
        return any(ready_time <= current_time for ready_time in self.guides_ready_time)

    def assign_server(self, current_time):
        for i in range(len(self.guides_ready_time)):
            if self.guides_ready_time[i] <= current_time:
                self.guides_ready_time[i] = datetime.max # נועלים
                self.available_servers -= 1
                return i
        return None

    def release_server(self, guide_index, finish_time):
        # חישוב זמן מוכנות (סיום + 30 דקות)
        next_ready = finish_time + timedelta(minutes=self.guide_break_duration)
        self.guides_ready_time[guide_index] = next_ready
        self.available_servers += 1


# %% [markdown] id="A3CWWPaX6YjY"
# #Foodstand Classes

# %% id="WN7EVoRy6fBF"
class FoodStand:
  def __init__(self, name):
    self.name = name
    self.bad_meal_prob = 0.1 # הסתברות למנה לא טובה (0.1 במצב רגיל)
    self.total_revenue = 0
    self.opening_hour = datetime(2025, 1, 1, 13, 0)
    self.closing_hour = datetime(2025, 1, 1, 15, 0)

  @staicmethod
  def decide_on_stand():
    rand = random.random()
    if rand < 3/8:  # 3/8 prefer burger
      return "burger"
    elif 3/8 < rand < 5/8:  # 1/4 prefer pizza
      return "pizza"
    else:  # Prefer salad
      return "salad"
  def calculate_service_and_eating(self):
    # שירות: נורמלי (תוחלת 5, סטיית תקן 1.5)
    service_time = Algorithm.sample_normal(5, 1.5)
    # אכילה: אחיד (15-35 דקות)
    eating_time = Algorithm.sample_continuous_uniform(15, 35)
    return service_time + eating_time

  def get_total_duration(self, group):
    # פונקציה אבסטרקטית לחישוב זמן הכנה + זמן שירות ואכילה
    pass

  def is_open(self, current_time):
    # בדיקה האם הדוכן פתוח לפי השעה בסימולציה
    return self.opening_hour <= current_time <= self.closing_hour

  def process_meal(self, group):
    # פונקציה מרכזית שמופעלת מה-Event: מחשבת הכנסה ובודקת איכות מנה (השפעה על הדירוג).

    # חישוב הכנסה לפי סוג הדוכן והקבוצה
    income = self.calculate_income(group)
    self.total_revenue += income

    # לוגיקת מנה לא טובה (מוריד 0.8 מהדירוג)
    if random.random() < self.bad_meal_prob:
        group.decrease_rank(0.8)

  def calculate_income(self, group):
    # שיטה אבסטרקטית שחייבת מימוש בכל דוכן
    pass

class PizzaFoodStand(FoodStand):
  def __init__(self):
    super().__init__("Pizza Stand")
    self.prices = {"Personal": 40, "Group": 100}

  def calculate_income(self, group):
    if isinstance(group, Family) or isinstance (group, Teenagers):
        return self.prices["Group"]
    return self.prices["Personal"]

  def get_total_duration(self,group):
    prep_time=Algorithm.sample_continuous_uniform(4, 6)
    return prep_time + self.calculate_service_and_eating()

class HamburgerFoodStand(FoodStand):
  def __init__(self):
    super().__init__("Hamburger Stand")

  def get_total_duration(self,group):
    prep_per_person=Algorithm.sample_continuous_uniform(3, 4)
    prep_time = len(group.members) * prep_per_person
    return prep_time + self.calculate_service_and_eating()

  def calculate_income(self, group):
    return 100 * len(group.members)

class SaladFoodStand(FoodStand):
  def __init__(self):
    super().__init__("Salad Stand")

  def calculate_income(self, group):
    return 65 * len(group.members)

  def get_total_duration(self,group):
    prep_per_person=Algorithm.sample_continuous_uniform(3, 7)
    prep_time = len(group.members) * prep_per_person
    return prep_time + self.calculate_service_and_eating()


# %% [markdown] id="AN5kQuai64-D"
# # Queue Classes

# %% id="O-K9LmHl7Bg5"
class Queue:
    def __init__(self, facility_name: str):
        self.facility_name: str = facility_name
        self.queue: List[Any] = []

        # All metrics are in MINUTES (float)
        self.waiting_times: List[float] = []
        self.renege_waiting_times: List[float] = []
        self.renege_count: int = 0

        # Weighted queue length is in PERSON-MINUTES (float)
        self.last_change_time: Optional[datetime] = None
        self.weighted_queue_length_sum: float = 0.0

    # ------------------------------------------------------------
    # Park entrance buffer 
    # ------------------------------------------------------------
    def add_to_park_entrance(self, group: Any) -> None:
        self.queue.append(group)

    def pop_from_park_entrance(self) -> Optional[Any]:
        if not self.queue:
            return None
        return self.queue.pop(0)

    # ------------------------------------------------------------
    # Core queue API (for EndAttractionEvent)
    # ------------------------------------------------------------
    def is_empty(self) -> bool:
        return len(self.queue) == 0

    def __len__(self) -> int:
        return len(self.queue)

    def peek(self) -> Optional[Any]:
        if not self.queue:
            return None
        return self.queue[0]

    def get(self, index: int) -> Any:
        return self.queue[index]

    def add_group(self, group: Any, current_time: datetime) -> None:
        # Update weighted length before any structural change
        self._update_weighted_length(current_time)

        # Timestamp used for waiting/renege stats
        group.entry_time = current_time

        # Express insertion: after all existing express groups (FIFO within express)
        if getattr(group, "has_express", False):
            insert_pos = 0
            for i, g in enumerate(self.queue):
                if getattr(g, "has_express", False):
                    insert_pos = i + 1
                else:
                    break
            self.queue.insert(insert_pos, group)
        else:
            self.queue.append(group)

    def pop_next_group(self, current_time: datetime) -> Optional[Any]:
        # Pop FIFO head (express already ordered by add_group)
        if not self.queue:
            return None

        self._update_weighted_length(current_time)

        group = self.queue.pop(0)
        self._record_waiting_time(group, current_time)
        return group

    def pop_first_group_of_size(self, size: int, current_time: datetime) -> Optional[Any]:
        # scan from front and pop the FIRST group with exact match (amount_of_members == size)
        if size <= 0 or not self.queue:
            return None

        for i, group in enumerate(self.queue):
            if getattr(group, "amount_of_members", None) == size:
                self._update_weighted_length(current_time)
                popped = self.queue.pop(i)
                self._record_waiting_time(popped, current_time)
                return popped

        return None

    def remove_group_on_renege(self, group: Any, current_time: datetime) -> bool:
        # Called by external RenegeEvent
        if group not in self.queue:
            return False

        self._update_weighted_length(current_time)

        entry_time = getattr(group, "entry_time", None)
        if entry_time is not None:
            renege_wait_minutes = (current_time - entry_time).total_seconds() / 60.0
            self.renege_waiting_times.append(renege_wait_minutes)

        self.renege_count += 1
        self.queue.remove(group)

        try:
            if not isinstance(group, Teenagers):
                if hasattr(group, "decrease_rank"):
                    group.decrease_rank(0.8)
        except NameError:
            # If Teenagers is not defined in this module scope, apply the decrease (safe fallback)
            if hasattr(group, "decrease_rank"):
                group.decrease_rank(0.8)

        return True

    # ------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------
    def _record_waiting_time(self, group: Any, current_time: datetime) -> None:
        entry_time = getattr(group, "entry_time", None)
        if entry_time is None:
            return
        waiting_minutes = (current_time - entry_time).total_seconds() / 60.0
        self.waiting_times.append(waiting_minutes)

    def _update_weighted_length(self, current_time: datetime) -> None:
        # Accumulate person-minutes since last structural change
        if self.last_change_time is None:
            self.last_change_time = current_time
            return

        duration_minutes = (current_time - self.last_change_time).total_seconds() / 60.0
        if duration_minutes < 0:
            duration_minutes = 0.0

        self.weighted_queue_length_sum += len(self.queue) * duration_minutes
        self.last_change_time = current_time


# %% [markdown] id="YLCWrUTT80j6"
# #Event Classes

# %% id="-0yAbdqy84Qh" colab={"base_uri": "https://localhost:8080/", "height": 110} outputId="22103ebf-a046-4d80-c841-dc1f6d9797ec"
class Event:
  def __init__(self, time):
    self.time = time

  def __lt__(self, other):
    return self.time < other.time  # Ensure the heap sorts by time

  def handle(self, simulation):
    pass

  def updateRating(self):
    pass

class SingleVisitorArrivalEvent(Event):
  def __init__(self, time, group):
    super().__init__(time)
    self.group = group

  def handle(self, simulation):

    # handeling the current arrival
    if self.time <= datetime(2025, 1, 1, 18, 30):
      if simulation.queue_parkEntrance.queue:
        simulation.queue_parkEntrance.add_to_park_entrance(self.group)
      else:
        service_time = self.time + timedelta(minutes=Algorithm.sample_continuous_uniform(0.5, 2))
        service_time += timedelta(minutes=Algorithm.sample_exponential(2))
        simulation.schedule_event(EndGettingTicketEvent(service_time, self.group))

    # 1.5 a minute
    time_until_next_arrival = Algorithm.sample_exponential(1.5)
    next_arrival_time = self.time + timedelta(minutes=time_until_next_arrival)

    # taking care of the next arrival event creation
    group = SingleVisitor.CreateSingleVisitor()
    Simulation.Park.visitor_groups.append(group)
    if next_arrival_time.time() <= datetime(2025, 1, 1, 18, 30):
        simulation.schedule_event(SingleVisitorArrivalEvent(next_arrival_time, group))
    else:
        return
    # when next arrival time sampled is not within the arrival hours, the arrival event wil stop it's on creation


class FamilyArrivalEvent(Event):
  def __init__(self, time, group):
      super().__init__(time)
      self.group = group

  def handle(self, simulation):

    # handeling the current arrival
    if datetime(2025, 1, 1, 09, 00) <= self.time <= datetime(2025, 1, 1, 12, 00):
        if simulation.queue_parkEntrance.queue:
          simulation.queue_parkEntrance.add_to_park_entrance(self.group)
        else:
          service_time = self.time + timedelta(minutes=Algorithm.sample_continuous_uniform(0.5, 2))
          service_time += timedelta(minutes=Algorithm.sample_exponential(2))
          simulation.schedule_event(EndGettingTicketEvent(service_time, self.group))

    # 1.5 a minute
    time_until_next_arrival = Algorithm.sample_exponential(1.5)
    next_arrival_time = self.time + timedelta(minutes=time_until_next_arrival)

    # taking care of the next arrival event creation
    group = Family.CreateFamily()
    Simulation.Park.visitor_groups.append(group)
    if  next_arrival_time.time() <= datetime(2025, 1, 1, 12, 00):
        simulation.schedule_event(FamilyArrivalEvent(next_arrival_time, group))
    else:
        return
    # when next arrival time sampled is not within the arrival hours, the arrival event wil stop it's on creation


class TeenagersArrivalEvent(Event):
  def __init__(self, time, group):
    super().__init__(time)
    self.group = group

  def handle(self, simulation):
    # handeling the current arrival
    if datetime(2025, 1, 1, 10, 00) <= self.time <= datetime(2025, 1, 1, 16, 00):
      if simulation.queue_parkEntrance.queue:
        simulation.queue_parkEntrance.add_to_park_entrance(self.group)
      else:
        service_time = self.time + timedelta(minutes=Algorithm.sample_continuous_uniform(0.5, 2))
        service_time += timedelta(minutes=Algorithm.sample_exponential(2))
        simulation.schedule_event(EndGettingTicketEvent(service_time, self.group))

    # 500 groups in a day (360 minutes) is a lambda of 18/25
    time_until_next_arrival = Algorithm.sample_exponential(18/25)
    next_arrival_time = self.time + timedelta(minutes=time_until_next_arrival)

    # taking care of the next arrival event creation
    group = Teenagers.CreateTeenagers()
    Simulation.Park.visitor_groups.append(group)
    if next_arrival_time.time() <= datetime(2025, 1, 1, 16, 00):
        simulation.schedule_event(TeenagersArrivalEvent(next_arrival_time, group))
    else:
        return
    # when next arrival time sampled is not within the arrival hours, the arrival event wil stop it's on creation

class EndGettingTicketEvent(Event):
  def __init__(self, time, group):
    super().__init__(time)
    self.group = group

  def handle(self, simulation):
    # schedule the next group to end getting tickets
    if simulation.queue_parkEntrance.queue:
      service_time = self.time + timedelta(minutes=Algorithm.sample_continuous_uniform(0.5,2))
      service_time += timedelta(minutes=Algorithm.sample_exponential(2))
      simulation.schedule_event(EndGettingTicketEvent(service_time, simulation.queue_parkEntrance.pop_from_park_entrance()))

    # Getting the next activity for the group
    next_activity = simulation.get_best_next_activity(self.group, self.activity)

    # Handle sending the group to the next activity
    current_queue, current_ride = simulation.get_activity_queue_and_ride(next_activity)

    # Checks if the queue to the ride is empty or not
    if not current_queue and current_ride.is_idle(): # אין תור
      current_ride.assign_server()
      end_time = self.time + timedelta(minutes = current_ride.get_ride_time())
      simulation.schedule_event(EndAttractionEvent(end_time, self.group, next_activity))

    else:
      current_queue.add_group(self.group, self.time)
      # יוצרים אירוע נטישה בכל מקרה ואז מוחקים אותו מהיומן אירועים בסימולציה במידת הצורך
      next_abandonment_time = self.time + timedelta(minutes = self.group.max_wait_time)
      simulation.schedule_event(QueueAbandonmentEvent(next_abandonment_time, self.group, next_activity))

class LeavingEvent(Event):
    pass

class EndAttractionEvent(Event):
  def __init__(self, time, group, activity):
    super().__init__(time)
    self.group = group
    self.activity = activity

  def handle(self, simulation):
    # Mark the current activity as 'visited' and update the rank
    for member in self.group:
      self.group.members.enter_successful(self.activity)

    if (random.random() < 0.5):
      self.group.increase_rank(self.simulation.park.facilities[activity].adrenalinLevel)
    self.group.decrease_rank(0.1)

    # Handle lunch situation
    added_food_time = 0
    if datetime(2025, 1, 1, 13, 00) <= self.time.time() <= datetime(2025, 1, 1, 15, 00) and not self.group.decided_on_lunch:
      self.group.decided_on_lunch = True
      if (random.random() < 0.7):
        current_stand = FoodStand.decide_on_stand()
        added_food_time = simulation.park.facilities[current_stand].get_total_duration(self.group)
        simulation.park.facilities[current_stand].process_meal(self.group)

    # Create splited family after communal activities
    if isinstance (self.group, Family) and self.group.is_phase_finished(2): # Family that finished the last activity for all ages
      self.group.decide_on_split()
      for splittedFamily in self.group.split_groups:
        # להכניס פה קוד כניסת כל משפחה מפוצלת אם בכלל לכל המתקנים
        pass #TODO

    # Getting the next activity for the group
    next_activity = simulation.get_best_next_activity(self.group, self.activity)

    # Handle leaving from park after completing all of the activities
    if not isinstance (self.group, Family) or not isinstance (self.group, SplittedFamily):
      if next_activity == None: # completed all activities
        simulation.schedule_event(LeavingEvent(self.time + added_food_time, self.group))

    # Handle sending the group to the next activity
    current_queue, current_ride = simulation.get_activity_queue_and_ride(next_activity)

    # Checks if the queue to the ride is empty or not
    if not current_queue and current_ride.is_idle(): # אין תור
      current_ride.assign_server()
      end_time = self.time + timedelta(minutes = current_ride.get_ride_time())
      simulation.schedule_event(EndAttractionEvent(end_time, self.group, next_activity))

    else:
      current_queue.add_group(self.group, self.time)
      # יוצרים אירוע נטישה בכל מקרה ואז מוחקים אותו מהיומן אירועים בסימולציה במידת הצורך
      next_abandonment_time = self.time + timedelta(minutes = self.group.max_wait_time)
      simulation.schedule_event(QueueAbandonmentEvent(next_abandonment_time, self.group, next_activity))

  def updateRating(self):
    pass

class EndLunchEvent(Event):
  def __init__(self, time, group):
      super().__init__(time)
      self.group = group

  def handle(self, simulation):
    # Choose restaurant based on preferences
   if FoodStand.decide_on_stand() == "burger":
    total_lunch_time = HamburgerFoodStand.get_total_duration(self,group)
   if FoodStand.decide_on_stand() == "pizza":
    total_lunch_time = PizzaFoodStand.get_total_duration(self,group)
   else:
    total_lunch_time = SaladFoodStand.get_total_duration(self,group)

    # Calculate actual end time
    lunch_end_time = self.time + timedelta(minutes=total_lunch_time)

    # Update rating if customer was unsatisfied (10% chance) and add income
    FoodStand.process_meal(self, group)

    # Get the next activity for the group
    activity = self.group.get_next_activity(None)#not sure


    def updateRating(self):
        # Rating update is handled in handle() method
        pass

  def handle(self, simulation):
    pass

  def updateRating(self):
    pass

class QueueAbandonmentEvent(Event):
  def __init__(self, time, group, activity):
    super().__init__(time)
    self.group = group
    self.activity = activity

    # לשים לב להוסיף בטיפול אירוע, כאשר נערים עוזבים וקונים צמיד, להוסיף את הקניית צמיד איכשהו להכנסות הכוללות בפארק

class EndOfSimulation(Event):
  def __init__(self):
    pass

class Session:
  def __init__(self, group, activity, arrival_time):
    self.group = group
    self.attraction = attraction
    self.total_units = group.amount_of_members # כמות האנשים בקבוצה המקורית
    self.remaining_to_start = self.total_units # כמה אנשים מהקבוצה עוד לא נכנסו למתקן
    self.in_service = 0 # כמה אנשים מהקבוצה נמצאים כרגע בתוך המתקן
    self.arrival_time = arrival_time # זמן ההגעה לתור (לצורך חישובי סטטיסטיקה של המתנה)
    self.meta = {} # מילון גמיש לנתונים נוספים (כמו מזהה מדריך, מזהה אבוב וכו')

  def is_finished(self):
    #הקבוצה סיימה את המתקן רק כשכולם נכנסו וכולם יצאו
    return self.remaining_to_start == 0 and self.in_service == 0

  def record_entry(self, units):
    # מעדכן כשחלק מהקבוצה נכנס למתקן
    if units > self.remaining_to_start:
        raise ValueError(f"נסיו להכניס {units} אנשים, אך רק {self.remaining_to_start} נותרו ב-Session")
    self.remaining_to_start -= units
    self.in_service += units

  def record_exit(self, units):
    # מעדכן כשחלק מהקבוצה מסיים את המתקן ויוצא ממנו
    if units > self.in_service:
        raise ValueError(f"נסיון להוציא {units} אנשים, אך רק {self.in_service} נמצאים בשירות")
    self.in_service -= units

# %% [markdown] id="EglZTYyj8EZI"
# #Simulation Class

# %% id="d2JfwrJ48KAg"
class Simulation:
  def __init__(self):
    self.park = Park()
    self.clock = datetime(2025, 1, 1, 09, 00)
    self.event_diary =[] #minimum heap

  def run(self):
    while self.event_diary:
      event = heapq.heappop(self.event_diary)
      event.handle(self)

  def schedule_event(self, event):
    heapq.heappush(self.event_diary, event)

  # Gets the queue and the attraction based on the activity name
  def get_activity_queue_and_ride(self, activity):
    return self.park.queues[activity], self.park.facilities[activity]


  # Gets the next activity for the group
  def get_best_next_activity(self, group, last_activity_tried):

    # Finding the candidates for the next activities
    candidates = group.get_candidate_activities(last_activity_tried)

    if not candidates:
      return None

    # Teenagers don't care about the queue length
    if isinstance(group, Teenagers):
        return candidates[0]

    # Families in the first phase don't care about the queue length
    if isinstance(group, Family) and not group.is_phase_finished(2):
        return candidates[0]

    # Families in the second phase or single visitors seek for the shortest queue
    return self.find_shortest_in_list(candidates)

  def find_shortest_in_list(self, candidates):
    if not candidates:
        return None
    # פונקציית עזר למציאת המינימום לפי אורך התור בפארק
    return min(candidates, key=lambda name: len(self.park.queues[name].queue))

  def service_next_visitors(self, attraction, simulation):
      # Pops the next session/group to enter the attraction and creating an event for them
            
      # As long as we have free space on the ride
      while self.attraction.has_free_capacity():
        next_group = None
        units_to_enter = 0
        
        # Getting the lastest session
        candidate_session = self.attraction.get_priority_session()
        
        if candidate_session:
          next_group = candidate_session.group
          # How much space there is on the tube vs how much space there is on the session
          units_to_enter = min(self.attraction.free_capacity, candidate_session.remaining_to_start)
          candidate_session.record_entry(units_to_enter)
        
        # If there is no such session
        elif not self.attraction.queue.is_empty():
          next_group = self.attraction.queue.pop_next_group() # לבדוק עם דנה איך קוראים לפונקציה
          
          # Creating a new session
          new_session = Session(next_group, self.attraction, self.time)
          simulation.sessions[(next_group, self.attraction.name)] = new_session
          
          # Calculates how many units enter
          units_to_enter = self.attraction.calculate_units_for_entry(next_group)
          new_session.record_entry(units_to_enter)
        
        # There is no one to pull
        else:
          break
              
        # Making the next event
        if units_to_enter > 0:
          self.attraction.occupancy += units_to_enter
          service_duration = self.attraction.get_ride_time()
          end_time = self.time + timedelta(minutes=service_duration)
          simulation.schedule_event(self.attraction(end_time, next_group, self.attraction, units_to_enter))
  def route_group_to_next(self, group, current_time, next_activity=None):
    # Routing a group to it's next destenation - next activity or leaving
    
    # Finding the next activity
    if next_activity is None:
        next_activity = self.get_best_next_activity(group)

    # In case of finishing all the activities (not for families or splitted families)
    if isinstance(group, SingleVisitor) or isinstance (group, Teenagers):
      if next_activity is None:
        self.schedule_event(LeavingEvent(current_time, group))
        return

    # Finiding the queue and attraction
    queue, attraction = self.get_activity_queue_and_ride(next_activity)

    # If the group can enter immediately
    if attraction.can_enter_immediately(group):
      end_time = current_time + timedelta(minutes=attraction.get_ride_time())
      self.schedule_event(EndAttractionEvent(end_time, group, attraction, group.units_for(next_activity)))
    
    # If they can't enter immediately
    else:
      # Inserts the group to the queue
      queue.add_group(group, current_time)
      
      # Making an abandonment event (will be canceled if needed)
      abandon_time = current_time + timedelta(minutes=group.max_wait_time)
      self.schedule_event(QueueAbandonmentEvent(abandon_time, group, next_activity))
      
      # ניסיון "למשוך" מהתור (למקרה שהמתקן פנוי והתור היה ריק)
      next_activity.try_pull_from_queue(self, current_time)
      self.service_next_visitors(attraction, current_time)

