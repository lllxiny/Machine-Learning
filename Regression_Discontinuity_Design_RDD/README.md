**Regression Discontinuity Design (RDD) Analysis**

This Python code performs a Regression Discontinuity Design (RDD) analysis using the "drinking.csv" dataset to explore whether the legal drinking age of 21 impacts the chances of death due to accidents, suicide, and other reasons. The code calculates average death rates on each side of the age threshold (21) and compares them.

1. **Data Preparation:**
   - The code loads required packages and the "drinking.csv" dataset.
   - The dataset is viewed, and a new column "legal_to_drink" is created to tag whether an individual is legally allowed to drink based on their age.
   - The dataset is divided into two groups: those who are legal to drink and those who are not, with a bandwidth of 1 year (ages 20 and 21).

2. **RDD Analysis with Bandwidth of 1:**
   - The code calculates average death rates for the "legal to drink" and "not legal to drink" groups for accidents, suicide, and other reasons.
   - A DataFrame is created to store the results and differences between the two groups.
   - Results are displayed, showing that becoming legally allowed to drink increases the death rates for all three reasons, but the increase is relatively small.

3. **Graphical Representation - RDD with Bandwidth of 1:**
   - Three plots are generated to show the discontinuity at the age threshold (21) for the chances of death due to accidents, suicide, and other reasons.
   - The plots include points, a vertical line at age 21, a smoothed line (using linear regression), and labels.

4. **RDD Analysis with Maximized Bandwidth:**
   - The code calculates average death rates for the "legal to drink" and "not legal to drink" groups using the entire dataset, maximizing the bandwidth.
   - Results are displayed, showing that becoming legally allowed to drink decreases the chance of death due to accidents but increases the chances of death due to suicide and other reasons.

5. **Graphical Representation - RDD with Maximized Bandwidth:**
   - Three plots are generated, similar to the previous set of plots, but with a larger bandwidth. This helps visualize the impact of wider bandwidth on the treatment effect.

**Observations:**
- With a smaller bandwidth (1 year), the treatment effect of becoming legal to drink on increasing death rates is more pronounced.
- With a larger bandwidth, observations further from the cutoff (21) might differ in other characteristics, potentially causing the treatment effect to be less obvious due to confounding variables.
