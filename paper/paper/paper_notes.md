We investigate whether RMT-based features (or sets of features, or RMT feature extraction
algorithms—here all just referred to as "RMT features", for brevity) have *general* potential for
predictive analyses in fMRI. However, it is unlikely that *any* feature has universal utility (cite:
no free lunch theorem), and in practice the value of a feature will depend on the data distribution,
data preprocessing decisions, extraction hyperparameters, choice of model, and other significant
analytic decisions.

Nevertheless, claims have been made to the potential *universal* value of RMT. [cites]. This is
despite the fact that even the most recent RMT advances only apply to iid random matrices, and
[@bouchaudFinancialApplicationsRandom2009]. In addition, previous studies typically choose one
unfolding procedure and one trimming procedure, and document this only minimally, despite this being
known to dramatically impact conclusions. Likewise, only one fMRI preprocessing procedure might be
used.

[MAYBE CITE RMT UNFOLDING / TRIMMING SENSITIVITY HERE]

For the skeptical reader, this does not present a very convincing analysis. Since most complex,
real-world data will not be IID or even approximately so, the relevance of particular RMT
distributions and/or metrics is theoretically unclear. If, in addition, relating real data to RMT
requires a number of analytic choices that can dramatically impact conclusions, and if there is no
way to *objectively* decide on which of such choices are correct, then these "researcher degrees of
freedom" [cite Gelman] greatly limit the universal applicability of RMT outside of mathematics and
physics.

Thus, to examine the general utility of RMT in fMRI, we perform a *multiverse* analysis []
which takes into account a number analytic choices relevant in fMRI and RMT. That is, we
examine the impact of:

- preprocessing degree
- unfolding trimming method
- unfolding polynomial degree
- feature subsetting / slicing
- normalization
- classifier
- evaluation metric

on the apparent performance of a number of RMT features extracted across a small selection
of fMRI datasets.

# Summarizing the Data


There are thus necessarily fundamental limits on the generalizability of any
feature, and one must exercise some caution when summarizing feature behaviour
across a wide variety of datasets and analytic choices.

For example, suppose one tests, on a collection of datasets, a wide variety of analytic choices
grouped into various categories, such as preprecessing, smoothing, dimensionality reduction, model
hyperparameters, and etc. Suppose there are datasets $\mathcal{D}_i$, $i = 1, \dots, d$, and for
each of the $n$ categories of analytic choice, $\mathcal{C}^{(j)}$ , each with $c_j$ choices.
Suppose for each of the $m \times \prod_{j=1}^n c_j$ combinations of analytic choices and datasets,
there is also a validation procedure (e.g. cross-validation) which produces a single *result* (such
as the mean performance metric across validation folds). This process is repeated for each of the
$F$ features $\mathcal{F}_k$ under investigation.

Notationaly, if we define the sets:

$$
\begin{aligned}
    \mathcal{F}       &= \{\mathcal{F}_i : i = 1, \;\dots,\; F\} \\
    \mathcal{D}       &= \{\mathcal{D}_i : i = 1, \;\dots,\; D\} \\
    \mathcal{C}^{k} &= \{\mathcal{C}^{k}_i : i = 1, \;\dots,\; c_i\}, \quad k = 1, \dots, n \\
    \mathcal{C}       &= \{ \mathcal{C}^{k} : k = 1, \;\dots,\; n \} \\
    \Omega            &= \{ \mathcal{F}, \mathcal{D}, \mathcal{C}^{1}, \;\dots,\; \mathcal{C}^{n} \}
\end{aligned}
$$

We can call each element of $\Omega$ a *component* of the analysis, and every unique combination of
components produces a *result*. In addition, for every subset of $\Omega$, we can define a *grouping* of
components across which produces a *summary* of results defined by that grouping.

For example, one such grouping is by data and features, i.e. $S = \mathcal{D} \times \mathcal{F}$,
where $\times$ is the Cartesian product. The components left behind by each grouping are those we
*summarize across*, i.e., for which we ignore or lump together the differences. E.g. for the
grouping $\mathcal{D} \times \mathcal{F}$, we summarize across all elements of $S^{\prime} =
\mathcal{C}^1 \times \dots \times \mathcal{C}^n$, i.e. across all analytic choices. Such a summary
$S$ has $|S|$ *summary values*, and each *summary value* reduces $|S^{\prime}|$ performance
metrics to a single real number.  Thus, to "group by $S$" is equivalent to "summarizing across $S^{\prime}$".

For any feature multiverse analsis, there are some natural groupings to use to summarize:

$$
\begin{aligned}
    & \mathcal{F} \\
    & \mathcal{D} \\
    & \mathcal{D} \times \mathcal{F} \\
    & \mathcal{D} \times \mathcal{F} \times \mathcal{C}^{k} \text{ for some } k\\
    & \mathcal{D} \times \mathcal{F} \times \prod \mathcal{C}_{\text{sub}}
    \text{ for any }
    \mathcal{C}_{\text{sub}} \subseteq \mathcal{C} \\
\end{aligned}
$$



In
practice, features are extracted by algorithms controlled by a number of
hyperparameters.
have a number of hyperparameters to tune. In addition, most data require
varying degrees and kinds of pre-processing. More complex data distributions
will generally require more flexible classifiers, whereas data with low
signal-to-noise ratio (SNR) will usually require less flexible classifiers, and
so one cannot generalize feature utility by using just a single predictive
algorithm alone.

In practice, we can simply
consider pre-processing decisions, and all other analytic choices as
*hyperparameters* in the feature extraction process.

Any modern examination of the utility of a
set of features must consider the sensitivity to such hyperparameters. In
addition, few features can be expected to be useful for all kinds of data.
Investing the utlity of a feature will thus create a large combination of
blocks, and interpreting the overall utility will require extensive
summarizations of the findings across various groupings of blocks.



When summarizing the value of a feature extraction procedure, the choice of summary metric reflects the   one must thus make a decision whetherWhether or not we describe the feature performance for the "best" hyperparameters for a set of data or the "median/mean performance" relative to the parameters is an important choice, each which has advantages and disadvantages.

