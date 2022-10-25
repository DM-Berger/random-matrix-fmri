We attempt to demonstrate that RMT features (or sets of features, or RMT
feature extraction algorithms - here all just referred to as "RMT features",
for brevity) have *general* potential in fMRI. However, it is unlikely that
*any* feature has universal utility (cite: no free lunch theorem), and in
practice the value of a feature will depend on the data distribution, data
preprocessing decisions, extraction hyperparameters, choice of model, and
other significant analytic decisions.

Nevertheless, claims have been made to the potential *universal* value of RMT.
[cites]. This is despite the fact that most of RMT requires the assumption of
iid random matrices, and very very large matrix dimensions, and that often there
are no

[MAYBE CITE RMT UNFOLDING / TRIMMING SENSITIVITY HERE]

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
\begin{align}
    \mathcal{F}       &= \{\mathcal{F}_i : i = 1, \;\dots,\; F\} \\
    \mathcal{D}       &= \{\mathcal{D}_i : i = 1, \;\dots,\; D\} \\
    \mathcal{C}^{k} &= \{\mathcal{C}^{k}_i : i = 1, \;\dots,\; c_i\}, \quad k = 1, \dots, n \\
    \mathcal{C}       &= \{ \mathcal{C}^{k} : k = 1, \;\dots,\; n \} \\
    \Omega            &= \{ \mathcal{F}, \mathcal{D}, \mathcal{C}^{1}, \;\dots,\; \mathcal{C}^{n} \}
\end{align}
$$

We can call each element of $\Omega$ a *component* of the analysis Then for every subset of
$\Omega$, we can define a *grouping* of components across which produces a *summary*. For example,
one such grouping is $S = \mathcal{D} \times \mathcal{F}$, where $\times$ is the Cartesian product.
The components left behind by each grouping are those we *summarize across*, i.e., for which we
ignore or lump together the differences. E.g. for the grouping $\mathcal{D} \times \mathcal{F}$, we
summarize across all elements of $S^{\prime} = \mathcal{C}^1 \times \dots \times \mathcal{C}^n$.
Such a summary $S$ has $|S|$ *summary values*, and each *summary value* in reduces $|S^{\prime}|$
performance metrics to a single real number.

Thus, to "group by $S$" is equivalent to "summarizing across $S^{\prime}$". In the above example,


 *size* of the summary is the number of values within in of

So for

$$

$$


and $\times$ is the Cartesian product, then we can define various groupings across
which to summarize the data. Namely,

$$
\begin{align}
    \mathcal{F} \\
    \mathcal{D} \\
    \mathcal{D} \times \mathcal{F} \\
    \mathcal{D} \times \mathcal{F} \times \mathcal{C}^{(k)} \\
    \mathcal{D} \times \mathcal{F} \times \mathcal{C}^{(k_i)} \\


    \mathcal{C}^{(k)} &= \{\mathcal{C}^{(k)}_i : i = 1, \dots, c_i\}, k = 1, \dots, C \\
\end{align}
$$



One can then summarize these results through various grouping and aggregation
decisions.



For each
combination of datasets and analytic choices
a validation procedure
procedure in this case

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

