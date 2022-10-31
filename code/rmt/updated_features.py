# fmt: off
import sys  # isort: skip
from pathlib import Path  # isort: skip
ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(ROOT))
# fmt: on

from abc import ABC, abstractproperty
from pathlib import Path
from typing import Type

import pandas as pd
from pandas import DataFrame
from scipy.ndimage import uniform_filter1d
from scipy.signal import savgol_filter

from rmt.enumerables import PreprocLevel, TrimMethod, UpdatedDataset
from rmt.updated_dataset import UpdatedProcessedDataset, levelvars, rigidities

PROJECT = ROOT.parent
RESULTS = PROJECT / "results"
PLOT_OUTDIR = RESULTS / "plots"


class UpdatedFeature(ABC):
    def __init__(
        self,
        source: UpdatedDataset,
        preproc: PreprocLevel,
        norm: bool,
        degree: int | None = None,
        trim: TrimMethod | None = None,
    ) -> None:
        super().__init__()
        self.source = source
        self.preproc = preproc
        self.norm: bool = norm
        self.degree = degree
        self.trim = trim
        self.name: str = self.__class__.__name__.lower()
        self.dataset: UpdatedProcessedDataset = UpdatedProcessedDataset(
            source=self.source,
            preproc_level=self.preproc,
        )
        self.is_combined = False
        self.feature_start_idxs = [int(0)]

    @property
    def suptitle(self) -> str:
        trim = "None" if self.trim is None else self.trim.value
        deg = "" if self.degree is None else f" deg={self.degree}"
        return f"{self.dataset}: norm={self.norm}{deg} trim={trim}"

    @property
    def fname(self) -> str:
        trim = "None" if self.trim is None else self.trim.value
        deg = "" if self.degree is None else f"_deg={self.degree}"
        src = self.source.name
        pre = self.preproc
        return f"{src}_pre={pre.name}_norm={self.norm}{deg}_trim={trim}.png"

    @classmethod
    def outdir(cls) -> Path:
        out = PLOT_OUTDIR / cls.__name__.lower()
        out.mkdir(exist_ok=True, parents=True)
        return out

    @abstractproperty
    def data(self) -> DataFrame:
        ...


class Rigidities(UpdatedFeature):
    def __init__(
        self,
        source: UpdatedDataset,
        preproc: PreprocLevel,
        norm: bool,
        degree: int | None = None,
        trim: TrimMethod | None = None,
    ) -> None:
        assert degree is not None
        self.degree: int
        super().__init__(
            source=source,
            preproc=preproc,
            norm=norm,
            degree=degree,
            trim=trim,
        )

    @property
    def data(self) -> DataFrame:
        df = rigidities(
            dataset=self.dataset, degree=self.degree, trim_method=self.trim, parallel=True
        )
        return df


class Levelvars(UpdatedFeature):
    def __init__(
        self,
        source: UpdatedDataset,
        preproc: PreprocLevel,
        norm: bool,
        degree: int | None = None,
        trim: TrimMethod | None = None,
    ) -> None:
        assert degree is not None
        self.degree: int
        super().__init__(
            source=source,
            preproc=preproc,
            norm=norm,
            degree=degree,
            trim=trim,
        )

    @property
    def data(self) -> DataFrame:
        df = levelvars(
            dataset=self.dataset, degree=self.degree, trim_method=self.trim, parallel=True
        )
        return df


class Eigenvalues(UpdatedFeature):
    def __init__(
        self,
        source: UpdatedDataset,
        preproc: PreprocLevel,
        norm: bool,
        degree: int | None = None,
        trim: TrimMethod | None = None,
    ) -> None:
        super().__init__(
            source=source,
            preproc=preproc,
            norm=norm,
            degree=None,
            trim=None,
        )

    @property
    def data(self) -> DataFrame:
        return self.dataset.eigs_df()


class EigsMinMax5(UpdatedFeature):
    def __init__(
        self,
        source: UpdatedDataset,
        preproc: PreprocLevel,
        norm: bool,
        degree: int | None = None,
        trim: TrimMethod | None = None,
    ) -> None:
        super().__init__(
            source=source,
            preproc=preproc,
            norm=norm,
            degree=None,
            trim=None,
        )

    @property
    def data(self) -> DataFrame:
        eigs = self.dataset.eigs_df()
        y = eigs["y"].copy()
        min5 = eigs.iloc[:, :5]
        max5 = eigs.drop(columns="y").iloc[:, -5:]
        df = pd.concat([min5, max5], axis=1)
        df["y"] = y
        return df


class EigsMinMax10(UpdatedFeature):
    def __init__(
        self,
        source: UpdatedDataset,
        preproc: PreprocLevel,
        norm: bool,
        degree: int | None = None,
        trim: TrimMethod | None = None,
    ) -> None:
        super().__init__(
            source=source,
            preproc=preproc,
            norm=norm,
            degree=None,
            trim=None,
        )

    @property
    def data(self) -> DataFrame:
        eigs = self.dataset.eigs_df()
        y = eigs["y"].copy()
        min10 = eigs.iloc[:, :10]
        max10 = eigs.drop(columns="y").iloc[:, -10:]
        df = pd.concat([min10, max10], axis=1)
        df["y"] = y
        return df


class EigsMinMax20(UpdatedFeature):
    def __init__(
        self,
        source: UpdatedDataset,
        preproc: PreprocLevel,
        norm: bool,
        degree: int | None = None,
        trim: TrimMethod | None = None,
    ) -> None:
        super().__init__(
            source=source,
            preproc=preproc,
            norm=norm,
            degree=None,
            trim=None,
        )

    @property
    def data(self) -> DataFrame:
        eigs = self.dataset.eigs_df()
        y = eigs["y"].copy()
        min20 = eigs.iloc[:, :20]
        max20 = eigs.drop(columns="y").iloc[:, -20:]
        df = pd.concat([min20, max20], axis=1)
        df["y"] = y
        return df


class EigsMiddle10(UpdatedFeature):
    def __init__(
        self,
        source: UpdatedDataset,
        preproc: PreprocLevel,
        norm: bool,
        degree: int | None = None,
        trim: TrimMethod | None = None,
    ) -> None:
        super().__init__(
            source=source,
            preproc=preproc,
            norm=norm,
            degree=None,
            trim=None,
        )
        self.length = 10

    @property
    def data(self) -> DataFrame:
        length = self.length
        eigs = self.dataset.eigs_df()
        y = eigs["y"].copy()
        eigs.drop(columns="y", inplace=True)
        half = len(eigs.columns) // 2
        left = eigs.iloc[:, half - (length // 2) : half]
        right = eigs.iloc[:, half : half + (length // 2)]
        df = pd.concat([left, right], axis=1)
        df["y"] = y
        return df


class EigsMiddle20(EigsMiddle10):
    def __init__(
        self,
        source: UpdatedDataset,
        preproc: PreprocLevel,
        norm: bool,
        degree: int | None = None,
        trim: TrimMethod | None = None,
    ) -> None:
        super().__init__(
            source=source,
            preproc=preproc,
            norm=norm,
            degree=None,
            trim=None,
        )
        self.length = 20

    @property
    def data(self) -> DataFrame:
        length = self.length
        eigs = self.dataset.eigs_df()
        y = eigs["y"].copy()
        eigs.drop(columns="y", inplace=True)
        half = len(eigs.columns) // 2
        left = eigs.iloc[:, half - (length // 2) : half]
        right = eigs.iloc[:, half : half + (length // 2)]
        df = pd.concat([left, right], axis=1)
        df["y"] = y
        return df


class EigsMiddle40(UpdatedFeature):
    def __init__(
        self,
        source: UpdatedDataset,
        preproc: PreprocLevel,
        norm: bool,
        degree: int | None = None,
        trim: TrimMethod | None = None,
    ) -> None:
        super().__init__(
            source=source,
            preproc=preproc,
            norm=norm,
            degree=None,
            trim=None,
        )
        self.length = 50

    @property
    def data(self) -> DataFrame:
        length = self.length
        eigs = self.dataset.eigs_df()
        y = eigs["y"].copy()
        eigs.drop(columns="y", inplace=True)
        half = len(eigs.columns) // 2
        left = eigs.iloc[:, half - (length // 2) : half]
        right = eigs.iloc[:, half : half + (length // 2)]
        df = pd.concat([left, right], axis=1)
        df["y"] = y
        return df


class EigenvaluesSmoothed(UpdatedFeature):
    def __init__(
        self,
        source: UpdatedDataset,
        preproc: PreprocLevel,
        norm: bool,
        degree: int | None = None,
        trim: TrimMethod | None = None,
    ) -> None:
        self.degree: int
        super().__init__(
            source=source,
            preproc=preproc,
            norm=norm,
            degree=degree,
            trim=None,
        )

    @property
    def data(self) -> DataFrame:
        df = self.dataset.eigs_df()
        y = df["y"].copy()
        df.drop(columns="y", inplace=True)
        smoothed = DataFrame(
            uniform_filter1d(df, size=self.degree, axis=-1, mode="constant")
        )
        smoothed.columns = df.columns
        smoothed["y"] = y
        return smoothed


class EigenvaluesSavGol(UpdatedFeature):
    def __init__(
        self,
        source: UpdatedDataset,
        preproc: PreprocLevel,
        norm: bool,
        degree: int | None = None,
        trim: TrimMethod | None = None,
    ) -> None:
        self.degree: int
        super().__init__(
            source=source,
            preproc=preproc,
            norm=norm,
            degree=degree,
            trim=None,
        )

    @property
    def data(self) -> DataFrame:
        df = self.dataset.eigs_df()
        y = df["y"].copy()
        df.drop(columns="y", inplace=True)
        order = 2 if self.degree == 3 else 3
        smoothed = DataFrame(
            savgol_filter(
                df, window_length=self.degree, polyorder=order, axis=-1, mode="constant"
            )
        )
        smoothed.columns = df.columns
        smoothed["y"] = y
        return smoothed


class EigenvaluesPlusEigenvaluesSmoothed(UpdatedFeature):
    def __init__(
        self,
        source: UpdatedDataset,
        preproc: PreprocLevel,
        norm: bool,
        degree: int | None = None,
        trim: TrimMethod | None = None,
    ) -> None:
        self.degree: int
        super().__init__(
            source=source,
            preproc=preproc,
            norm=norm,
            degree=degree,
            trim=None,
        )

    @property
    def data(self) -> DataFrame:
        eigs = self.dataset.eigs_df()
        y = eigs["y"].copy()
        eigs.drop(columns="y", inplace=True)  # remove target column
        self.feature_start_idxs.append(len(eigs.columns))

        df = self.dataset.eigs_df()
        df.drop(columns="y", inplace=True)
        smoothed = DataFrame(
            uniform_filter1d(df, size=self.degree, axis=-1, mode="constant")
        )
        smoothed.columns = df.columns

        df = pd.concat([eigs, smoothed], axis=1, ignore_index=True)
        df["y"] = y
        return df


class EigenvaluesPlusSavGol(UpdatedFeature):
    def __init__(
        self,
        source: UpdatedDataset,
        preproc: PreprocLevel,
        norm: bool,
        degree: int | None = None,
        trim: TrimMethod | None = None,
    ) -> None:
        self.degree: int
        super().__init__(
            source=source,
            preproc=preproc,
            norm=norm,
            degree=degree,
            trim=None,
        )

    @property
    def data(self) -> DataFrame:
        eigs = self.dataset.eigs_df()
        y = eigs["y"].copy()
        eigs.drop(columns="y", inplace=True)  # remove target column
        self.feature_start_idxs.append(len(eigs.columns))

        df = self.dataset.eigs_df()
        df.drop(columns="y", inplace=True)
        order = 2 if self.degree == 3 else 3
        smoothed = DataFrame(
            savgol_filter(
                df, window_length=self.degree, polyorder=order, axis=-1, mode="constant"
            )
        )
        smoothed.columns = df.columns

        df = pd.concat([eigs, smoothed], axis=1, ignore_index=True)
        df["y"] = y
        return df


class Unfolded(UpdatedFeature):
    def __init__(
        self,
        source: UpdatedDataset,
        preproc: PreprocLevel,
        norm: bool,
        degree: int | None = None,
        trim: TrimMethod | None = None,
    ) -> None:
        self.degree: int
        super().__init__(
            source=source,
            preproc=preproc,
            norm=norm,
            degree=degree,
            trim=trim,
        )

    @property
    def data(self) -> DataFrame:
        return self.dataset.unfolded_df(self.degree, self.trim)


class EigPlusUnfolded(UpdatedFeature):
    def __init__(
        self,
        source: UpdatedDataset,
        preproc: PreprocLevel,
        norm: bool,
        degree: int,
        trim: TrimMethod | None = None,
    ) -> None:
        self.degree: int
        super().__init__(
            source=source,
            preproc=preproc,
            norm=norm,
            degree=int(degree),
            trim=trim,
        )
        self.is_combined = True

    @property
    def data(self) -> DataFrame:
        eigs = self.dataset.eigs_df()
        y = eigs["y"].copy()
        eigs.drop(columns="y", inplace=True)  # remove target column
        self.feature_start_idxs.append(len(eigs.columns))
        unfs = self.dataset.unfolded_df(self.degree, self.trim)
        unfs.drop(columns="y", inplace=True)
        df = pd.concat([eigs, unfs], axis=1, ignore_index=True)
        df["y"] = y
        return df


class EigPlusRigidity(UpdatedFeature):
    def __init__(
        self,
        source: UpdatedDataset,
        preproc: PreprocLevel,
        norm: bool,
        degree: int,
        trim: TrimMethod | None = None,
    ) -> None:
        self.degree: int
        super().__init__(
            source=source,
            preproc=preproc,
            norm=norm,
            degree=int(degree),
            trim=trim,
        )
        self.is_combined = True

    @property
    def data(self) -> DataFrame:
        eigs = self.dataset.eigs_df()
        y = eigs["y"].copy()
        eigs.drop(columns="y", inplace=True)  # remove target column
        self.feature_start_idxs.append(len(eigs.columns))
        rigs = rigidities(
            dataset=self.dataset, degree=self.degree, trim_method=self.trim, parallel=True
        )
        rigs.drop(columns="y", inplace=True)
        df = pd.concat([eigs, rigs], axis=1, ignore_index=True)
        df["y"] = y
        return df


class EigPlusLevelvar(UpdatedFeature):
    def __init__(
        self,
        source: UpdatedDataset,
        preproc: PreprocLevel,
        norm: bool,
        degree: int,
        trim: TrimMethod | None = None,
    ) -> None:
        self.degree: int
        super().__init__(
            source=source,
            preproc=preproc,
            norm=norm,
            degree=int(degree),
            trim=trim,
        )
        self.is_combined = True

    @property
    def data(self) -> DataFrame:
        eigs = self.dataset.eigs_df()
        y = eigs["y"].copy()
        eigs.drop(columns="y", inplace=True)  # remove target column
        self.feature_start_idxs.append(len(eigs.columns))
        lvars = levelvars(
            dataset=self.dataset, degree=self.degree, trim_method=self.trim, parallel=True
        )
        lvars.drop(columns="y", inplace=True)
        df = pd.concat([eigs, lvars], axis=1, ignore_index=True)
        df["y"] = y
        return df


class UnfoldedPlusLevelvar(UpdatedFeature):
    def __init__(
        self,
        source: UpdatedDataset,
        preproc: PreprocLevel,
        norm: bool,
        degree: int,
        trim: TrimMethod | None = None,
    ) -> None:
        self.degree: int
        super().__init__(
            source=source,
            preproc=preproc,
            norm=norm,
            degree=int(degree),
            trim=trim,
        )
        self.is_combined = True

    @property
    def data(self) -> DataFrame:
        unfs = self.dataset.unfolded_df(self.degree, self.trim)
        y = unfs["y"].copy()
        unfs.drop(columns="y", inplace=True)
        self.feature_start_idxs.append(len(unfs.columns))

        lvars = levelvars(
            dataset=self.dataset, degree=self.degree, trim_method=self.trim, parallel=True
        )
        lvars.drop(columns="y", inplace=True)
        df = pd.concat([unfs, lvars], axis=1, ignore_index=True)
        df["y"] = y
        return df


class UnfoldedPlusRigidity(UpdatedFeature):
    def __init__(
        self,
        source: UpdatedDataset,
        preproc: PreprocLevel,
        norm: bool,
        degree: int,
        trim: TrimMethod | None = None,
    ) -> None:
        self.degree: int
        super().__init__(
            source=source,
            preproc=preproc,
            norm=norm,
            degree=int(degree),
            trim=trim,
        )
        self.is_combined = True

    @property
    def data(self) -> DataFrame:
        unfs = self.dataset.unfolded_df(self.degree, self.trim)
        y = unfs["y"].copy()
        unfs.drop(columns="y", inplace=True)
        self.feature_start_idxs.append(len(unfs.columns))

        rigs = rigidities(
            dataset=self.dataset, degree=self.degree, trim_method=self.trim, parallel=True
        )
        rigs.drop(columns="y", inplace=True)
        df = pd.concat([unfs, rigs], axis=1, ignore_index=True)
        df["y"] = y
        return df


class EigPlusUnfoldedPlusRigidity(UpdatedFeature):
    def __init__(
        self,
        source: UpdatedDataset,
        preproc: PreprocLevel,
        norm: bool,
        degree: int,
        trim: TrimMethod | None = None,
    ) -> None:
        self.degree: int
        super().__init__(
            source=source,
            preproc=preproc,
            norm=norm,
            degree=int(degree),
            trim=trim,
        )
        self.is_combined = True

    @property
    def data(self) -> DataFrame:
        eigs = self.dataset.eigs_df()
        y = eigs["y"].copy()
        eigs.drop(columns="y", inplace=True)  # remove target column
        self.feature_start_idxs.append(len(eigs.columns))

        unfs = self.dataset.unfolded_df(self.degree, self.trim)
        unfs.drop(columns="y", inplace=True)
        self.feature_start_idxs.append(len(unfs.columns))

        rigs = rigidities(
            dataset=self.dataset, degree=self.degree, trim_method=self.trim, parallel=True
        )
        rigs.drop(columns="y", inplace=True)
        df = pd.concat([eigs, unfs, rigs], axis=1, ignore_index=True)
        df["y"] = y
        return df


class EigPlusUnfoldedPlusLevelvar(UpdatedFeature):
    def __init__(
        self,
        source: UpdatedDataset,
        preproc: PreprocLevel,
        norm: bool,
        degree: int,
        trim: TrimMethod | None = None,
    ) -> None:
        self.degree: int
        super().__init__(
            source=source,
            preproc=preproc,
            norm=norm,
            degree=int(degree),
            trim=trim,
        )
        self.is_combined = True

    @property
    def data(self) -> DataFrame:
        eigs = self.dataset.eigs_df()
        y = eigs["y"].copy()
        eigs.drop(columns="y", inplace=True)  # remove target column
        self.feature_start_idxs.append(len(eigs.columns))

        unfs = self.dataset.unfolded_df(self.degree, self.trim)
        unfs.drop(columns="y", inplace=True)
        self.feature_start_idxs.append(len(unfs.columns))

        lvars = levelvars(
            dataset=self.dataset, degree=self.degree, trim_method=self.trim, parallel=True
        )
        lvars.drop(columns="y", inplace=True)
        df = pd.concat([eigs, unfs, lvars], axis=1, ignore_index=True)
        df["y"] = y
        return df


class RigidityPlusLevelvar(UpdatedFeature):
    def __init__(
        self,
        source: UpdatedDataset,
        preproc: PreprocLevel,
        norm: bool,
        degree: int,
        trim: TrimMethod | None = None,
    ) -> None:
        self.degree: int
        super().__init__(
            source=source,
            preproc=preproc,
            norm=norm,
            degree=int(degree),
            trim=trim,
        )
        self.is_combined = True

    @property
    def data(self) -> DataFrame:
        rigs = rigidities(
            dataset=self.dataset, degree=self.degree, trim_method=self.trim, parallel=True
        )
        y = rigs["y"].copy()
        rigs.drop(columns="y", inplace=True)
        self.feature_start_idxs.append(len(rigs.columns))
        lvars = levelvars(
            dataset=self.dataset, degree=self.degree, trim_method=self.trim, parallel=True
        )
        lvars.drop(columns="y", inplace=True)
        df = pd.concat([rigs, lvars], axis=1, ignore_index=True)
        df["y"] = y
        return df


class UnfoldedPlusRigidityPlusLevelvar(UpdatedFeature):
    def __init__(
        self,
        source: UpdatedDataset,
        preproc: PreprocLevel,
        norm: bool,
        degree: int,
        trim: TrimMethod | None = None,
    ) -> None:
        self.degree: int
        super().__init__(
            source=source,
            preproc=preproc,
            norm=norm,
            degree=int(degree),
            trim=trim,
        )
        self.is_combined = True

    @property
    def data(self) -> DataFrame:
        unfs = self.dataset.unfolded_df(self.degree, self.trim)
        y = unfs["y"].copy()
        unfs.drop(columns="y", inplace=True)
        self.feature_start_idxs.append(len(unfs.columns))

        rigs = rigidities(
            dataset=self.dataset, degree=self.degree, trim_method=self.trim, parallel=True
        )
        y = rigs["y"].copy()
        rigs.drop(columns="y", inplace=True)
        self.feature_start_idxs.append(len(rigs.columns))

        lvars = levelvars(
            dataset=self.dataset, degree=self.degree, trim_method=self.trim, parallel=True
        )
        lvars.drop(columns="y", inplace=True)
        df = pd.concat([unfs, rigs, lvars], axis=1, ignore_index=True)
        df["y"] = y
        return df


class AllFeatures(UpdatedFeature):
    def __init__(
        self,
        source: UpdatedDataset,
        preproc: PreprocLevel,
        norm: bool,
        degree: int,
        trim: TrimMethod | None = None,
    ) -> None:
        self.degree: int
        super().__init__(
            source=source,
            preproc=preproc,
            norm=norm,
            degree=int(degree),
            trim=trim,
        )
        self.is_combined = True

    @property
    def data(self) -> DataFrame:
        eigs = self.dataset.eigs_df()
        y = eigs["y"].copy()
        eigs.drop(columns="y", inplace=True)  # remove target column
        self.feature_start_idxs.append(len(eigs.columns))

        unfs = self.dataset.unfolded_df(self.degree, self.trim)
        unfs.drop(columns="y", inplace=True)
        self.feature_start_idxs.append(len(unfs.columns))

        rigs = rigidities(
            dataset=self.dataset, degree=self.degree, trim_method=self.trim, parallel=True
        )
        rigs.drop(columns="y", inplace=True)
        self.feature_start_idxs.append(len(rigs.columns))

        lvars = levelvars(
            dataset=self.dataset, degree=self.degree, trim_method=self.trim, parallel=True
        )
        lvars.drop(columns="y", inplace=True)
        df = pd.concat([eigs, unfs, rigs, lvars], axis=1, ignore_index=True)
        df["y"] = y
        return df


FEATURE_OUTFILES: dict[Type[UpdatedFeature], Path] = {
    AllFeatures: PROJECT / "all_combined_predictions_updated.json",
    Eigenvalues: PROJECT / "eigenvalue_predictions_updated.json",
    EigsMinMax5: PROJECT / "eigs-minmax-5_predictions_updated.json",
    EigsMinMax10: PROJECT / "eigs-minmax-10_predictions_updated.json",
    EigsMinMax20: PROJECT / "eigs-minmax-20_predictions_updated.json",
    EigsMiddle10: PROJECT / "eigs-middle-10_predictions_updated.json",
    EigsMiddle20: PROJECT / "eigs-middle-20_predictions_updated.json",
    EigsMiddle40: PROJECT / "eigs-middle-40_predictions_updated.json",
    EigenvaluesSmoothed: PROJECT / "eig_smoothed_predictions_updated.json",
    EigenvaluesPlusEigenvaluesSmoothed: PROJECT
    / "eigenvalues+eig_smoothed_predictions_updated.json",
    EigenvaluesPlusSavGol: PROJECT / "eigenvalues+eig_savgol_predictions_updated.json",
    EigenvaluesSavGol: PROJECT / "eig_savgol_predictions_updated.json",
    Rigidities: PROJECT / "rigidity_predictions_updated.json",
    Levelvars: PROJECT / "levelvar_predictions_updated.json",
    EigPlusLevelvar: PROJECT / "eig+levelvar_predictions_updated.json",
    EigPlusRigidity: PROJECT / "eig+rigidity_predictions_updated.json",
    EigPlusUnfolded: PROJECT / "eig+unfolded_predictions_updated.json",
    EigPlusUnfoldedPlusLevelvar: PROJECT
    / "eig+unfolded+levelvar_predictions_updated.json",
    EigPlusUnfoldedPlusRigidity: PROJECT
    / "eig+unfolded+rigidity_predictions_updated.json",
    RigidityPlusLevelvar: PROJECT / "rigidity+levelvar_predictions_updated.json",
    Unfolded: PROJECT / "unfolded_predictions_updated.json",
    UnfoldedPlusLevelvar: PROJECT / "unfolded+levelvar_predictions_updated.json",
    UnfoldedPlusRigidity: PROJECT / "unfolded+rigidity_predictions_updated.json",
    UnfoldedPlusRigidityPlusLevelvar: PROJECT
    / "unfolded+rigidity+levelvar_predictions_updated.json",
}
