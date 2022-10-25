# Potential Preprocessing Steps:

* remove first N (e.g. 4) volumes to allow field to stabilize
* MCFLIRT motion correction
* field map unwarping
* slice-timing correction
* BET
* spatial smoothing?
* temporal filtering (high pass?)


# Definitely Do:
* FSL MCFLIRT motion correction
* FSL slicetime correction


# Commands (FSL)

* MCFLIRT (Do first)
  ```bash
  mcflirt -in <infile> -out <outfile> -verbose 0 -stages 3 -meanvol
  ```
  - took only 2 minutes total for high-resolution dataset
  - seems to use only a single core, and thus trivial to parallelize with gnu-parallel
  - there are about 1600 scans..., so 54 hours / 8 == Overnight with 8 cores
  ```bash
  mcflirt -in <infile> -out <outfile> -verbose 1 -stages 4 -meanvol -report
  ```
  - took about 11 minutes total for high-resolution dataset
  - with about 1600 scans, that's 293 hours / 8 == 1.5 days with 8 cores
  - mild quality improvement

* slice-timing correction (Do second)
    ```bash
    slicetimer -i <infile> -o <outfile> --repeat=<TR> --direction=<json Encoding Direction i=>x=>1, j=>k=>2, k=>z=>1 (default)> --tcustom=<json slicetimes>
    ```
    - took 2.9 minutes on high-res data, BUT used a bunch of cores it seems
    - with about 1600 scans, that's 
  - "Slice timing correction works by using (Hanning-windowed) sinc
    interpolation to shift each time-series by an appropriate fraction of a TR
    relative to the middle of the TR period. 
  - If slices were acquired from the bottom of the brain to the top select
    *Regular up*. If slices were acquired from the top of the brain to the
    bottom select Regular down. If the slices were acquired with interleaved
    order (0, 2, 4 ... 1, 3, 5 ...) then choose the Interleaved option.
  - If slices were not acquired in regular order you will need to use a slice
    order file or a slice timings file. If a slice order file is to be used,
    create a text file with:
    - a single number on each line
    - the first line states which slice was acquired first, the second line
      states which slice was acquired second, etc.
    - The first slice is numbered 1 not 0.
    - If a slice timings file is to be used, put one value (ie for each slice)
      on each line of a text file.
    - The units are in TRs, with 0 corresponding to no shift. Therefore a
      sensible range of values will be between -0.5 and 0.5. 

* BET (do last)

* Field Map / Inhomogeneity Corrections
  ```bash
  fugue 
  ```

# Notes

* Madhyasatha "We did not perform bandpass filtering to avoid artifi-
cially inflating correlations or inducing structure that was not
actually present in the data, and because resting-state net-
works exhibit different levels of phase synchrony at differ-
ent frequencies (Handwerker et al., 2012; Niazy et al.,
2011)."

* do not remove global BOLD signal (since this may differ across states and is potentially useful information)
