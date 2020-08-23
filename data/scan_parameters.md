# Gorgolewski / PSYCH

* Blood pressure and pulse:
    - systolic and diastolic blood pressure, as well as pulse, were measured for both arms.
* MRI:
    - 7 T whole-body MR scanner 
    - During the scan the participants’ pulse was monitored using a pulse oximeter
    - Breathing was measured using a pneumatic sensor.
        - Both breathing and pulse signals were sampled at 5,000 Hz
    - 3D MP2RAGE 29 sequence was used:
    - 3D-acquisition with field of view 224 × 224 × 168 mm 3 (H-F; A-P; R-L)
    - imaging matrix 320 × 320 × 240, 0.7 mm 3 isotropic
    - Time of Repetition (TR) = 5.0 s
    - Time of Echo (TE) = 2.45 ms
    - Time of Inversion (TI) 1/2 = 0.9 s/2.75 s
    - Flip Angle (FA) 1/2 = 5°/3°
    - Bandwidth (BW) = 250 Hz/Px
    - Partial Fourier 6/8, and
    - GRAPPA acceleration with iPAT factor of 2 (24 reference lines).
* Field map
    - For estimating B0 inhomogeneities, a 2D gradient echo sequence was used
    - axial orientation with field of view 192 × 192 mm 2 (R-L; A-P)
    - imaging matrix 64 × 64, 35 slices with
    - 3.0 mm thickness, 3.0 mm 3 isotropic voxel size
    - TR = 1.5 s
    - TE1/2 = 6.00 ms/7.02 ms (which gives delta TE = 1.02 ms)
    - FA = 72°
    - BW = 256 Hz/Px.
* Whole-brain rs-fMRI
    - 2D sequence, axial orientation
    - FOV = 192 × 192 mm 2 (R-L; A-P)
    - imaging matrix 128 × 128, 70 slices
    - 1.5 mm 3 isotropic voxel size
    - TR = 3.0 s
    - TE = 17 ms
    - FA = 70°
    - BW = 1,116 Hz/Px, Partial Fourier 6/8
    - GRAPPA acceleration with iPAT factor of 3 (36 reference lines)
    - 300 repetitions resulting in 15 min of scanning time. 

# Tétreault / OSTEO

**NO SLICE TIMING DATA AVAILABLE**

* MPRAGE type T1-anatomical brain images were acquired as described before [15]
    - 3T Siemens Trio whole-body scanner with echo-planar imaging (EPI)
    - voxel size 1 × 1 × 1 mm
    - TR = 2,500 ms
    - TE = 3.36 ms
    - flip angle = 9°;
    - matrix resolution = 256 × 256; slices = 160
    - FOV = 256 mm
* rs-fMRI images were acquired on the same day and scanner with the following parameters:
    - multi-slice T2  -weighted echo-planar images
    - TR = 2.5 s
    - TE = 30 ms
    - flip angle = 90°,
    - 64 x 64 x 40@3mm 
    - 300 volumes

# DuPre / ECHO

Pulse and respiration were monitored continuously during scanning using an integrated pulse oximeter and respiratory belt.
Due to a software upgrade, physiological sampling occurred at 50 Hz for 16 participants and 40 Hz for
15 participants; therefore, physiological sampling rate is provided for each participant as detailed in
Data Records.

* Resting fMRI
    - multi-echo echo planar imaging (ME-EPI) sequence with online reconstruction
    - TR = 3000 ms
    - TE = 13.7, 30, and 47 ms
    - FA = 83°
    - matrix size = 72 × 72; 46 axial slices
    - 3.0 mm isotropic voxels
    - FOV = 210 mm
    - **slice order = inferior-superior interleaved** (data in jsons)
    - Resting-state functional scans were acquired with 2.5x acceleration with sensitivity encoding.

* Task fMRI
    - ME-EPI sequence with online reconstruction
    - TR = 2000 ms
    - TE = 13, 27, and 43 ms;
    - FA = 77°
    - matrix size = 64 × 64; 33 axial slices
    - slice thickness 3.8 mm
    - FOV = 240
    - **slice order = inferior-superior interleaved** (data in jsons)
    - Functional scans were acquired with 2x acceleration with sensitivity encoding

# Madhyastha / PARK

* fMRI data were acquired using a Philips 3T with a 32-channel SENSE, whole-brain axial echo-planar images
    - 43 **sequential ascending slices** (data in json)
    - 3 mL isotropic voxels,
    - FOV = 240 · 240 · 129
    - TR = 2400 ms,
    - TE = 25 ms
    - FA = 79°
    - SENSE acceleration factor = 2
    - were collected parallel to the anterior-commissure– posterior commissure (AC-PC) line for all functional runs
    - run duration was 300 volumes (12 min) for the resting-state run and 149 volumes (5.96 min) for each task run


# Schapiro / LEARN

* fMRI data acquisition using a 3T Siemens Skyra scanner with a volume head coil.
    - T2\*-weighted gradient-echo EPI sequence
    - matrix = 64 × 64, 36 oblique axial slices
    - 3 × 3 mm inplane, 3 mm thickness
    - TE = 30 ms
    - TR = 2000 ms
    - FA = 71°
    - each run contained 195 volumes.
    - An in-plane **magnetic field map** image was also acquired for EPI undistortion.
    - **slicetime data in json**

We collected two anatomical runs for registration across subjects to standard space: a coplanar T1-weighted FLASH
sequence and a high-resolution 3D T1-weighted MPRAGE sequence.
