# Dataset Descriptions

This file contains full citations and descriptions for the 5 DPM datasets used in the Rocco evaluation study and the example description used in the enhancement demonstration.

## Evaluated Descriptions

### Evaluated Description 1:

**Dataset Title:** Multimineral Model

**Description:**

A North Sea sandstone with 23% porosity and 640 mD permeability was cleaned with solvents, dried at 60°C, and imaged using micro-CT at a 2.3 μm resolution over 22 hours. It was then cut, resin-impregnated, polished, and carbon-coated for 2D mineral mapping using SEM-EDS with QEMSCAN at a 2.0 μm resolution. The 2D mineral map was registered to the 3D micro-CT image, and minerals were segmented into seven groups based on X-ray intensity, though some minerals with similar attenuation could not be fully distinguished. The data is uploaded as 4 netCDF blocks that combine into a full 3D data set of 1,000^3. Full details are provided in the netCDF header files. The segmented domain is associated with LBPM (https://github.com/OPM) simulations results available at https://zenodo.org/records/13836047

**Citation:**

 Armstrong, R., Mostaghimi, P., McClure, J. (2025, February 15). Multimineral Model [Dataset]. Digital Porous Media Portal. https://www.doi.org/10.17612/NXDJ-0Y17

---

### Evaluated Description 2:

**Dataset Title:** 16 Brazilian pre-salt carbonates: multi-resolution micro-CT images

**Description:**

The database consists of a total of 16 samples of carbonate rocks from the Brazilian Pre-Salt. The samples come in two distinct sizes: those with a 2.54 centimeters in diameter correspond to sidewall samples, while those with a 3.81 centimeters in diameter are represented by plugs. In this database, each sample has two different resolutions: high and low. For the sidewall samples, the high resolution is 6 micrometers (µm) and the low resolution is 48 µm. For the plug samples, the high resolution is 8 µm, while the low resolution is 64 µm. The database comprises grayscale images that have been processed, filtered to reduce artifacts, registered, and cropped. Additionally, it includes segmented files obtained through the Otsu and Watershed methods, enabling a detailed and accurate analysis of sample characteristics. Thus, the dataset consists of a total of 32 grayscale image files. With the inclusion of segmented images, the dataset expands to a total of 64 files, where each original image has a corresponding segmented version into pore and matrix, totaling 32 additional files. All the rocks were acquired using the Ge X-ray emission microtomography equipment, model VTomex M. These microtomography scans were acquired at low resolution (48 and 64 µm) with the following the parameters: energy (150 kV), current (250 µA), average (3), skip (1) and filter (0,15 mmCu). And high resolution (6 and 8 µm) with the following the parameters: energy (140 kV), current (140 µA), average (6), skip (1) and filter (0,15 mmCu). This dataset offers a wide range of petrophysical, geological, and artificial intelligence applications. Among the petrophysical applications, highlights include porosity and permeability analysis, characterization of rock connectivity and heterogeneity, and simulations of more advanced properties on high-resolution images, such as acoustics, wettability, and relative permeability. For geology, high-resolution images are particularly useful for investigating the diagenetic processes that impact the rock over time, such as cementation and mineral dissolution. The images allow for a three-dimensional visualization of the texture and internal structure of rocks, facilitating the analysis of mineral distribution, fractures, and heterogeneities. On the other hand, low-resolution images serve the purpose of providing information about the overall distribution of these minerals and components over a broader area. Artificial intelligence further enhances the utility of the dataset through various applications. Image classification algorithms can automatically categorize rock samples based on their resolution and segmentation into pore and matrix files, streamlining dataset organization and analysis.

**Citation:**

Vidal, A., Menezes dos Anjos, C., Medeiros, L., Surmas, R., Neta, A., Evsukoff, A., Vargas, J. (2024, May 20). 16 Brazilian pre-salt carbonates: multi-resolution micro-CT images [Dataset]. Digital Porous Media Portal. https://www.doi.org/10.17612/xr50-s717

---

### Evaluated Description 3:

**Dataset Title:** Data for pore-scale imaging of brine-nitrogen co-injection in cm-scale Bentheimer sandstone sample

**Description:**

The dataset contains micro-CT images for steady-state co-injection experiments performed on a cylindrical Bentheimer sandstone with one-inch diameter. Scans were obtained using the “High Energy micro-CT Optimized for Research” scanner (HECTOR) at the center for X-ray tomography at Ghent University (UGCT). The experiments were conducted with nitrogen as the non-wetting phase and brine (25 wt% potassium iodine) as the wetting phase. The confining pressure and the back pressure were set to 4000 kPa and 2000 kPa respectively.

**Citation:**

Wang, S., Bultreys, T., Spurin, C. (2023, April 22). Data for pore-scale imaging of brine-nitrogen co-injection in cm-scale Bentheimer sandstone sample [Dataset]. Digital Porous Media Portal. https://www.doi.org/10.17612/PHHX-4K68

---

### Evaluated Description 4:

**Dataset Title:** Fractures with variable roughness and wettability

**Description:**

Four fractures with periodic boundaries exhibiting increasing fractal dimensions (1.5, 1.75, 2.0, 2.25) and 3 different wettability patterns (A, B, C) are presented. The fractures were created with SynFrac and mirrored to obtain a longer domain. Each fracture is 128 x 256 and periodic in the y-direction. Each wettability pattern has a similar histogram, and is based on the supplemental information Figure Si.7c from Gerke et al. (2015). These fractures were used for CO2 displacement lattice Boltzmann simulations.

**Citation:**

Guiltinan, E., E. Santos, J., Kang, Q., Cardenas, B., Espinoza, D. (2020, September 20). Fractures with variable roughness and wettability [Dataset]. Digital Porous Media Portal. https://www.doi.org/10.17612/p522-cc94

---

### Evaluated Description 5:

**Dataset Title:** Dataset of 3D fluid phase distribution from drainage simulations (in micromodel and real rock geometry) examining inertial effects

**Description:**

The dataset contains fluid phase distribution obtained from high-resolution drainage simulations in both a heterogenous micromodel and Bentheimer sandstone. The purpose is to investigate the influence of inertial effects on scCO2-brine displacement in complex geometries, where scCO2 is much less viscous than brine or oil meaning that the inertial effects may not be negligible. The direct numerical simulations in this work employ the continuum-surface-force based color-gradient multiple-relaxation-time lattice Boltzmann model combined with a geometrical wetting model. The inertial effects are investigated by varying the Ohnesorge number (Oh) while keeping other conditions the same. A manuscript from this work has been submitted to a journal and is currently in revision. 1). Dataset from Bentheirmer sandstone simulations The geometry of the Bentheimer sandstone is from Herring et al., 2016, where the original data size is 1920\*1920\*1200 with a voxel resolution of 3.182 micron. A subdomain is cropped from the original data (porosity: 0.19), resulting in a simulation grid of 720\*720\*900 with a grid resolution of 3.182 micron. The simulation conditions are as follows: case 1: Ca=2.0e-6, Oh=2.23e-3 case 2: Ca=2.0e-6, Oh=5.57e-3 case 3: Ca=2.0e-6, Oh=11.14e-3 contact angle: 30 degree viscosity ratio: 37.7 (nonwetting phase is less viscous) density ratio: 1 Fluid phase distribution data (location and order parameter value) at the breakthrough time from the above three cases are provided. 2). Dataset from micromodel simulations The geometry of the micromodel is from Li et al., 2017, where the original 2D scan is 11275\*8460 with a voxel resolution of 0.630 micron. The simulation grid, 1953\*13\*2640 is constructed from resized 2D scan image (porosity: 0.49), with a grid resolution of 2.727 micron. The simulation conditions are as follows: case 1: Ca=2.6e-5, Oh=2.177e-3 case 2: Ca=2.6e-5, Oh=11.62e-3 contact angle: 10 degree viscosity ratio: 13.2 (nonwetting phase is less viscous) density ratio: 1 Fluid phase distribution data (2D gap-averaged saturation and location on the 2D plane) after one pore-volume injection from the above two cases are provided. The geometry data (solid points location) used in all simulations can be derived from the fluid phase distribution data. 1. Herring, Anna L., Linnéa Andersson, and Dorthe Wildenschild. "Enhancing residual trapping of supercritical CO2 via cyclic injections." Geophysical Research Letters 43, no. 18 (2016): 9677-9685. 2. Li, Yaofa, Farzan Kazemifar, Gianluca Blois, and Kenneth T. Christensen. "Micro‐PIV measurements of multiphase flow of water and liquid CO2 in 2‐D heterogeneous porous micromodels." Water Resources Research 53, no. 7 (2017): 6178-6196.


**Citation:**

Chen, Y., Kang, Q., Valocchi, A., Viswanathan, H. (2019, September 30). Dataset of 3D fluid phase distribution from drainage simulations (in micromodel and real rock geometry) examining inertial effects [Dataset]. Digital Porous Media Portal. https://www.doi.org/10.17612/1fhh-q252

---

For detailed scoring results by evaluator and rubric item, see `evaluation_results.xlsx`.


## Enhanced Description
### Enhanced Description 1:

**Dataset Title:** Niobrara formation fracture

**Description:**

Microtomography image of a fracture from Niobrara formation, CO, USA (tight carbonate).

**Citation:**

Prodanovic, M., Landry, C., Tokan-Lawal, A., Eichhubl, P. (2016, April 18). Niobrara formation fracture [Dataset]. Digital Porous Media Portal. https://www.doi.org/10.17612/P7SG6Z
