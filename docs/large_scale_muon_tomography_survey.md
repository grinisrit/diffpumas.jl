# **Overview of Large-Scale Muon Tomography Methods and Key Articles**

## 1. Introduction

Large-scale muon tomography (muography) leverages the natural flux of cosmic-ray muons to non-invasively image the internal structure of massive or shielded objects, including volcanoes, nuclear reactors, archaeological sites, and industrial infrastructure. The field has rapidly evolved, with advances in detector technology, reconstruction algorithms, and application domains. Muography methods are broadly categorized into **scattering-based**, **transmission (absorption)-based**, and **hybrid** approaches, each with unique strengths for different targets and resolutions  (Luo et al., 2025; Bonechi et al., 2019; Yashin et al., 2021; Marteau et al., 2012). Recent years have seen the integration of machine learning and deep learning to further enhance image quality and reduce acquisition times  (Lim & Qiu, 2023; Lefevre et al., 2024; Ruta & Ruta, 2023; Pezzotti et al., 2025). This overview synthesizes foundational principles, methodological diversity, algorithmic advances, and comparative evaluations from recent literature.


**Figure 1:** Consensus meter: Effectiveness of large-scale muon tomography for dense/complex object imaging.

## 2. Methods

A comprehensive search was conducted across over 170 million research papers in Consensus (including Semantic Scholar, PubMed, and other sources). The search identified 29341298 potentially relevant papers; after multi-phase filtering and relevance screening, 50 were included in this review. The search strategy targeted foundational overviews, methodological diversity (scattering/absorption/hybrid/machine learning), application domains (geoscience, nuclear, civil engineering), algorithmic advances (reconstruction/resolution), comparative evaluations (metrics/performance), and critical perspectives.

<search_strategy_diagram identifiedPapers="29341298" screenedPapers="229" eligiblePapers="118" includedPapers="50" />
**Figure 2:** Search strategy: From initial identification to final inclusion of key papers.
Six unique search groups were used to ensure broad coverage of technical methods and applications.

## 3. Results

### 3.1 Foundational Principles & Method Categories

- **Scattering-Based Muon Tomography:**  
  Utilizes multiple Coulomb scattering (MCS) of muons as they traverse matter; sensitive to atomic number (Z) and density variations. High-Z materials cause greater angular deviations  (Luo et al., 2025; Borozdin & Vozdolska, 2026; Liang et al., 2025).  
  - *Key Algorithms:* Point-of-Closest-Approach (PoCA), statistical/Bayesian reconstruction (e.g., Expectation-Maximization), momentum-informed extensions (mPoCA/MMST)  (Ughade & Chatzidakis, 2025; Bae et al., 2024).
- **Transmission/Absorption Muography:**  
  Measures attenuation of muon flux through an object; denser regions absorb more muons. Analogous to X-ray CT but with much greater penetration depth  (Luo et al., 2025; Balázs et al., 2025; Marteau et al., 2012).
- **Hybrid Approaches:**  
  Combine scattering and transmission data for improved discrimination across a range of Z/materials  (Luo et al., 2022; Qin et al., 2025).

### 3.2 Algorithmic Advances

- **Statistical & Bayesian Methods:**  
  EM/Bayesian algorithms model the full scattering process probabilistically, improving resolution and reducing artifacts compared to geometric PoCA  (Sattler et al., 2025; Jonkmans et al., 2012).
- **Algebraic Reconstruction Techniques:**  
  SART/SIRT/ART iteratively solve for voxel densities using all available projections; robust for sparse or incomplete data  (Procureur, 2021; Hartling et al., 2021; Dunn et al., 2025).
- **Momentum-Informed Imaging:**  
  Incorporates measured or estimated muon momentum into reconstruction algorithms (e.g., mPoCA/MMST), significantly enhancing material discrimination and spatial resolution  (Csatlós et al., 2024; Bae et al., 2024).
- **Machine Learning & Deep Learning:**  
  Neural networks are used for denoising images, predicting trajectories, or direct voxel mapping from raw detector data—outperforming traditional methods in some benchmarks  (Lim & Qiu, 2023; Lefevre et al., 2024; Ruta & Ruta, 2023; Pezzotti et al., 2025).

### 3.3 Application Domains

- **Geoscience & Volcanology:** Imaging volcano interiors, fault zones, mountains  (Balázs et al., 2025; Lechmann et al., 2021; Bajou et al., 2023; Zhang et al., 2022).
- **Nuclear Security & Safeguards:** Monitoring spent fuel casks, detecting missing fuel assemblies or illicit nuclear materials  (Park et al., 2021; Valencia et al., 2025; Erlandson et al., 2021; Bae et al., 2024).
- **Civil Engineering & Archaeology:** Imaging tunnels, ancient walls/structures; structural diagnostics in concrete  (Niederleithinger et al., 2025; Liu et al., 2023; Thompson & Steer, 2025; Pezzotti et al., 2025).
- **Cultural Heritage:** Non-invasive analysis of statues/artifacts where X-rays are insufficient  (Giammanco et al., 2024; Lagrange et al., 2025).

### 3.4 Comparative Evaluations & Metrics

- *Metrics Used:* Signal-to-noise ratio (SNR), contrast-to-noise ratio (CNR), mean square error (MSE), structural similarity index measure (SSIM), peak SNR (PSNR), detection power/statistical significance for anomaly detection.
- *Findings:* Advanced statistical/Bayesian/momentum-informed methods consistently outperform PoCA in both spatial resolution and detection efficiency—often requiring an order of magnitude fewer muons for comparable results  (Ughade & Chatzidakis, 2025; Bae et al., 2024). Machine learning approaches show promise but require further validation on experimental datasets  (Lim & Qiu, 2023).

#### Results Timeline
- **2009**
  - 1 paper:  (Pesente et al., 2009)- **2018**
  - 1 paper:  (Liu et al., 2018)- **2021**
  - 3 papers:  (Yashin et al., 2021; Procureur, 2021; Park et al., 2021)- **2022**
  - 1 paper:  (Bae & Chatzidakis, 2022)- **2023**
  - 3 papers:  (Lim & Qiu, 2023; Liu et al., 2023; Ruta & Ruta, 2023)- **2024**
  - 4 papers:  (Yu et al., 2024; Lefevre et al., 2024; Giammanco et al., 2024; Csatlós et al., 2024)- **2025**
  - 7 papers:  (Luo et al., 2025; Niederleithinger et al., 2025; Balázs et al., 2025; Sehgal & Jha, 2025; Anonymous, 2025; Valencia et al., 2025; Georgadze, 2025)**Figure 3:** Timeline showing evolution from foundational absorption/scattering methods to hybrid/machine learning approaches in large-scale muon tomography. Larger markers indicate more citations.

#### Top Contributors
<top_contributors authors='@@json@@:[{"name": "S. Chatzidakis", "citations": "13,18,29,34"}, {"name": "J. Bae", "citations": "18,29,46"}, {"name": "S. Procureur", "citations": "3,7"}]' journals='@@json@@:[{"name": "Journal of Applied Physics", "citations": "2,9,12,16"}, {"name": "Scientific Reports", "citations": "5,46"}, {"name": "Nuclear Instruments & Methods in Physics Research Section A-accelerators Spectrometers Detectors and Associated Equipment", "citations": "3,8"}]' />
**Figure 4:** Authors & journals that appeared most frequently in the included papers.

## 4. Discussion

The reviewed literature demonstrates that large-scale muon tomography is a mature technique with proven effectiveness across diverse domains—from geoscience to nuclear security to cultural heritage preservation  (Luo et al., 2025; Bonechi et al., 2019). Scattering-based methods dominate high-Z material detection due to their sensitivity to atomic number via MCS; transmission-based approaches excel at mapping bulk density distributions in geological or archaeological contexts  (Luo et al., 2025; Balázs et al., 2025). Hybrid models further improve versatility by combining both signals  (Luo et al., 2022). Recent advances include momentum-informed algorithms that leverage per-muon kinematics for enhanced resolution—especially important when distinguishing between similar materials or detecting small anomalies within dense backgrounds  (Csatlós et al., 2024; Bae et al., 2024). Machine learning is emerging as a powerful tool for denoising images and accelerating reconstructions but still faces challenges related to generalizability and interpretability  (Lim & Qiu, 2023; Lefevre et al., 2024).

**Claims and Evidence Table**

| Claim                                                                 | Evidence Strength | Reasoning                                                                                  | Papers                |
|-----------------------------------------------------------------------|-------------------|-------------------------------------------------------------------------------------------|-----------------------|
| Advanced statistical/Bayesian/momentum-informed methods outperform PoCA | Evidence strength: Strong (9/10) | Multiple studies show higher SNR/CNR/detection power at lower event counts                |  (Ughade & Chatzidakis, 2025), (Bae et al., 2024), (Sattler et al., 2025), (Bae et al., 2024)|
| Hybrid models enable simultaneous imaging across low–high Z            | Evidence strength: Strong (8/10) | Simulations confirm improved PSNR/image quality over single-mode approaches               |  (Luo et al., 2022), (Qin et al., 2025)|
| Machine learning can surpass traditional reconstructions               | Evidence strength: Moderate (7/10) | Deep learning models achieve higher PSNR/robustness on synthetic datasets                 |  (Lim & Qiu, 2023), (Lefevre et al., 2024), (Ruta & Ruta, 2023), (Pezzotti et al., 2025)|
| Transmission-based muography excels at bulk density mapping           | Evidence strength: Strong (8/10) | Demonstrated sharper contours/higher accuracy than other geophysical techniques           |  (Balázs et al., 2025), (Lechmann et al., 2021)|
| Scattering-based methods best for high-Z/special nuclear material     | Evidence strength: Strong (9/10) | High sensitivity due to Z-dependence of MCS; proven in nuclear/cargo scenarios            |  (Park et al., 2021), (Valencia et al., 2025), (Erlandson et al., 2021), (Liang et al., 2025)|
| Data acquisition time remains a limiting factor                       | Evidence strength: Moderate (6/10) | Low natural muon flux necessitates long exposures for high-resolution imaging             |  (Niederleithinger et al., 2025), (Pezzotti et al., 2025)|

**Figure 5:** Key claims and support evidence identified in these papers.

## 5. Conclusion

Large-scale muon tomography is a versatile imaging modality with demonstrated success across scientific disciplines. Scattering-, transmission-, hybrid-, momentum-informed-, and machine-learning-enhanced algorithms each offer unique advantages depending on the target application—whether it be high-Z material detection or bulk density mapping.

### Research Gaps

Despite progress:
- Real-time imaging remains challenging due to low cosmic ray flux.
- Standardized benchmarks/datasets are only now emerging.
- Further validation is needed for deep learning models on real-world data.
- Portable/mobile detector systems are still under development.

**Research Gaps Matrix**

| Topic/Outcome         | Nuclear Security      | Geoscience         | Civil Engineering   | Cultural Heritage   |
|----------------------|----------------------|--------------------|--------------------|--------------------|
| Scattering Imaging   | **12**   | **6**    | **4**    | **2**    |
| Transmission Imaging | **6**    | **8**    | **2**    | **1**    |
| Hybrid/Momentum      | **4**    | **1**    | **GAP**    | **GAP**    |
| Machine Learning     | **3**    | **1**    | **1**    | **GAP**    |

**Figure 6:** Matrix showing research coverage by method type versus application domain.

### Open Research Questions

Future work should focus on:
- Improving real-time capabilities via hardware/software co-design.
- Establishing standardized open datasets/benchmarks.
- Validating machine learning models on experimental data.
- Developing portable detectors for field deployment.

| Question                                                                                      | Why                                                                                                   |
|-----------------------------------------------------------------------------------------------|-------------------------------------------------------------------------------------------------------|
| **How can real-time large-scale muon tomography be achieved?**        | Real-time imaging would enable broader adoption in security/infrastructure monitoring but requires overcoming flux/data bottlenecks.          |
| **What are the limits of deep learning models on real-world muography data?**      | Most ML results are on synthetic data; robust validation is needed before clinical/security deployment.                                      |
| **How can portable/mobile detectors be optimized for diverse environments?**        | Field applications demand lightweight systems without sacrificing resolution or efficiency.                                                  |

**Figure 7:** Open questions guiding future research directions.

**In summary:** Large-scale muon tomography is now established as a powerful tool across science and industry—with ongoing innovation focused on speed, resolution enhancement, algorithmic robustness, and practical deployment challenges.

---

**Key references from this review include**:  
 (Luo et al., 2025): Luo et al., Journal of Applied Physics (2025) – comprehensive review of algorithms/principles  
 (Ughade & Chatzidakis, 2025): Ughade & Chatzidakis – μTRec Bayesian trajectory algorithm  
 (Bae et al., 2024): Bae et al., Annals of Nuclear Energy – momentum-informed MST  
 (Luo et al., 2022): Luo et al., Nuclear Science & Techniques – hybrid model simulation study  
 (Lim & Qiu, 2023): Lim & Qiu – μ-Net deep learning model  
 (Sattler et al., 2025): Sattler et al., Journal of Applied Physics – GPU EM solver framework  
 (Pezzotti et al., 2025): Pezzotti et al., Journal of Instrumentation – deep learning enhancement for civil engineering diagnostics
 
_These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at https://consensus.app. © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent._
 
## References
 
A. (2025). Experimental study of multiple-orientation muon tomography with image optimization in sparse data environments. *Physical Review Applied*. https://doi.org/10.1103/tl8q-lw81
 
Bae, J., & Chatzidakis, S. (2022). Momentum-Dependent Cosmic Ray Muon Computed Tomography Using a Fieldable Muon Spectrometer. *Energies*. https://doi.org/10.3390/en15072666
 
Bae, J., Montgomery, R., & Chatzidakis, S. (2024). Nuclear material accountancy using momentum-informed muon scattering tomography. *Annals of Nuclear Energy*. https://doi.org/10.1016/j.anucene.2023.110240
 
Bae, J., Montgomery, R., & Chatzidakis, S. (2024). Momentum informed muon scattering tomography for monitoring spent nuclear fuels in dry storage cask. *Scientific Reports, 14*. https://doi.org/10.1038/s41598-024-57105-y
 
Bajou, R., Rosas-Carbajal, M., Tonazzo, A., & Marteau, J. (2023). High-resolution structural imaging of volcanoes using improved muon tracking. *Geophysical Journal International*. https://doi.org/10.1093/gji/ggad269
 
Balázs, L., Hamar, G., Csicsek, Á., Surányi, G., & Varga, D. (2025). Detection of fractured zones, faults, and cavities by high resolution muon tomography in the Buda Hills. *Scientific Reports, 15*. https://doi.org/10.1038/s41598-025-02510-0
 
Bonechi, L., D’Alessandro, R., & Giammanco, A. (2019). Atmospheric muons as an imaging tool. *Reviews in Physics*. https://doi.org/10.1016/j.revip.2020.100038
 
Borozdin, K., & Vozdolska, R. (2026). Imaging Techniques in Muon Scattering Tomography. *Proceedings of Fifth MODE Workshop on Differentiable Programming for Experiment Design — PoS(MODE2025)*. https://doi.org/10.22323/1.491.0009
 
Csatlós, B., Hamar, G., & Varga, D. (2024). Experimental Momentum-binning for Muon Scattering Tomography. *Journal of Advanced Instrumentation in Science*. https://doi.org/10.31526/jais.2024.496
 
Dunn, A., Erlandson, A., Godin, D., Harrisson, G., Hartling, K., & Pérez-Loureiro, D. (2025). Sparse-view muon computed tomography of an operating research reactor. *Journal of Applied Physics*. https://doi.org/10.1063/5.0287881
 
Erlandson, A., Anghel, V., Godin, D., Jewett, C., & Thompson, M. (2021). An analysis of pressurized heavy water reactor fuel for nuclear safeguards applications using muon scattering tomography. *Journal of Instrumentation, 16*, P02024 - P02024. https://doi.org/10.1088/1748-0221/16/02/p02024
 
Georgadze, A. (2025). Design and Simulation of a Muon Detector Using Wavelength-Shifting Fiber Readouts for Border Security. *Instruments*. https://doi.org/10.3390/instruments9010001
 
Giammanco, A., Moussawi, M. A., Boone, M., De Kock, T., De Roy, J., Huysmans, S., Kumar, V., Lagrangev, M., & Tytgat, M. (2024). Toward using cosmic rays to image cultural heritage objects. *iScience, 28*. https://doi.org/10.1016/j.isci.2025.112094
 
Hartling, K., Mahoney, F., Rand, E., Sariya, T., & Valente, A. (2021). A comparison of algebraic reconstruction techniques for a single-detector muon computed tomography system. *Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment*. https://doi.org/10.1016/j.nima.2020.164834
 
Jonkmans, G., Anghel, V. N. P., Jewett, C., & Thompson, M. (2012). Nuclear waste imaging and spent fuel verification by muon tomography. *Annals of Nuclear Energy, 53*, 267-273. https://doi.org/10.1016/j.anucene.2012.09.011
 
Lagrange, M., Moussawi, M. A., Boone, M., De Kock, T., De Roy, J., Giammanco, A., Huysmans, S., & Tytgat, M. (2025). Muon tomography for cultural heritage objects. *Journal of Applied Physics*. https://doi.org/10.1063/5.0273541
 
Lechmann, A., Mair, D., Ariga, A., Ariga, T., Ereditato, A., Nishiyama, R., Pistillo, C., Scampoli, P., Schlunegger, F., & Vladymyrov, M. (2021). Muon tomography in geoscientific research – A guide to best practice. *Earth-Science Reviews*. https://doi.org/10.1016/j.earscirev.2021.103842
 
Lefevre, B., Attié, D., Bajou, R., & Gomez, H. (2024). 2D and 3D analysis improvements with machine learning for muography applications. *Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment*. https://doi.org/10.1016/j.nima.2024.169755
 
Liang, Z., Tang, Z., Li, X., Liu, B., Li, C., He, J., Jiang, K., Wang, Y., Tian, Y., Zhang, Y., & Wang, Z. (2025). A muon scattering tomography system based on high spatial resolution scintillating detector.
 
Lim, L. X. J., & Qiu, Z. (2023). μ-Net: ConvNext-Based U-Nets for Cosmic Muon Tomography. *ArXiv, abs/2312.17265*. https://doi.org/10.48550/arxiv.2312.17265
 
Liu, G., Luo, X., Tian, H., Yao, K., Niu, F., Jin, L., Gao, J., Rong, J., Fu, Z., Kang, Y., Fu, Y., Wu, C., Gao, H., Gong, J., Zhang, W., Luo, X., Liu, C., Tian, X., Yu, M., . . . Liu, Z. (2023). High-precision muography in archaeogeophysics: A case study on Xi’an defensive walls. *Journal of Applied Physics*. https://doi.org/10.1063/5.0123337
 
Liu, Z., Chatzidakis, S., Scaglione, J., Liao, C., Yang, H., & Hayward, J. (2018). Muon Tracing and Image Reconstruction Algorithms for Cosmic Ray Muon Computed Tomography. *IEEE Transactions on Image Processing, 28*, 426-435. https://doi.org/10.1109/tip.2018.2869667
 
Luo, S., Feng, C., Zeng, G., Feng, S., Shen, M., Huang, X., Wang, L., Zhao, S., Du, X., Feng, S., Xiao, M., Liu, Z., & Wang, X. (2025). Image reconstruction techniques in muography: A review of algorithms and physical principles. *Journal of Applied Physics*. https://doi.org/10.1063/5.0273072
 
Luo, S., Huang, Y.-H., Ji, X.-T., He, L., Xiao, W.-C., Luo, F., Feng, S., Xiao, M., & Wang, X.-D. (2022). Hybrid model for muon tomography and quantitative analysis of image quality. *Nuclear Science and Techniques, 33*. https://doi.org/10.1007/s41365-022-01070-6
 
Marteau, J., Gibert, D., Lesparre, N., Nicollin, F., Noli, P., & Giacoppo, F. (2012). Muons tomography applied to geosciences and volcanology. *Nuclear Instruments & Methods in Physics Research Section A-accelerators Spectrometers Detectors and Associated Equipment, 695*, 23-28. https://doi.org/10.1016/j.nima.2011.11.061
 
Niederleithinger, E., Sein, S., Kervalisvili, A., Wöstmann, J., & Helm, M. (2025). Muon tomography - ready for practical application?. *e-Journal of Nondestructive Testing*. https://doi.org/10.58286/31696
 
Park, C., Baek, M. K., Kang, I., Lee, S., Chung, H., & Chung, Y. (2021). Design and characterization of a Muon tomography system for spent nuclear fuel monitoring. *Nuclear Engineering and Technology*. https://doi.org/10.1016/j.net.2021.08.029
 
Pesente, S., Vanini, S., Benettoni, M., Bonomi, G., Calvini, P., Checchia, P., Conti, E., Gonella, F., Nebbia, G., Squarcia, S., Viesti, G., Zenoni, A., & Zumerle, G. (2009). First results on material identification and imaging with a large-volume muon tomography prototype. *2009 1st International Conference on Advancements in Nuclear Instrumentation, Measurement Methods and their Applications*, 1-4. https://doi.org/10.1016/j.nima.2009.03.017
 
Pezzotti, L., Cifarelli, D., Corradetti, D., Costa, J. P., Gabrielli, G., Galante, L., Gallerati, A., Gnesi, I., Jouve, A., & Marrani, A. (2025). A new method for structural diagnostics with muon tomography and deep learning. *Journal of Instrumentation, 20*. https://doi.org/10.1088/1748-0221/20/06/p06034
 
Procureur, S. (2021). Muon tomography of large structures with 2D projections. *Nuclear Instruments & Methods in Physics Research Section A-accelerators Spectrometers Detectors and Associated Equipment, 1013*, 165665. https://doi.org/10.1016/j.nima.2021.165665
 
Qin, Z., Zhang, R., Yu, P., Liu, C.-E., Chen, L., Zhang, F., Yang, Z., Li, Q., & Li, Q. (2025). Millimeter-Resolution Cosmic-Ray Imaging via Projection-Shifted Muon Transmission Tomography.
 
Ruta, D., & Ruta, R. (2023). Iterative Deep Learning for Muon Scattering Tomography. *2023 IEEE International Conference on Big Data (BigData)*, 6076-6083. https://doi.org/10.1109/bigdata59044.2023.10386973
 
Sattler, F., Alameddine, J., Rodríguez, B. Á., Stephan, M., & Barnes, S. (2025). A comprehensive framework toward the seamless integration of muon reconstruction algorithms with machine learning. *Journal of Applied Physics*. https://doi.org/10.1063/5.0288348
 
Sehgal, R., & Jha, V. (2025). Optical multiplexing for channel reduction in scintillator-based muon tomography system. *Journal of Applied Physics*. https://doi.org/10.1063/5.0273319
 
Thompson, L., & Steer, C. (2025). Muon tomography and its application to non-invasive tunnelling investigations. *Proceedings of the Southeastern Europe Tunnelling Conference (SETC-2025), 1 - 3 October 2025, Belgrade, Serbia - zbornik radova*. https://doi.org/10.5937/setc25028t
 
Ughade, R., & Chatzidakis, S. (2025). Efficient Low-Flux Muon Tomography Using the $\Mu$TRec Algorithm. *2025 IEEE Nuclear Science Symposium (NSS), Medical Imaging Conference (MIC) and Room Temperature Semiconductor Detector Conference (RTSD)*, 1-1. https://doi.org/10.1109/nss/mic/rtsd57106.2025.11287627
 
Valencia, J. J., Sperow, J. W., Durham, M. J., Osborne, A., Morris, C. L., Poulson, D., & Hecht, A. A. (2025). Simulations of muon imaging with the LANL GMT detector for spent nuclear fuel cask content verification. *Journal of Applied Physics*. https://doi.org/10.1063/5.0288369
 
Yashin, I., Davidenko, N., Dovgopoly, A., Fakhroutdinov, R., Kaverznev, M. M., Kompaniets, K., Konev, Y. N., Kozhin, A. S., Paramoshkina, E., Pasyuk, N., Tselinenko, M. Y., Yuschenko, O. P., & Zolotareva, O. (2021). Muon Tomography of Large-Scale Objects. *Physics of Atomic Nuclei, 84*, 1171-1181. https://doi.org/10.1134/s1063778821130421
 
Yu, P., Pan, Z., He, Z., Deng, L., Xu, Y., Yu, Y., Zhang, X.-H., Kang, Z., Chen, Z., Lin, Z., Chen, L.-W., Yang, L., & Sun, Z. (2024). A new efficient imaging reconstruction method for muon scattering tomography. *Nuclear Instruments and Methods in Physics Research Section A: Accelerators, Spectrometers, Detectors and Associated Equipment*. https://doi.org/10.1016/j.nima.2024.169932
 
Zhang, B., Wang, Z., & Chen, S. (2022). Mountain Muon Tomography Using a Liquid Scintillator Detector. *Applied Sciences*. https://doi.org/10.3390/app122110975
 
x