## **Detailed Technical Description and Comparison of μTRec, EM/Bayesian Scattering CT, and SART + Advanced Tracing + Momentum-Aware (M-value/mPoCA) Methods in Muon Tomography**

This review provides a detailed technical breakdown of the three leading classes of muon tomography reconstruction algorithms—μTRec, EM/Bayesian scattering CT, and SART with advanced tracing and momentum-aware extensions (M-value/mPoCA)—along with the principal metrics used for their quantitative comparison.

---

### **1. μTRec (Muon Trajectory Reconstruction Algorithm)**

**Technical Description:**
- **Physics-Informed Bayesian Framework:** μTRec reconstructs curved muon trajectories by explicitly modeling multiple Coulomb scattering (MCS) using a Gaussian approximation and Bayesian updating  (Ughade & Chatzidakis, 2026; Ughade & Chatzidakis, 2025).
- **Trajectory Estimation:** For each muon, μTRec estimates the most probable path through the object by:
  - Setting theoretical values for radiation length, energy loss constant, and average momentum.
  - Applying a linear energy loss model to account for energy dissipation along the path.
  - Computing scattering variances from MCS theory.
  - Constructing covariance and transport matrices to propagate uncertainties.
  - Using Bayesian updates to refine the estimated trajectory at each step  (Ughade & Chatzidakis, 2025).
- **Voxelized Mapping:** The reconstruction volume is discretized into voxels. Each voxel traversed by a muon receives an update based on the local scattering angle; statistics are aggregated over all muons to form a 3D scattering density map  (Ughade & Chatzidakis, 2025).
- **Momentum Integration:** Incorporates per-muon momentum measurements (when available), further refining trajectory estimation and enabling calculation of the M-value (a function of both scattering angle and momentum) for improved material discrimination  (Ughade & Chatzidakis, 2026; Ughade & Chatzidakis, 2025).

**Performance Highlights:**
- Achieves high spatial resolution (down to 1–5 cm voxels).
- Detects small defects (e.g., single missing fuel assemblies or flakes) with fewer muons compared to PoCA or simpler methods  (Ughade & Chatzidakis, 2026; Ughade & Chatzidakis, 2025).
- Up to ~20× reduction in required muon counts versus PoCA for comparable detection performance  (Ughade & Chatzidakis, 2025).
- Robust against practical detector limitations; only minor reductions in detectability with coarser spatial or energy resolution  (Ughade & Chatzidakis, 2026).

---

### **2. EM/Bayesian Scattering Computed Tomography (CT)**

**Technical Description:**
- **Statistical Model:** Models the probability distribution of observed scattering angles as a function of unknown local scattering densities within voxels  (Schultz et al., 2007; Wang et al., 2009).
- **Expectation-Maximization (EM):** Iteratively maximizes the likelihood (or posterior probability in Bayesian variants) that the reconstructed image explains the measured data:
  - **E-step:** Computes expected values of hidden variables (e.g., true scatter locations/angles) given current parameter estimates.
  - **M-step:** Updates voxel-wise scattering densities to maximize likelihood/posterior given these expectations  (Schultz et al., 2007; Wang et al., 2009).
- **Bayesian Regularization:** Introduces prior distributions on voxel values (e.g., smoothness priors, Laplacian/Gaussian), leading to maximum a posteriori (MAP) estimation. Regularization suppresses noise/artifacts but may introduce bias if priors are not well matched to reality  (Wang et al., 2009).
- **Shrinkage Algorithms:** Iterative shrinkage steps can be derived for specific priors, e.g., inverse quadratic/cubic shrinkage for Laplacian/Gaussian priors  (Wang et al., 2009).

**Performance Highlights:**
- Improves image quality and detection power over unregularized maximum likelihood approaches.
- Particularly effective at enhancing detectability of high-Z materials in noisy or low-statistics regimes  (Wang et al., 2009).
- Can outperform classical EM when accurate system response models are included  (Frese et al., 2003).

---

### **3. SART + Advanced Tracing + Momentum-Aware Extensions (M-value/mPoCA)**

**Technical Description:**
- **SART (Simultaneous Algebraic Reconstruction Technique):** An iterative algebraic method that updates all voxels simultaneously based on projection errors across all rays in each iteration. Correction terms are weighted along ray paths; bilinear elements can be used for more accurate integration over continuous images  (Andersen & Kak, 1984; Liu et al., 2018).
- **Advanced Tracing:**
  - Uses improved path models such as PoCA trajectory, straight-line through PoCA point, or full incident/outgoing track information rather than simple straight-line approximations.
  - Back-projects measured scattering angles along these refined paths for more accurate localization of interactions  (Liu et al., 2018; Bae & Chatzidakis, 2022).
- **Momentum-Aware Extensions:**
  - Incorporate measured or estimated muon momentum into reconstruction via M-value or mPoCA algorithms.
  - The M-value is defined as a mathematical combination of measured scattering angle and momentum, providing better discrimination between materials/densities without increasing computational cost  (Bae et al., 2023).
  - mPoCA assigns scatter points using both angular deflection and momentum information.

**Performance Highlights:**
- SART with advanced tracing outperforms filtered back-projection and basic ART in terms of noise suppression and artifact reduction but may have slightly lower spatial resolution due to smoothing effects from iterative updates  (Liu et al., 2018; Vaniqui et al., 2019; Andersen & Kak, 1984).
- Momentum-aware methods significantly improve image resolution and material discrimination compared to momentum-blind approaches; M-value mapping yields higher fidelity reconstructions at no extra computational cost when momentum data is available  (Bae et al., 2023).

---

### **4. Metrics for Quantitative Comparison**

The following metrics are commonly used across studies to compare these algorithms:

| Metric Name                | Definition / Use Case                                                                                  | Typical Application Papers |
|----------------------------|-------------------------------------------------------------------------------------------------------|---------------------------|
| Signal-to-noise ratio (SNR)| Ratio of mean signal intensity difference between regions-of-interest to standard deviation of background noise; higher SNR indicates clearer feature separation |  (Ughade & Chatzidakis, 2025; Liu et al., 2018)|
| Contrast-to-noise ratio (CNR)| Difference in mean signal between two regions divided by combined noise; measures ability to distinguish features/materials |  (Ughade & Chatzidakis, 2025; Liu et al., 2018)|
| Detection Power (DP)       | Statistical measure quantifying ability to detect anomalies/defects; often defined as Z-score or area under ROC curve |  (Ughade & Chatzidakis, 2025; Liu et al., 2018)|
| Peak Signal-to-noise Ratio (PSNR)| Measures maximum possible signal relative to noise; often used in simulation studies comparing reconstructed vs ground truth images |  (Luo et al., 2022)|
| Structural Similarity Index Measure (SSIM)| Quantifies perceived image quality by comparing luminance, contrast, structure between images |  (Xiang et al., 2020; Luo et al., 2022)|
| Mean Square Error (MSE)    | Average squared difference between reconstructed image and reference/ground truth                      |  (Xiang et al., 2020; Luo et al., 2022)|
| Spatial Resolution         | Smallest feature size reliably detected; often reported as minimum voxel size where features remain discernible |  (Ughade & Chatzidakis, 2025; Bae et al., 2023)|

#### Example Results:
- μTRec achieves up to 122% higher SNR, 35% higher CNR, and >200% higher DP than PoCA at equal muon flux/voxel size ( (Ughade & Chatzidakis, 2025)).
- SART with advanced tracing yields up to 76% higher CNR than FDK/Iterative FDK but up to 28% lower spatial resolution due to smoothing ( (Vaniqui et al., 2019)).
- Momentum-aware methods improve defect detectability by >100% compared with non-momentum-informed reconstructions ( (Ughade & Chatzidakis, 2026; Bae et al., 2023)).
- Hybrid models combining multiple techniques yield significantly higher PSNR than individual methods ( (Luo et al., 2022)).

---

### **Summary Table**

| Method                                   | Core Principle & Features                                                                                 | Key Strengths                                 | Main Metrics Used           | Representative Papers      |
|-------------------------------------------|----------------------------------------------------------------------------------------------------------|-----------------------------------------------|-----------------------------|---------------------------|
| μTRec                                    | Physics-informed Bayesian trajectory estimation with explicit MCS modeling & per-muon kinematics          | High accuracy/resolution at low muon counts   | SNR, CNR, DP, spatial res.  |  (Ughade & Chatzidakis, 2026; Ughade & Chatzidakis, 2025)|
| EM/Bayesian Scattering CT                 | Statistical ML/MAP estimation with regularization/prior incorporation                                     | Noise/artifact suppression; robust detection   | DP, ROC AUC, SSIM           |  (Schultz et al., 2007; Wang et al., 2009; Frese et al., 2003)|
| SART + Adv. Tracing + Momentum-Aware      | Simultaneous algebraic updates using refined path models & M-value/mPoCA extensions                       | Fast convergence; improved material ID         | CNR, PSNR, spatial res.     |  (Liu et al., 2018; Andersen & Kak, 1984; Bae et al., 2023)|

**Figure 1:** Comparison table summarizing technical details and evaluation metrics for μTRec, EM/Bayesian CT, and SART+momentum-aware methods.

---

### **References**

 (Ughade & Chatzidakis, 2026): "Non-intrusive Monitoring of Sealed Microreactor Cores Using Physics-Informed Muon Scattering Tomography With Momentum Measurements"  
 (Ughade & Chatzidakis, 2025): "$\mu$TRec: A Muon Trajectory Reconstruction Algorithm for Enhanced Scattering Tomography"  
 (Ughade & Chatzidakis, 2025): "Efficient Low-Flux Muon Tomography Using the $\\Mu$TRec Algorithm"  
 (Liu et al., 2018): "Muon Tracing and Image Reconstruction Algorithms for Cosmic Ray Muon Computed Tomography"  
 (Schultz et al., 2007): "Statistical Reconstruction for Cosmic Ray Muon Tomography"  
 (Luo et al., 2022): "Hybrid model for muon tomography and quantitative analysis of image quality"  
 (Frese et al., 2003): "Quantitative comparison of FBP, EM, and Bayesian reconstruction algorithms for the IndyPET scanner"  
 (Andersen & Kak, 1984): "Simultaneous Algebraic Reconstruction Technique (SART): A Superior Implementation of the Art Algorithm"  
 (Wang et al., 2009): "Bayesian Image Reconstruction for Improving Detection Performance of Muon Tomography"  
 (Vaniqui et al., 2019): "The effect of different image reconstruction techniques on pre-clinical quantitative imaging and dual-energy CT."  
 (Bae et al., 2023): "Image reconstruction algorithm for momentum dependent muon scattering tomography"

---

**In summary:**  
μTRec provides state-of-the-art performance by leveraging physics-based Bayesian inference with per-muon kinematics; EM/Bayesian CT offers robust statistical regularization especially valuable in noisy/low-statistics settings; SART plus advanced tracing/momentum-aware extensions deliver fast convergence with significant gains when accurate path modeling or momentum data are available. Their performance is rigorously benchmarked using standardized metrics such as SNR, CNR, DP, PSNR, SSIM, MSE, and direct measures of spatial resolution.
 
_These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at https://consensus.app. © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent._
 
## References
 
Andersen, A., & Kak, A. (1984). Simultaneous Algebraic Reconstruction Technique (SART): A Superior Implementation of the Art Algorithm. *Ultrasonic Imaging, 6*, 81 - 94. https://doi.org/10.1177/016173468400600107
 
Bae, J., & Chatzidakis, S. (2022). Momentum-Dependent Cosmic Ray Muon Computed Tomography Using a Fieldable Muon Spectrometer. *Energies*. https://doi.org/10.3390/en15072666
 
Bae, J., Montgomery, R., & Chatzidakis, S. (2023). Image reconstruction algorithm for momentum dependent muon scattering tomography. *Nuclear Engineering and Technology*. https://doi.org/10.1016/j.net.2023.12.009
 
Frese, T., Rouze, N., Bouman, C., Sauer, K., & Hutchins, G. (2003). Quantitative comparison of FBP, EM, and Bayesian reconstruction algorithms for the IndyPET scanner. *IEEE Transactions on Medical Imaging, 22*, 258-276. https://doi.org/10.1109/tmi.2002.808353
 
Liu, Z., Chatzidakis, S., Scaglione, J., Liao, C., Yang, H., & Hayward, J. (2018). Muon Tracing and Image Reconstruction Algorithms for Cosmic Ray Muon Computed Tomography. *IEEE Transactions on Image Processing, 28*, 426-435. https://doi.org/10.1109/tip.2018.2869667
 
Luo, S., Huang, Y., Ji, X., He, L., Xiao, W., Luo, F., Feng, S., Xiao, M., & Wang, X. (2022). Hybrid model for muon tomography and quantitative analysis of image quality. *Nuclear Science and Techniques, 33*. https://doi.org/10.1007/s41365-022-01070-6
 
Schultz, L., Blanpied, G., Borozdin, K., Fraser, A., Hengartner, N., Klimenko, A., Morris, C., Orum, C., & Sossong, M. (2007). Statistical Reconstruction for Cosmic Ray Muon Tomography. *IEEE Transactions on Image Processing, 16*, 1985-1993. https://doi.org/10.1109/tip.2007.901239
 
Ughade, R., & Chatzidakis, S. (2026). Non-intrusive Monitoring of Sealed Microreactor Cores Using Physics-Informed Muon Scattering Tomography With Momentum Measurements.
 
Ughade, R., & Chatzidakis, S. (2025). $\mu$TRec: A Muon Trajectory Reconstruction Algorithm for Enhanced Scattering Tomography. https://doi.org/10.1063/5.0278370
 
Ughade, R., & Chatzidakis, S. (2025). Efficient Low-Flux Muon Tomography Using the $\Mu$TRec Algorithm. *2025 IEEE Nuclear Science Symposium (NSS), Medical Imaging Conference (MIC) and Room Temperature Semiconductor Detector Conference (RTSD)*, 1-1. https://doi.org/10.1109/nss/mic/rtsd57106.2025.11287627
 
Vaniqui, A., Schyns, L., Almeida, I., Van Der Heyden, B., Podesta, M., & Verhaegen, F. (2019). The effect of different image reconstruction techniques on pre-clinical quantitative imaging and dual-energy CT.. *The British journal of radiology, 92 1095*, 20180447. https://doi.org/10.1259/bjr.20180447
 
Wang, G., Schultz, L., & Qi, J. (2009). Bayesian Image Reconstruction for Improving Detection Performance of Muon Tomography. *IEEE Transactions on Image Processing, 18*, 1080-1089. https://doi.org/10.1109/tip.2009.2014423
 
Xiang, J., Dong, Y., & Yang, Y. (2020). FISTA-Net: Learning a Fast Iterative Shrinkage Thresholding Network for Inverse Problems in Imaging. *IEEE Transactions on Medical Imaging, 40*, 1329-1339. https://doi.org/10.1109/tmi.2021.3054167
 
