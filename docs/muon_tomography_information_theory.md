# **Optimizing geophysical muon radiography using information theory:** A focused review

This work develops a fully probabilistic, information‑theoretic framework to quantify and optimize what geophysical muon radiography can reveal, especially for very **low‑contrast density changes** such as CO₂ storage monitoring. It combines Poisson communication theory with a fast expectation‑value muon transport model and demonstrates performance on a realistic North Sea–type case.

## Conceptual framework and information-theoretic methods

**Muon counts as a Poisson information channel**  
- Muon arrivals in each angular bin are modeled as independent **Poisson counts** with expectation λₘ = t μₘ, where μₘ is the mean arrival rate in bin m and t is integration time  (Benton et al., 2020).  
- Small geological perturbations are parameterized by quantities χₙ (e.g., layer densities, plume extent), with fractional rate change  
  ηₘ = Σₙ ζₙₘ χₙ, where ζₙₘ = (1/μₘ)(∂μₘ/∂χₙ) at a reference model  (Benton et al., 2020).  
- Using **Poisson communication theory**, muon radiography is mapped onto a noisy multi‑input multi‑output photon channel with high dark noise. The **channel capacity** (maximum information rate) in bits/s is  
  C_b = (ln2 / 8) Σₘ μ̃ₘ [Σₙ |ζₙₘ| / Δχₙ]²,  
  where μ̃ₘ is a typical rate and Δχₙ is the feasible range of χₙ  (Benton et al., 2020).  

**Orthogonal geological parameters and hypothesis testing**  
- When multiple χₙ affect overlapping paths, parameters are rotated to an **orthogonal basis** χ′ᵢ with coefficients ζ′ᵢₘ chosen so that different χ′ᵢ are statistically independent in the data (eigen‑decomposition of T = ζ ζᵀ)  (Benton et al., 2020).  
- For each χ′ᵢ, an **optimal linear test statistic** is  
  Ψᵢ = Σₘ ζ′ᵢₘ kₘ, where kₘ are observed counts  (Benton et al., 2020).  
- Under large total counts, Ψᵢ is approximately Gaussian with mean and variance computed from λₘ, enabling **Z‑tests** between hypotheses (e.g., “no CO₂” vs “CO₂ plume”); the mean time to separate two hypotheses A,B is  
  tᵢ = 1 / { (ln2/8) Σₘ μ̃ₘ ζ′ᵢₘ² (χ′ᵢ(A)–χ′ᵢ(B))² }  (Benton et al., 2020).  

## Muon flux modeling: expectation-based ray tracing

**Survival probability and effective distance**  
- A **survival probability** Ξ(s,θ) (probability a muon survives distance s in rock for zenith angle θ) is computed once using the MUSIC Monte Carlo code with a realistic surface spectrum (Gaisser parameterization, E ≥ 100 GeV)  (Benton et al., 2020).  
- To handle variable density ρ(s), an **effective distance** is defined as  
  ŝ = ∫ (ρ(s)/ρ₀) ds, with ρ₀ = 2.65 g cm⁻³ (“standard rock”); Ξ is tabulated in ŝ  (Benton et al., 2020).  

**Probability ray tracing**  
- For each direction m, a straight‑line ray is traced through a 3‑D density model, computing ŝₘ and using  
  Jₘ = G(θₘ) Ξ(ŝₘ,θₘ)  
  where G(θ) is the integrated ground‑level flux  (Benton et al., 2020).  
- This gives **direct expectation values Jₘ**, free of Monte‑Carlo sampling noise, allowing stable numerical derivatives (∂Jₘ/∂χₙ) needed for ζₙₘ  (Benton et al., 2020).  

## Application: CO₂ storage monitoring case study

**Geological and flow model**  
- Case based on the Boulby Mine / North Sea–type setting: sea, Liassic shale, Mercia mudstone, Bunter sandstone, and evaporites; a deep detector in mine workings views a CO₂ plume injected into Bunter sandstone and trapped by Mercia mudstone  (Benton et al., 2020).  
- CO₂ migration is modeled analytically: Bunter split into “filled” and “unfilled” subregions, with plume thickness h(x,t) from a similarity solution for vertical injection into a confined, porous aquifer (functions of injection rate, viscosities, densities, porosity, relative permeabilities)  (Benton et al., 2020).  

**Radiography and information gain**  
- Ray tracing produces Jₘ with and without 1 year of injection. Relative flux changes are ∼1 % or less across the field of view  (Benton et al., 2020).  
- From these Jₘ, ζₙₘ and per‑direction information rate are computed. Angular bins contributing most to information are those passing through the plume and contrasting density interfaces  (Benton et al., 2020).  

**Comparison of test statistics**  
- Two statistics are compared:  
  - Simple sum over a plume‑crossing ROI: Φ = Σₘ δₘ kₘ (δₘ=1 in ROI, 0 otherwise).  
  - Optimal weighted sum: Ψ = Σₘ ζₘ kₘ  (Benton et al., 2020).  
- Using analytical expressions for means/variances and a one‑sided Z‑test, the ratio of required observation times for same detection significance and detector area is  
  t_Ψ a_Ψ ≈ 0.48 t_Φ a_Φ,  
  implying **roughly a factor of two improvement in effective sensitivity** (time or area) by using the information‑theoretic weighting  (Benton et al., 2020).  

### Key optimization insights

| Aspect                         | Information-theory optimization                                                  | Citations |
|-------------------------------|----------------------------------------------------------------------------------|----------|
| Fundamental limit             | Channel capacity C_b from Poisson MIMO model of muon counts                     |  (Benton et al., 2020)|
| Optimal statistic             | Linear combination with weights ζ′ᵢₘ, derived from ∂Jₘ/∂χₙ and orthogonalization |  (Benton et al., 2020)|
| Detector/geometry design      | Maximize Σₘ μ̃ₘ [Σₙ |ζₙₘ| / Δχₙ]², i.e., both flux and contrast along paths      |  (Benton et al., 2020)|
| Angular resolution guideline  | Limiting angular resolution φ_lim ≈ 1 / (Δηₘ √(Jₘ aₘ t)) from information budget |  (Benton et al., 2020)|
| Practical gain in CO₂ case    | ~2× reduction in required area or observation time vs unweighted counts         |  (Benton et al., 2020)|

**Figure 1:** Core components of the information-theoretic muon radiography framework.

## Relation to broader muon radiography work

A second paper reviews information extraction from **scattering muon radiography** (for high‑Z detection) using PoCA‑based imaging, maximum‑likelihood tomographic reconstruction, voxel‑wise high‑scatter counting, SVM classifiers, and clustering, and discusses added value from energy and stopping‑power measurements  (Borozdin et al., 2024). That work focuses on imaging and classification algorithms; the Benton et al. paper instead addresses the **information limit and optimal statistics** for low‑contrast geophysical radiography.

## Summary

“Optimizing geophysical muon radiography using information theory” provides:

- A rigorous mapping of muon radiography to **Poisson information channels**, yielding explicit formulas for maximum information rate and for the minimal time to distinguish geological hypotheses.  
- An **expectation‑value ray tracer** for fast, noise‑free computation of muon flux and its sensitivities to geological parameters.  
- A demonstration in a CO₂ storage scenario, showing that using optimally weighted test statistics can **approximately double the effective sensitivity** of muon detectors for subtle density changes.  

Together, these results give a principled way to design detector layouts, choose angular binning, and analyze data for low‑contrast, large‑scale geophysical muon applications.
 
_These search results were found and analyzed using Consensus, an AI-powered search engine for research. Try it at https://consensus.app. © 2026 Consensus NLP, Inc. Personal, non-commercial use only; redistribution requires copyright holders’ consent._
 
## References
 
Benton, C., Mitchell, C., Coleman, M., Paling, S., Lincoln, D., Thompson, L., Clark, S. J., & Gluyas, J. (2020). Optimizing geophysical muon radiography using information theory. *Geophysical Journal International*. https://doi.org/10.1093/gji/ggz503
 
Borozdin, K., Asaki, T., Chartrand, R., Hengartner, N., Hogan, G., Morris, C., Priedhorsky, W., Schirato, R., Schultz, L., Sottile, M., Vixie, K., Wohlberg, B., & Blanpied, G. (2024). Information extraction from muon radiography data.
 
