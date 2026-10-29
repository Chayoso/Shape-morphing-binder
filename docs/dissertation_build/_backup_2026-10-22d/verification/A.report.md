# Batch A report (MPM / continuum simulation, 24 keys)

Access notes: dblp.org/search and dl.acm.org, ScienceDirect, Wiley and Springer pages were bot-blocked (Anubis/Cloudflare/403)
during this run. DBLP was read via its official SPARQL endpoint (https://sparql.dblp.org/sparql, record keys given below), which
returns the same records (authors with ordinals, pagination incl. ACM article numbers, volume, issue, year). Crossref API was
read for every DOI. Where ACM page counts were absent from Crossref/DBLP, the ACM-formatted accepted manuscript on NSF PAR
("ACM Reference Format" line) was opened. Elsevier chapter author lists came from the institutional repositories (Monash, TU Delft)
because Crossref holds no authors for those two chapters.

Status rule used: VERIFIED-UNCHANGED = every factual value in the input was correct (article numbers / page counts already in
articleno/numpages); only re-formatting to POLICY.md (e.g. articleno+numpages -> pages = {N:1--N:M}, dropped url/issn/month/publisher,
title braces). CORRECTED = a factual value was wrong or a fact (DOI, issue, article number) was missing and was added.

---

KEY: 2013-mpm
STATUS: VERIFIED-UNCHANGED
CHANGES: formatting only (pages 1--10 -> 102:1--102:10 from the verified article number 102; title braces removed; url/issn/month/publisher dropped)
SOURCES: https://api.crossref.org/works/10.1145/2461912.2461948 ; DBLP journals/tog/StomakhinSCTS13 (SPARQL: 102:1-102:10, 32(4), 2013)
CLAIMS: OK (Ch1:21 elastoplastic animation; Ch5:109 particles + background grid; App:186 fixed corotated -- the snow paper uses the fixed-corotated energy with plasticity)

KEY: 2015-APIC
STATUS: VERIFIED-UNCHANGED
CHANGES: formatting only (articleno 51 / numpages 10 -> pages 51:1--51:10)
SOURCES: https://api.crossref.org/works/10.1145/2766996 (abstract read) ; DBLP journals/tog/JiangSSTS15 (51:1-51:10)
CLAIMS: OK (abstract: removes PIC dissipation, exact conservation of angular momentum across transfers)

KEY: 2018-MLSMPM
STATUS: VERIFIED-UNCHANGED
CHANGES: formatting only (articleno 150 / numpages 14 -> pages 150:1--150:14; Crossref page 1-14)
SOURCES: https://api.crossref.org/works/10.1145/3197517.3201293 (abstract read) ; DBLP journals/tog/HuFGQZPJ18 (article 150)
CLAIMS: MISMATCH: Chapter2.tex:15 "MLS-MPM~\citep{2018-MLSMPM} leverages Moving Least Squares to robustly estimate the deformation gradient ($\mathbf{F}$), alleviating the classic cell-crossing instabilities when particles traverse grid boundaries and improving stability under extreme deformations."
  -- paper says: MLS shape functions give APIC/PolyPIC transfers consistent with a Galerkin weak form, a new stress-divergence discretization that makes MPM about two times faster, and the CPIC algorithm for cutting, dynamic open boundaries and two-way rigid-body coupling. It does not claim to estimate F more robustly or to remove cell-crossing instability (that is GIMP / B-spline MPM, e.g. bardenhagen2004generalized).
  -- proposed: "MLS-MPM~\citep{2018-MLSMPM} uses Moving Least Squares shape functions to derive APIC-consistent transfers and a faster stress-divergence discretization, and its compatible particle-in-cell extension supports cutting and two-way rigid-body coupling."
  Other contexts (Ch1:26, Ch2:56, Ch3:14/16/41, Ch5:24, Ch5:109, App:198): OK.

KEY: 2019-cdmpm
STATUS: VERIFIED-UNCHANGED
CHANGES: formatting only (articleno 119 / numpages 15 -> pages 119:1--119:15; title braces)
SOURCES: https://api.crossref.org/works/10.1145/3306346.3322949 (title "CD-MPM" + subtitle; abstract read) ; DBLP journals/tog/WolperFLLGJ19 (119:1-119:15)
CLAIMS: MISMATCH: Chapter2.tex:19 "complex plasticity (e.g., sand, snow)~\citep{2019-cdmpm}"
  -- paper says: dynamic fracture of elastoplastic materials via continuum damage (PFF-MPM) and a Non-Associated Cam-Clay plasticity model with fracture-friendly hardening; its subject is fracture, not sand/snow plasticity (snow = 2013-mpm).
  -- proposed: "complex plasticity (e.g., snow)~\citep{2013-mpm}, elastoplastic fracture~\citep{2019-cdmpm}"
  Ch1:21, Ch3:14, Ch5:109 ("continuum damage models permit progressive weakening"): OK.

KEY: jiang2016material
STATUS: VERIFIED-UNCHANGED
CHANGES: formatting only (booktitle "ACM SIGGRAPH 2016 Courses" -> "{ACM SIGGRAPH} Courses"; articleno 24 / numpages 52 -> pages 24:1--24:52; dropped series/location/isbn/publisher)
SOURCES: https://api.crossref.org/works/10.1145/2897826.2927348 ; DBLP conf/siggraph/JiangSTSS16 (24:1-24:52) ; course notes PDF (title page, authors, DOI) https://web.archive.org/web/2023id_/https://www.math.ucla.edu/~cffjiang/research/mpmcourse/mpmcourse.pdf
CLAIMS: MISMATCH (weak support): Chapter4.tex:103 "since $\boldsymbol{\tau}_p$ is symmetric, APIC keeps the angular momentum as well~\citep{jiang2016material}."
  -- the course notes' APIC section (Sec. 10.1) gives the transfers and only says APIC has "nice numerical properties"; they state generally that a symmetric stress is needed for angular-momentum conservation. The proof that APIC transfers conserve angular momentum is in jiang2017angular (and 2015-APIC).
  -- proposed: "... APIC keeps the angular momentum as well~\citep{jiang2017angular}."
  Minor: Appendix.tex:198 "with APIC~\citep{jiang2016material}" -- the primary APIC source is 2015-APIC; proposed "with APIC~\citep{2015-APIC}". (Ch4:68 in the current source already cites 2015-APIC,jiang2016material: OK.)
  Ch2:52, Ch3:74, App:52, App:96: OK (notes cover MPM, fixed corotated P = 2mu(F-R) + lambda(J-1)JF^{-T}).

KEY: jiang2017angular
STATUS: VERIFIED-UNCHANGED
CHANGES: formatting only (pages 137-164 -> 137--164; url/issn dropped)
SOURCES: https://api.crossref.org/works/10.1016/j.jcp.2017.02.050 ; Semantic Scholar record (abstract; arXiv 1603.06188), https://api.semanticscholar.org/graph/v1/paper/DOI:10.1016/j.jcp.2017.02.050
CLAIMS: MISMATCH: Chapter2.tex:15 "the \textbf{APIC} (Affine Particle-in-Cell) scheme~\citep{jiang2017angular} was proposed"
  -- APIC was proposed in Jiang et al. 2015 (ACM TOG, key 2015-APIC); the JCP 2017 paper is the follow-up analysis (angular-momentum conservation with lumped mass, midpoint rule).
  -- proposed: "scheme~\citep{2015-APIC,jiang2017angular} was proposed"
  Ch1:25, Ch3:26, Ch3:35, Ch5:109, Ch5:134, App:109, App:138: OK.

KEY: sulsky1994particle
STATUS: CORRECTED
CHANGES: number 1 -> 1--2 ; pages 179-196 -> 179--196 ; author initials formatted "Schreyer, H. L."
SOURCES: https://api.crossref.org/works/10.1016/0045-7825(94)90112-0 (vol 118, issue 1-2, 179-196, Sept 1994) ; paper PDF https://users.cs.utah.edu/~mhall/cs6963s09/lectures/Sulsky_1994.pdf (header, authors, abstract)
CLAIMS: MISMATCH: Chapter2.tex:13 "introduced by~\citet{sulsky1994particle} to solve solid mechanics problems involving large deformations and fracture."
  -- paper says: penetration, impact and large rotations of solid bodies with history-dependent constitutive laws (elasticity, strain-hardening plasticity, no-slip impact); fracture is not treated.
  -- proposed: "... problems involving large deformations, impact, and penetration."
  Ch1:21 ("introduced for history-dependent materials"), Ch2:52, Ch3:14 ("30 years ago"), Ch5:109: OK.

KEY: de2020material
STATUS: CORRECTED
CHANGES: author "De Vaucorbeil, Alban" -> "de Vaucorbeil, Alban" (lower-case particle as published); doi added 10.1016/bs.aams.2019.11.001; publisher dropped. Kept as @article (Advances in Applied Mechanics is a serial, ISSN 0065-2156, vol 53); Crossref types it "book-chapter" -- switch to @incollection only if the coordinator wants serial volumes as chapters.
SOURCES: https://api.crossref.org/works/10.1016/bs.aams.2019.11.001 (title, 185-398, 2020; no authors in Crossref) ; https://research.monash.edu/en/publications/material-point-method-after-25-years-theory-implementation-and-ap/ (authors, Vol. 53, 185-398, DOI) ; https://api.openalex.org/works/https://doi.org/10.1016/bs.aams.2019.11.001 (raw author names)
CLAIMS: OK

KEY: solowski2021material
STATUS: CORRECTED
CHANGES: doi added 10.1016/bs.aams.2020.12.002; author names re-formatted (So{\l}owski, Wojciech T.; Guilkey, James E.; M{\"o}ller, Matthias; ...), order and spelling unchanged; publisher dropped. Kept as @article (serial vol 54), see de2020material.
SOURCES: https://api.crossref.org/works/10.1016/bs.aams.2020.12.002 (title, 113-204, 2021) ; https://research.tudelft.nl/en/publications/material-point-method-overview-and-challenges-ahead (10 authors in order, vol 54, 113-204) ; OpenAlex record (raw names incl. "Sołowski")
CLAIMS: OK

KEY: bardenhagen2004generalized
STATUS: VERIFIED-UNCHANGED
CHANGES: formatting only (url/issn dropped)
SOURCES: https://doi.org/10.3970/cmes.2004.005.477 (resolves to) https://www.techscience.com/CMES/v5n6/33378 (title, S. G. Bardenhagen, E. M. Kober, CMES 5(6) 477-496, 2004). Not in Crossref (Crossref returns 404 for this DOI; the DOI resolves).
CLAIMS: OK

KEY: bardenhagen2000granular
STATUS: VERIFIED-UNCHANGED
CHANGES: none
SOURCES: https://api.crossref.org/works/10.1016/S0045-7825(99)00338-2 (187(3-4) 529-541, July 2000) ; paper PDF http://www.math.unm.edu/~sulsky/papers/CMAME.pdf (abstract, contact algorithm)
CLAIMS: OK (Appendix.tex:362: the paper's contact algorithm compares each grain's nodal velocity with the centre-of-mass velocity, adjusts the normal component and applies Coulomb friction, with normals from mass-density gradients. Note: Bardenhagen uses each body's own gradient, "not along a common normal"; the dissertation's "difference of the fields' mass gradients" is a later variant and is not attributed to Bardenhagen in that sentence, so no change needed)

KEY: stomakhin2012energetically
STATUS: CORRECTED
CHANGES: author "Schroeder, Craig A" -> "Schroeder, Craig" (as printed on the paper; DBLP lists "Craig A. Schroeder"); booktitle "Proceedings of the ACM SIGGRAPH/Eurographics Symposium on Computer Animation" -> policy form with ({SCA}); dropped numpages/publisher/series/isbn/url. Pages 25--32, year, DOI unchanged.
SOURCES: DBLP conf/sca/StomakhinHST12 (SPARQL: 25-32, 2012, DOI 10.2312/SCA/SCA12/025-032) ; https://doi.org/10.2312/SCA/SCA12/025-032 (resolves to diglib.eg.org handle; diglib itself refused connections) ; author PDF https://math.ucdavis.edu/~jteran/papers/SHST12.pdf (header "Eurographics/ACM SIGGRAPH Symposium on Computer Animation (2012), P. Kry and J. Lee (Editors)", author line, abstract)
CLAIMS: MISMATCH (wording): Appendix.tex:96 "We use the fixed-corotated linear elastic model \citep{stomakhin2012energetically, jiang2016material}"
  -- fixed corotated is a hyperelastic model (Stomakhin et al. fix the psi term of the hyperelastic corotated model; the course notes call it a modification of corotated linear elasticity), not a linear elastic model.
  -- proposed: "We use the fixed-corotated hyperelastic model"
  MISMATCH (wording): Chapter5.tex:147 "We employ corotated elasticity~\citep{stomakhin2012energetically}" -- the formula that follows (P = 2mu(F-R) + lambda(J-1)JF^{-T}) is the fixed corotated model; the paper shows the plain corotated model is the problematic one.
  -- proposed: "We employ fixed corotated elasticity~\citep{stomakhin2012energetically}"
  Ch3:74, App:186: OK. (Aside, not a citation issue: "Piola-Kirchoff" at Ch3:74 and App:96 should be "Piola--Kirchhoff".)

KEY: gast2016implicit
STATUS: CORRECTED
CHANGES: entry type @article with journal = {Technical report} and publisher {University of California} -> @techreport with institution = {University of California, Los Angeles}; title "...3$\times$ 3 matrices" -> "Implicit-Shifted Symmetric {QR} Singular Value Decomposition of {$3 \times 3$} Matrices". No report number exists on the document; none added. (@techreport is not in POLICY.md's list; apalike prints it as "Technical report, University of California, Los Angeles". If the coordinator prefers @misc: howpublished = {Technical report, University of California, Los Angeles}.)
SOURCES: report PDF https://web.archive.org/web/2023id_/https://www.math.ucla.edu/~cffjiang/research/svd/svd.pdf (title, 4 authors, UCLA Department of Mathematics, PDF CreationDate 2016-04-27) ; the authors' own citation in the 2016 MPM course notes: "Gast, T., Fu, C., Jiang, C., and Teran, J. (2016). ... Technical report, University of California Los Angeles."
CLAIMS: OK (the report computes the 3x3 SVD and includes polar decomposition; R = UV^T)

KEY: fang2020iq
STATUS: VERIFIED-UNCHANGED
CHANGES: formatting only (articleno 51 / numpages 16 -> 51:1--51:16; "Non-sticky" shown as Title Case "Non-Sticky")
SOURCES: https://api.crossref.org/works/10.1145/3386569.3392438 ; DBLP journals/tog/FangQLZZAJ20 (article 51) ; NSF PAR accepted manuscript https://par.nsf.gov/servlets/purl/10180981 ("ACM Trans. Graph. 39, 4, Article 51 (July 2020), 16 pages")
CLAIMS: OK

KEY: conservative_surface_tension
STATUS: VERIFIED-UNCHANGED
CHANGES: formatting only (pages 1--16 + articleno 111 -> 111:1--111:16)
SOURCES: https://api.crossref.org/works/10.1145/3450626.3459874 ; DBLP journals/tog/ChenKMGHT21 (111:1-111:16)
CLAIMS: OK

KEY: implicit_mpm_surface_tension
STATUS: VERIFIED-UNCHANGED
CHANGES: formatting only (pages 1--13 + articleno 183 -> 183:1--183:13)
SOURCES: https://api.crossref.org/works/10.1145/3414685.3417845 (abstract read) ; DBLP journals/tog/HydeGMT20 (183:1-183:13)
CLAIMS: MISMATCH (minor): Chapter2.tex:17 "Implicit time-integration techniques~\citep{implicit_mpm_surface_tension, wang2020hierarchical} increase robustness for stiff materials and allow larger time steps"
  -- Hyde et al. make implicit time stepping efficient for liquids with high surface tension (e.g. liquid metals); "stiff materials" fits HOT, the surface-tension paper is about stiff surface energy.
  -- proposed: "increase robustness for stiff materials and strong surface tension and allow larger time steps"
  Ch3:86: OK.

KEY: guo2018material
STATUS: CORRECTED
CHANGES: pages 1--15 -> 147:1--147:15 (article number 147 was missing); doi added 10.1145/3197517.3201346; publisher dropped
SOURCES: https://api.crossref.org/works/10.1145/3197517.3201346 (found via Crossref bibliographic search; abstract read) ; DBLP journals/tog/GuoHFGTT18 (article 147)
CLAIMS: OK

KEY: wang2020hierarchical
STATUS: CORRECTED
CHANGES: pages 1--16 -> 21:1--21:16 (article number 21 was missing); doi added 10.1145/3386760; title capitals per published title (Title Case, {CFL}, {MPM}); publisher dropped
SOURCES: https://api.crossref.org/works/10.1145/3386760 (39(3), 1-16, 2020; abstract read) ; DBLP journals/tog/WangLFZGTKJ20 (21:1-21:16)
CLAIMS: OK

KEY: wang2020massively
STATUS: VERIFIED-UNCHANGED
CHANGES: formatting only (articleno 30 / numpages 15 -> 30:1--30:15). Note: DBLP's record title has a typo ("multi-CPU"); Crossref and the paper say "Multi-GPU", kept.
SOURCES: https://api.crossref.org/works/10.1145/3386569.3392442 (abstract read) ; DBLP journals/tog/WangQSFLZZ0MJ20 (article 30) ; NSF PAR manuscript https://par.nsf.gov/servlets/purl/10180983 ("Article 30 (July 2020), 15 pages")
CLAIMS: MISMATCH: Chapter2.tex:19 "and fracture coupled with topological change~\citep{wang2020massively}"
  -- paper says: a multi-GPU MPM system (particle data structure, G2P2G kernel fusion, sparse grids; 100x over CPU, ~100M particles); demos are elastoplasticity, granular media and fluids. It proposes no fracture model.
  -- proposed: move it to the performance sentence (Ch2:21): "GPU acceleration~\citep{wang2020massively}" and cite fracture to "fracture coupled with topological change~\citep{2019-cdmpm,10.1145/3340259}".

KEY: chen2025sparse
STATUS: CORRECTED
CHANGES: doi added 10.1016/j.compgeo.2025.107113 (pages = article number 107113, volume 180, 2025 unchanged); title capitals per policy ({GPU-MPM}); publisher dropped
SOURCES: https://api.crossref.org/works/10.1016/j.compgeo.2025.107113 (authors, vol 180, article 107113, April 2025)
CLAIMS: OK (by title; Crossref has no abstract and ScienceDirect was blocked)

KEY: kakouris2024high
STATUS: PUBLISHED-VERSION-SUBSTITUTED
CHANGES: @misc arXiv:2403.13534 (2024, "A high-fidelity material point method for frictional contact problems") -> @article International Journal for Numerical Methods in Engineering 126(3), e70003, 2025, "An Extended {B-Spline}-Based Material Point Method for Contact Problems", doi 10.1002/nme.70003. Same three authors in the same order and a near-verbatim abstract (penalty contact at discretised boundaries + Extended B-splines). Third author as published: "Triantafyllou, Savvas" (arXiv: "Savvas P."). Year 2024 -> 2025 (the key stays kakouris2024high).
SOURCES: https://export.arxiv.org/api/query?id_list=2403.13534 (v1 only, no journal-ref) ; https://api.crossref.org/works/10.1002/nme.70003 (authors with ORCIDs, title, vol 126, issue 3, e70003, online 2025-02-07, abstract) ; arXiv PDF https://arxiv.org/pdf/2403.13534v1 (friction treated throughout; impact examples). Wiley page blocked (403).
CLAIMS: OK with a note: Chapter2.tex:17 "high-fidelity frictional contact algorithms~\citep{kakouris2024high} are now available for scenarios with repeated contact and separation" -- the paper does treat Coulomb-type frictional contact and impact (bodies meet and separate); the published title dropped "frictional"/"high-fidelity", so the wording is still supported. Optional: "accurate frictional contact algorithms".

KEY: du2025jax
STATUS: CORRECTED
CHANGES: number added 3; article number 85 moved from note/articleno to pages = {85}; note dropped. Authors, title, volume 42, year 2026, DOI unchanged.
SOURCES: https://api.crossref.org/works/10.1007/s00366-026-02320-6 (42(3), article 85, online 2026-04-28, print June 2026) ; OpenAlex abstract https://api.openalex.org/works/https://doi.org/10.1007/s00366-026-02320-6. Springer page redirected to a login.
CLAIMS: OK (JAX, GPU acceleration, automatic differentiation; 2.7M-particle 3D benchmark)

KEY: 10.1145/3386569.3392428
STATUS: VERIFIED-UNCHANGED
CHANGES: formatting only (title "{AnisoMPM}: animating anisotropic damage mechanics" -> Title Case "{AnisoMPM}: Animating Anisotropic Damage Mechanics"; articleno 37 / numpages 16 -> 37:1--37:16)
SOURCES: https://api.crossref.org/works/10.1145/3386569.3392428 (abstract) ; DBLP journals/tog/WolperCLFQLCJ20 (article 37) ; NSF PAR manuscript https://par.nsf.gov/servlets/purl/10180979 ("Article 37 (July 2020), 16 pages"; intro: steak, pork belly, mozzarella, orange)
CLAIMS: OK ("anisotropic media (e.g., meats and various fruits)")

KEY: 10.1145/3340259
STATUS: VERIFIED-UNCHANGED
CHANGES: formatting only (articleno 18 / numpages 20 -> 18:1--18:20)
SOURCES: https://api.crossref.org/works/10.1145/3340259 (abstract) ; DBLP journals/pacmcgit/WangDGZGJT19 (18:1-18:20)
CLAIMS: OK (Ch5:109 "progressive weakening": paper models failure with elastoplasticity + damage, a softening model shrinking the yield surface)
