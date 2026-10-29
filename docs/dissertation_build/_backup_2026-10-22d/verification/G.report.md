# Batch G report: robotic cutting simulators and food-science material papers (13 keys)

Access notes: DBLP refused automated access (HTTP 429, then an Anubis bot wall / 403) and Semantic Scholar was rate-limited.
I verified metadata through the Crossref API, arXiv API, PMLR, roboticsproceedings.org, PubMed/PMC, OpenAlex and the
publishers' own PDFs. Taylor & Francis (banana, peach) returned 403 for the full text, so for those two papers only the
abstract (OpenAlex/Crossref) and the publisher metadata were checked.

Context for the claim checks: Chapter5.tex:82 says the comparison table lists "capabilities" and that "--" means "not
identified as part of the referenced benchmark". Chapter5.tex:248 says the material values "are initialized from
food-engineering references and adjusted when necessary for numerical stability". The table nonetheless attaches one
citation to each row, so a reader will assume that the cited paper reports all three numbers. The checks below test that
assumption.

---

KEY: disect2022
STATUS: CORRECTED
CHANGES: title "DiSECt: A Differentiable Simulator for Parameter Inference and Control in Robotic Cutting" -> "DiSECt: A
  Differentiable Simulation Engine for Autonomous Robotic Cutting". The old title belongs to the extended version
  (arXiv:2203.10263, 2022), which was published as Autonomous Robots 47(5):549--578, 2023, DOI 10.1007/s10514-023-10094-9.
  The entry's own booktitle (RSS) and year (2021) belong to the RSS paper, so I kept RSS 2021 as the parent asked and fixed
  the title to match. DOI added: 10.15607/RSS.2021.XVII.067. The url was dropped. Note that the key reads "2022". If the
  author meant the extended version instead, the entry would be:
  @article{disect2022, author = {Heiden, Eric and Macklin, Miles and Narang, Yashraj and Fox, Dieter and Garg, Animesh and
  Ramos, Fabio}, title = {{DiSECt}: A Differentiable Simulator for Parameter Inference and Control in Robotic Cutting},
  journal = {Autonomous Robots}, volume = {47}, number = {5}, pages = {549--578}, year = {2023}, doi = {10.1007/s10514-023-10094-9}}
  (verified on Crossref). The SliceIt! paper cites that journal version.
SOURCES: https://api.crossref.org/works/10.15607/RSS.2021.XVII.067 ; http://www.roboticsproceedings.org/rss17/p067.pdf (title page read) ;
  https://export.arxiv.org/api/query?id_list=2105.12244,2203.10263 ; Crossref search, which returned 10.1007/s10514-023-10094-9
CLAIMS:
  - Chapter2.tex:97 "differentiable cutting simulator with small real datasets for calibration": OK. The paper is calibrated to
    real-world knife-force datasets (Jamdagni's).
  - Chapter5.tex:77 "employ finite-element approaches to cutting": OK. The paper describes FEM plus an SDF contact and
    damage-spring model.
  - MISMATCH: Chapter5.tex:96 table row "DiSECt & FEM & -- & Synthetic & Yes & ..." -- the "Multi-cam." column says Yes. The
    paper says: DiSECt has no camera observations or images at all. It is a force/deformation simulator, and the RSS text has
    no "camera" anywhere. Proposed: Multi-cam. "--".
  - Minor: the "Data origin: Synthetic" column conflicts with Chapter 2's own phrase "small real datasets for calibration".
    Proposed: "Hybrid", or leave it, since the caption is about capabilities.

KEY: sliceit2024
STATUS: CORRECTED
CHANGES: author "Beltran-Hernandez, Carlos C. and others" -> "Beltran-Hernandez, Cristian C. and Erbetti, Nicolas and
  Hamaya, Masashi". The first name was wrong: it is Cristian, not Carlos, and the full list has three authors. Booktitle
  -> policy form (ICRA). pages added: 4296--4302. DOI added: 10.1109/ICRA57147.2024.10611596. The url (arXiv) was dropped
  because the published ICRA version is cited. Crossref gives the title as "SliceIt! - A Dual Simulator Framework for
  Learning Robot Food Slicing"; it is kept with the dash as "--".
SOURCES: https://api.crossref.org/works/10.1109/icra57147.2024.10611596 ; https://export.arxiv.org/api/query?id_list=2404.02569 ;
  https://arxiv.org/pdf/2404.02569v2 (full text read)
CLAIMS:
  - Chapter2.tex:97 "dual high-fidelity cutting+robot simulator with limited real slicing for real2sim2real transfer": OK.
    The paper describes a real2sim2real setup with DiSECt as CutSim and Gazebo as RoboSim, needing "as few as one slice
    motion of each food".
  - Chapter5.tex:77 "employ finite-element approaches": OK. The cutting simulator is DiSECt, which is FEM-based.
  - Chapter5.tex:95 table "FEM & Gazebo & Synthetic & -- & -- & -- & Yes": FEM and Gazebo are OK.
    Possible MISMATCH in "Cont. action --". The paper says it trains SAC, a continuous-action RL algorithm, whose actions are
    "the reference trajectory x_c and the control parameters [K_c, K_p, K_d]" of a compliance controller. Proposed:
    Cont. action "Yes". Data origin: the policy is trained in simulation but calibrated on, and deployed to, a real robot;
    "Synthetic" is defensible, "Hybrid" is more exact.

KEY: topocut2025
STATUS: VERIFIED-UNCHANGED
CHANGES: Formatting only. The booktitle "Proceedings of the 9th Conference on Robot Learning" was put in policy form
  (CoRL), and the url and publisher fields were dropped. Authors, title, volume 305, pages 1282--1310 and year 2025 all match
  PMLR. PMLR gives no DOI.
SOURCES: https://proceedings.mlr.press/v305/wang25c.html ; https://raw.githubusercontent.com/mlresearch/v305/main/assets/wang25c/wang25c.pdf (text read) ;
  https://arxiv.org/html/2509.19712v1 ; arXiv API (2509.19712)
CLAIMS:
  - Chapter2.tex:97 "introduces a benchmark for multi-step cutting with spectral rewards and diffusion policies": OK. The paper
    says "we introduce TopoCut, a comprehensive benchmark for multi-step robotic cutting tasks".
  - Chapter5.tex:77 "explores MPM-based topology changes": OK. It uses Taichi MLS-MPM (extending FluidLab) with damage-driven
    topology discovery.
  - MISMATCH: Chapter5.tex:97 table row "TopoCut & MPM & -- & Synthetic & Yes & ..." -- the "Multi-cam." column says Yes. The
    paper says: the policy consumes particle / "raw depth point clouds", and neither the PMLR PDF nor the arXiv HTML mentions
    any camera or multi-view images. Proposed: Multi-cam. "--". The other columns are OK: there is a knife agent only and no
    robot simulator, the data is simulation-only, and there is no language.

KEY: dikshit2023robochop
STATUS: VERIFIED-UNCHANGED
CHANGES: Formatting only: note -> howpublished = {arXiv preprint arXiv:2307.13159}, and the title is in title case. I found no
  peer-reviewed version. arXiv has only v1 with no journal-ref. It is not in the PMLR CoRL 2023 (v229) list, and neither
  Crossref nor a web search shows a venue. DBLP could not be queried because it blocked the request.
SOURCES: https://export.arxiv.org/api/query?id_list=2307.13159 ; https://arxiv.org/html/2307.13159v1 ;
  https://proceedings.mlr.press/v229/ ; Crossref bibliographic search ; web search
CLAIMS:
  - Chapter5.tex:92 table "RoboChop & -- & -- & Real & -- & -- & -- & Yes": OK. The paper says all experiments are real-world,
    with a single wrist-mounted RealSense D415 and discrete action primitives ("cut, disturb, observe, detect collision and
    push").
  - Borderline: "Language --". The paper says "the text-based prompts are parsed to identify the goal conditions (number of
    slices, cutting style)". This is not language-conditioned learning, so "--" is defensible.

KEY: shi2023robocook
STATUS: CORRECTED
CHANGES: Added volume = {229}, series = {Proceedings of Machine Learning Research} and pages = {642--660}. The booktitle
  was put in policy form (CoRL) and the title in title case. PMLR gives no DOI.
SOURCES: https://proceedings.mlr.press/v229/shi23a.html ; https://proceedings.mlr.press/v229/ ; https://arxiv.org/abs/2306.14447 ;
  https://arxiv.org/html/2306.14447v2
CLAIMS:
  - MISMATCH: Chapter5.tex:77 "RoboCook~\cite{shi2023robocook} studies material interaction using an MPM formulation." -- the
    paper says: RoboCook models tool-object interaction with graph neural networks trained on "20 minutes of real-world
    interaction data per tool". MPM appears only as a baseline ("CEM with a Material Point Method (MPM) simulator"), which the
    paper criticises because "an MPM simulator suffers from a sim2real gap and relies on careful system identification".
    Proposed: "RoboCook~\cite{shi2023robocook} learns tool--dough interaction with graph-network dynamics trained on real
    point clouds."
  - MISMATCH: Chapter5.tex:93 table row "RoboCook & MPM & ..." -- same reason. Proposed: Physics sim. "--", or "GNN (learned)".
    The other columns are OK: "Hybrid" fits real data plus a synthetic dataset generated with the learned dynamics model,
    "Multi-cam. Yes" fits the four RealSense D415 RGB-D cameras, and "Cutting Yes" fits the knife among the 15 tools.

KEY: jamdagni2021robotic
STATUS: CORRECTED
CHANGES: DOI added: 10.1109/ICRA48506.2021.9560761. The organization field was dropped, the booktitle was put in policy
  form (ICRA) and the title in title case. Authors, pages 6607--6613 and year 2021 are confirmed.
SOURCES: https://api.crossref.org/works/10.1109/icra48506.2021.9560761 ; https://faculty.sites.iastate.edu/jia/conference-papers ;
  https://faculty.sites.iastate.edu/jia/files/inline-files/ICRA21.pdf (full text read) ;
  https://www.cs.iastate.edu/news/2021/jamdagni-and-jia-advance-robotic-slicing-icra-2021-paper
CLAIMS:
  - Chapter5.tex:94 "Jamdagni et al.~\cite{jamdagni2021robotic} & FEM & -- & Synthetic & ...": FEM is OK, since forces are
    "predicted using the finite element method".
    MISMATCH: "Data origin: Synthetic". The paper says it ran real experiments with a kitchen knife on an Adept Cobra 800
    robot, slicing potatoes and eggplants, and compared them with FEM simulation (including laser-scanned meshes). Proposed:
    "Hybrid", or "Real".
    MISMATCH (attribution): there are only two authors, so "Jamdagni et al." is wrong. Proposed: "Jamdagni and Jia", or use
    \citet, which prints "Jamdagni and Jia".

KEY: an2025relationship
STATUS: CORRECTED
CHANGES: number = {2} added. Everything else is confirmed: vol. 18, pp. 1244--1258, the title and all four authors. The
  article appeared online on 2024-07-20 and in the February 2025 issue; year 2025 is kept, matching the issue and the
  journal header "(2025) 18:1244-1258".
SOURCES: https://api.crossref.org/works/10.1007/s11947-024-03491-0 ; https://link.springer.com/content/pdf/10.1007/s11947-024-03491-0.pdf (full text read) ;
  https://api.openalex.org/works/doi:10.1007/s11947-024-03491-0
CLAIMS:
  - Chapter5.tex:262 "Strawberry (berry) & 1010 & 2.0e5 & 0.40": OK. Table 2's footnote gives Poisson's ratio = 0.40 and
    density = 1.01 g/cm^3 for strawberry tissue (the paper takes these from Li et al., 2023). Table 2 gives a cortex elastic
    modulus of 0.149--4.108 MPa depending on treatment and ripeness, for example 0.201 MPa for the control at the over-ripe
    stage, so 2.0e5 Pa is consistent.

KEY: jahanbakhshi2020determination
STATUS: CORRECTED
CHANGES: Added volume = {20}, number = {3}, pages = {314--322} and doi = {10.1080/15538362.2019.1633723}. The title is now in
  the published title case: "...Banana Fruit under Quasi-Static Loading in Pressure, Bending, and Shearing Tests". The
  article appeared online on 2019-06-26 and in issue 20(3) in 2020; year 2020 is kept.
SOURCES: https://api.crossref.org/works/10.1080/15538362.2019.1633723 ; https://api.openalex.org/works/doi:10.1080/15538362.2019.1633723 (abstract) ;
  tandfonline.com full text returned 403
CLAIMS:
  - MISMATCH (partly unverifiable): Chapter5.tex:259 "Banana & 960 & 7.0e3 & 0.35" -- the paper's abstract says "The low
    modulus of elasticity of 0.00308 MPa in banana fruit", which is 3.08e3 Pa, not 7.0e3 Pa. Density and Poisson's ratio are
    not in the abstract, and the full text was not accessible. Proposed: E = $3.1\times10^3$ Pa, or state that E was adjusted.
    Check the full text for 960 kg/m^3 and 0.35.

KEY: kabas2015determination
STATUS: VERIFIED-UNCHANGED
CHANGES: none. Authors (Crossref: Onder Kabas, Valentin Vladut), the title, vol. 18(11), pp. 2584--2592, year 2015 and the
  DOI are all confirmed. Only the title was put in title case.
SOURCES: https://api.crossref.org/works/10.1080/10942912.2014.994069 ; https://api.openalex.org/works/doi:10.1080/10942912.2014.994069 (abstract) ;
  https://avesis.akdeniz.edu.tr/yayin/2dcdf938-0e89-45ac-a083-4e10e98af950/ ; tandfonline full text returned 403
CLAIMS:
  - Chapter5.tex:261 "Peach & 968 & 8.9e5 & 0.29": E and Poisson's ratio are OK. The abstract gives "Poisson's ratio ... 0.29"
    and "modulus of elasticity ... 0.89 N mm^-2", which is 8.9e5 Pa. The density of 968 kg/m^3 could not be verified because
    it is not in the abstract and the full text was blocked.

KEY: kheiralipour2008some
STATUS: CORRECTED
CHANGES: author "Kheiralipour, K and ... and Sahraroo, A Sahraroo and Rajabipour, A and Jafari, A and others" ->
  "Kheiralipour, K. and Tabatabaeefar, A. and Mobli, H. and Rafiee, S. and Sahraroo, A. and Rajabipour, A. and Jafari, A.".
  The printed byline has exactly these seven authors, so "and others" was wrong. "A. Sahraroo Sahraroo" is a Crossref data
  error; the paper prints "A. Sahraroo". The paper gives initials only. The title is now in title case.
SOURCES: https://api.crossref.org/works/10.3923/pjn.2008.667.672 ; https://pjnonline.org/pjn/article/view/712 ;
  https://pjnonline.org/pjn/article/download/712/710/1451 (scanned PDF; pages 667, 670 and 671 read)
CLAIMS:
  - MISMATCH: Chapter5.tex:256 "Apple & 850 & 3.0e6 & 0.33". The paper's Table 1 gives:
    - true density 837 kg/m^3 (Redspar) and 827.91 kg/m^3 (Delbarstival); 850 is not reported;
    - modulus of elasticity 2.53 MPa (Redspar) and 1.77 MPa (Delbarstival) as means, with per-fruit ranges of 1.87--3.69 and
      0.98--4.44 MPa; 3.0 MPa is not reported;
    - no Poisson's ratio at all.
    Proposed: rho = 830--840, E = $2.5\times10^6$, or mark which values were assumed or adjusted (see the overall
    recommendation at the end).

KEY: kurniawati2023determination
STATUS: VERIFIED-UNCHANGED
CHANGES: Formatting only: the title is in title case with {Siamese}, {Citrus nobilis} and {Microcarpa} protected, and the url
  was dropped. Pages 67--76 are confirmed from the article's running headers ("3(2); 67-76"). Note that the article is
  written in Indonesian, with the Indonesian title "Penentuan sifat fisik jeruk siam ..." and the English title cited here.
  The publisher site is currently broken (PHP error), so it was read through the Wayback Machine.
SOURCES: https://api.crossref.org/works/10.31764/jafp.v3i2.20423 ; https://api.openalex.org/works/doi:10.31764/jafp.v3i2.20423 ;
  http://web.archive.org/web/20250717210209/https://journal.ummat.ac.id/index.php/JAFP/article/view/20423 ;
  http://web.archive.org/web/20250620083201id_/https://journal.ummat.ac.id/index.php/JAFP/article/download/20423/pdf (full text read)
CLAIMS:
  - MISMATCH: Chapter5.tex:260 "Orange & 1020 & 5.84e5 & 0.24". The density is OK ("density is 1.02 g/cm^3"). The paper
    measures only sphericity, roundness, volume and density, and reports no Young's modulus and no Poisson's ratio. Proposed:
    give the source of E = 5.84e5 and nu = 0.24 separately, or mark them as assumed.

KEY: mattos2005heat
STATUS: CORRECTED
CHANGES: author "Breidt Jr, F" -> "Breidt, Jr., F." (the BibTeX Jr form) and "Passos, Frederico V" -> "Passos, F. V.". The paper
  and Crossref print only "F.V."; "Frederico" is not in any source I opened. All other names are given as the printed
  initials. Pages E324--E330 are confirmed from the page headers, and vol. 70(5), year 2005 and the DOI are confirmed. The
  title is now in title case and the url was dropped.
SOURCES: https://api.crossref.org/works/10.1111/j.1365-2621.2005.tb09972.x ; https://ars.usda.gov/ARSUserFiles/60701000/Pickle%20Pubs/p335.pdf (full text read) ;
  onlinelibrary.wiley.com returned 403
CLAIMS:
  - MISMATCH: Chapter5.tex:257 "Cucumber & 980 & 1.0e6 & 0.35". The density is OK: Table 1 gives "Density 980 kg/m3" and
    Table 2 gives 0.98 g/mL. This is a heat-transfer and microbial-kinetics paper, and it reports no Young's modulus and no
    Poisson's ratio; it lists only thermal conductivity, specific heat and similar quantities. Proposed: cite a mechanical
    source for E and nu, or mark them as assumed.

KEY: nourain2005firmness
STATUS: VERIFIED-UNCHANGED
CHANGES: Formatting only: the title is in title case. All five authors, J. Zhejiang Univ. Sci. B 6(6):483--490, year 2005
  and the DOI are confirmed. PMID 15909331, PMC1389877.
SOURCES: https://api.crossref.org/works/10.1631/jzus.2005.B0483 ; https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?db=pubmed&id=15909331 ;
  https://pmc.ncbi.nlm.nih.gov/articles/PMC1389877/ (full text, Table 1 and Fig. 6 read)
CLAIMS:
  - Chapter5.tex:258 "Melon & 1100 & 1.2e6 & 0.45":
    - Density 1100 is consistent: Table 1 gives per-fruit densities of 990--1648 kg/m^3, mostly 1100--1270.
    - E = 1.2 MPa is consistent: Fig. 6 gives a static elastic modulus of 0.84--1.54 MPa from the outer to the inner flesh.
      The FE-estimated dynamic modulus is 2.0--8.2 MPa.
    - MISMATCH: no Poisson's ratio appears anywhere in the text or tables, so 0.45 is not from this paper. Proposed: mark nu
      as assumed.

---

## Overall recommendation for Table tab:cut_material_properties (Chapter5.tex:248-262)

The cited sources support the following values:

| Food | Supported by the cited source | Not reported in the cited source |
|---|---|---|
| Apple | none exactly (rho about 830--840; E 1.8--2.5 MPa as means) | Poisson's ratio |
| Cucumber | rho | E, Poisson's ratio |
| Melon | rho, E (approximately) | Poisson's ratio |
| Banana | none of the three verified (E in the abstract is 3.08 kPa, not 7.0 kPa) | rho and Poisson's ratio not checkable in the full text |
| Orange | rho | E, Poisson's ratio |
| Peach | E, Poisson's ratio | rho not checkable |
| Strawberry | all three | -- |

Minimal fix: add to the caption "Values not reported in the cited source (marked $^\dagger$) are assumed." Then mark Apple
E/nu, Cucumber E/nu, Melon nu, Orange E/nu, Banana E (or change it to 3.1e3), and Apple rho (or change it to 830).

The text at Chapter5.tex:216 ("initialized from published food-engineering references") is then accurate. Note that one of
those references (Mattos et al.) is a heat-transfer paper and supplies only a density.
