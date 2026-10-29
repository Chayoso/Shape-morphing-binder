# Batch D report (neural shape deformation, differentiable rendering, Gaussian splatting, metrics; 25 entries)

Notes on method
- DBLP (dblp.org, dblp.uni-trier.de, dblp.dagstuhl.de) returned a bot-check page for every request, and ACM DL returned 403. ACM article numbers were therefore checked in the authors' versions of the papers (the "ACM Reference Format" line or running headers), in the authors' own BibTeX, or in an archived copy of the ACM DL page (web.archive.org).
- CVPR/ICCV page numbers: the CVF open-access PDFs state that "the final published version of the proceedings is available on IEEE Xplore". The output therefore gives the IEEE Xplore pages, which match the IEEE DOI it also gives. For CVPR 2019 and 2021 the IEEE pages are 5 lower than the CVF open-access pages (listed per entry below). For CVPR 2018, 2024 and 2025 the two agree. The input already used IEEE pages for NISE and 4Deform.
- Every Crossref record was read from https://api.crossref.org/works/<DOI>, and for the Wiley article number also from /transform/application/vnd.crossref.unixsd+xml.

---

KEY: nicolet2021largesteps
STATUS: VERIFIED-UNCHANGED
CHANGES: none
SOURCES: https://api.crossref.org/works/10.1145/3478513.3480501 ; DOI content negotiation (doi.org, BibTeX); authors' version https://rgl.s3.eu-central-1.amazonaws.com/media/papers/Nicolet2021Large.pdf ("ACM Trans. Graph. 40, 6, Article 248 (December 2021), 13 pages")
CLAIMS: OK. Chapter4.tex:470 cites it as "render-only mesh optimization with Large Steps", which is right: it is inverse rendering of mesh geometry with a smoothness preconditioner (Crossref abstract).

KEY: novello2023nise
STATUS: VERIFIED-UNCHANGED (formatting only: braces in booktitle)
CHANGES: none factual. IEEE pages 14233--14243 match the IEEE DOI (CVF open-access pages are 14279--14289). Crossref spells the second author "Da Silva", CVF spells him "da Silva"; kept "da Silva".
SOURCES: https://api.crossref.org/works/10.1109/ICCV51070.2023.01313 ; https://openaccess.thecvf.com/content/ICCV2023/html/Novello_Neural_Implicit_Surface_Evolution_ICCV_2023_paper.html
CLAIMS: OK. Chapter4.tex:467 calls it a "neural implicit morph", and the abstract describes level-set evolution of neural implicit surfaces including "interpolations of initial conditions". Appendix.tex:613 says the volume-preserving morph and NISE "run from the former's code": confirmed. The Buonomo et al. repository README (github.com/camillebnm/neural_shape_morphing) says it builds on NISE's code and offers `-p nise` to run the NISE implementation. The "signed-distance fitting per shape" footnote matches that repository's `sdf_train.py`.

KEY: sang2025fourdeform
STATUS: VERIFIED-UNCHANGED (formatting only)
CHANGES: none factual (IEEE pages = CVF pages = 6542--6551)
SOURCES: https://api.crossref.org/works/10.1109/CVPR52734.2025.00613 ; https://openaccess.thecvf.com/content/CVPR2025/html/Sang_4Deform_Neural_Surface_Deformation_for_Robust_Shape_Interpolation_CVPR_2025_paper.html ; arXiv:2502.20208 (PDF read)
CLAIMS: OK, with a note. The paper's input is "pairs of point clouds and estimated sparse, noisy correspondences". In the paper's words, "our method begins by establishing correspondences through a matching module" (a functional-map correspondence block). So "since a sphere-to-target pair has no correspondences, their matching weight is zero" (Appendix.tex:614-615) correctly describes ISD's and 4Deform's matching loss. However, Appendix.tex:684 "4Deform without correspondences ends at a silhouette IoU of 0.50--0.76" invites a reviewer to say that 4Deform normally estimates its own correspondences. NOTE (not a mismatch): consider adding a few words to Appendix.tex:615, e.g. "... their matching weight is zero (4Deform's correspondence block is not run) ...", but only if that is what was done.

KEY: sang2025isd
STATUS: VERIFIED-UNCHANGED (formatting only: booktitle form; eprint/archivePrefix dropped)
CHANGES: none factual
SOURCES: https://arxiv.org/abs/2501.14038 (comments "ICLR 2025"); https://proceedings.iclr.cc/paper_files/paper/2025/hash/208b3bfaf8b6685e8d88fd7aaaaa1a9d-Abstract-Conference.html (authors, "ICLR 2025"); OpenReview search (forum 4XOQ0f4a66, "ICLR 2025 Poster"); arXiv PDF read
CLAIMS: OK. The paper recovers the deformation "based on sparse correspondences" (conclusion) and has an appendix section "No correspondences" (A.12), so the matching-weight-zero arm and the radial-correspondence arm describe it fairly. The authors' code was used (experiments.md B14: Sangluisme/Implicit-surf-Deformation), consistent with Appendix.tex:614.

KEY: shen2023flexicubes
STATUS: CORRECTED
CHANGES: pages 1--16 -> 37:1--37:16 (article 37, 16 pages)
SOURCES: https://api.crossref.org/works/10.1145/3592430 ; authors' page https://research.nvidia.com/labs/toronto-ai/flexicubes/ (BibTeX: articleno 37, numpages 16, vol 42, no 4) ; arXiv:2308.05371
CLAIMS: OK. "Render-only mesh optimization ... FlexiCubes" matches: it is an isosurface representation for gradient-based mesh optimization, demonstrated in multiview reconstruction among other uses.

KEY: buonomo2025vpnsm
STATUS: VERIFIED-UNCHANGED
CHANGES: none. The article number e70196 is confirmed in Crossref's XML (`<item_number item_number_type="article_number">e70196</item_number>`).
SOURCES: https://api.crossref.org/works/10.1111/cgf.70196 and .../transform/application/vnd.crossref.unixsd+xml ; HAL API record hal-05113996 (CGF 44(5), 2025-08) ; https://github.com/camillebnm/neural_shape_morphing README ("Symposium on Geometry Processing 2025 (Computer Graphics Forum)")
CLAIMS: OK. "Neural implicit morphs of Buonomo et al." fits a level-set representation advected by a neural velocity field. The abstract says it works "without landmark correspondences", consistent with running it from a sphere.

KEY: feydy2019sinkhorn
STATUS: VERIFIED-UNCHANGED (formatting only: booktitle without the ordinal "Twenty-Second", per policy)
CHANGES: none factual. PMLR's metadata spells "Alain Trouve", but the paper PDF has "Alain Trouvé"; kept Trouv{\'e}.
SOURCES: https://proceedings.mlr.press/v89/feydy19a.html (pages 2681-2690, vol 89, AISTATS 2019) ; PDF http://proceedings.mlr.press/v89/feydy19a/feydy19a.pdf
CLAIMS: OK. Chapter4.tex:140 and Appendix.tex:257-262 give S_eps = OT_eps(a,b) - 1/2 OT_eps(a,a) - 1/2 OT_eps(b,b), with an entropic KL(pi | a (x) b) term. This matches the paper's eq. (1) and its debiasing formula (3).

KEY: yu2020pcgrad
STATUS: CORRECTED
CHANGES: added volume 33 and pages 5824--5836; booktitle "Advances in Neural Information Processing Systems 33 (NeurIPS 2020)" -> "Advances in Neural Information Processing Systems ({NeurIPS})"
SOURCES: https://proceedings.neurips.cc/paper_files/paper/2020/hash/3fe78a8acf5fda99de95303940a2420c-Abstract.html ; .../3fe78a8acf5fda99de95303940a2420c-Bibtex.bib
CLAIMS: OK. PCGrad "projects a task's gradient onto the normal plane of the gradient of any other task that has a conflicting gradient". The equation at Chapter4.tex:186-189 is this projection applied to the render gradient only (a one-sided form). Optional precision: "...when the two conflict, as in PCGrad~\citep{yu2020pcgrad} but one-sided".

KEY: cheng2021boundaryiou
STATUS: CORRECTED
CHANGES: pages 15334--15342 (CVF open access) -> 15329--15337 (IEEE Xplore, matching the DOI); doi added 10.1109/CVPR46437.2021.01508
SOURCES: https://api.crossref.org/works?query.bibliographic=... -> 10.1109/cvpr46437.2021.01508 ; https://openaccess.thecvf.com/content/CVPR2021/html/Cheng_Boundary_IoU_..._CVPR_2021_paper.html ; CVF PDF (author "Piotr Dollár")
CLAIMS: OK. Appendix.tex:429 "Boundary IoU at 2 % of the image diagonal" is the paper's own setting: d = 2 % of the image diagonal for COCO/ADE20k-like resolutions (Sec. 3 and the d ablation).

KEY: knapitsch2017tanks
STATUS: VERIFIED-UNCHANGED
CHANGES: none (Article 78, 13 pages)
SOURCES: https://api.crossref.org/works/10.1145/3072959.3073599 (vol 36, issue 4, pages 1-13) ; archived ACM DL page http://web.archive.org/web/20250616230956/https://dl.acm.org/doi/10.1145/3072959.3073599 ("Article No.: 78", "Pages 1 - 13", "Volume 36, Issue 4") ; https://www.tanksandtemples.org/ BibTeX
CLAIMS: OK. Appendix.tex:431: F-score at a distance threshold tau (precision/recall), as defined by the benchmark.

KEY: mescheder2019occupancy
STATUS: CORRECTED
CHANGES: pages 4460--4470 (CVF) -> 4455--4465 (IEEE Xplore); doi added 10.1109/CVPR.2019.00459
SOURCES: Crossref search -> 10.1109/cvpr.2019.00459 (pages 4455-4465) ; https://openaccess.thecvf.com/content_CVPR_2019/html/Mescheder_Occupancy_Networks_..._CVPR_2019_paper.html
CLAIMS: OK. "Chamfer-L1 and normal consistency" are among the metrics Occupancy Networks defines.

KEY: tatarchenko2019singleview
STATUS: CORRECTED
CHANGES: pages 3405--3414 (CVF) -> 3400--3409 (IEEE Xplore); doi added 10.1109/CVPR.2019.00352
SOURCES: Crossref search -> 10.1109/cvpr.2019.00352 ; https://openaccess.thecvf.com/content_CVPR_2019/html/Tatarchenko_What_Do_Single-View_3D_Reconstruction_Networks_Learn_CVPR_2019_paper.html
CLAIMS: OK (the paper argues for the F-score at a distance threshold).

KEY: wang2004ssim
STATUS: CORRECTED
CHANGES: authors "Bovik, Alan C." -> "Bovik, Alan Conrad" and "Sheikh, Hamid R." -> "Sheikh, Hamid Rahim" (full given names as in the published byline; the printed initials are unchanged)
SOURCES: https://api.crossref.org/works/10.1109/TIP.2003.819861 ; published reprint https://www.cns.nyu.edu/pub/eero/wang03-reprint.pdf (byline, vol 13 no 4 April 2004, p. 600)
CLAIMS: OK

KEY: zhang2018lpips
STATUS: CORRECTED
CHANGES: doi added 10.1109/CVPR.2018.00068 (pages 586--595 identical in IEEE and CVF)
SOURCES: Crossref search -> 10.1109/cvpr.2018.00068 ("2018 IEEE/CVF Conference on Computer Vision and Pattern Recognition", 586-595) ; https://openaccess.thecvf.com/content_cvpr_2018/html/Zhang_The_Unreasonable_Effectiveness_CVPR_2018_paper.html
CLAIMS: OK

KEY: Yifan_2019
STATUS: CORRECTED
CHANGES: first author "Wang, Yifan" -> "Yifan, Wang". The published record (Crossref: given "Wang", family "Yifan"), the arXiv metadata ("Yifan, Wang"), and the authors' own lab BibTeX (IGL ETH, github.com/yifita/DSS: "Yifan, Wang") all treat Yifan as the family name, so the in-text citation becomes "Yifan et al. (2019)", consistent with the key. pages 1--14 -> 230:1--230:14 (article 230). Title capitalised to Title Case. Dropped publisher/month/url/issn/articleno.
SOURCES: https://api.crossref.org/works/10.1145/3355089.3356513 ; https://arxiv.org/abs/1906.04173 ; https://igl.ethz.ch/publications/igl-bibtex.php#Yifan:DSS:2019 ; https://raw.githubusercontent.com/yifita/DSS/master/README.md ; https://igl.ethz.ch/projects/differentiable-surface-splatting/DSS-2019-SA-Yifan-etal.pdf (running heads "230:2" ... "230:14")
CLAIMS: MISMATCH (see the combined item under laine2020modular)

KEY: laine2020modular
STATUS: CORRECTED
CHANGES: pages 1--14 -> 194:1--194:14 (article 194); doi added 10.1145/3414685.3417861; title in Title Case; publisher dropped
SOURCES: https://api.crossref.org/works/10.1145/3414685.3417861 ; arXiv:2011.03277 PDF ("ACM Trans. Graph. 39, 6, Article 194 (December 2020), 14 pages")
CLAIMS: MISMATCH: Chapter2.tex:71 "Early attempts using meshes or implicit representations~\citep{Yifan_2019,laine2020modular,wu2025pbr} struggled respectively with topology changes or computational complexity."
-- the papers say:
   (i) none of the three couples rendering with physics simulation (the subsection is "Differentiable Physics Coupled with Rendering"). They are differentiable renderers: DSS for point clouds, nvdiffrast for triangle-mesh rasterization, and PBR-NeRF for NeRF-based inverse rendering of materials and illumination, where "physics-based" means physically based rendering, not dynamics.
   (ii) DSS is point-based (neither mesh nor implicit), and its abstract says topology changes "are accurately and robustly handled without requiring explicit connectivity", which is the opposite of "struggled with topology changes".
   (iii) PBR-NeRF is a 2025 paper, not an "early attempt".
   (iv) "respectively" pairs three citations with two problems.
-- proposed: "Differentiable renderers for points~\citep{Yifan_2019}, meshes~\citep{laine2020modular} and neural fields~\citep{wu2025pbr} recover geometry and appearance from images but are not coupled to a physics simulation."

KEY: yang2021geometry
STATUS: VERIFIED-UNCHANGED (formatting only: whole-title braces, editor, publisher and url removed; booktitle form)
CHANGES: none factual
SOURCES: https://proceedings.neurips.cc/paper_files/paper/2021/file/bd686fd640be98efaae0091fa301e613-Bibtex.bib ; .../hash/bd686fd640be98efaae0091fa301e613-Abstract.html
CLAIMS: OK. The abstract states that "deformation and filtering can be done with neural fields", which supports "new routes to geometric editing" (Chapter2.tex:34).

KEY: wu2025pbr
STATUS: CORRECTED
CHANGES: author "Broedermann, Tim" -> "Br{\"o}dermann, Tim". The paper PDF and IEEE/Crossref have the umlaut; CVF's BibTeX and arXiv transliterate it. doi added 10.1109/CVPR52734.2025.01025. Booktitle given the ({CVPR}) form.
SOURCES: https://openaccess.thecvf.com/content/CVPR2025/html/Wu_PBR-NeRF_..._CVPR_2025_paper.html (pages 10974-10984) and its PDF (byline "Tim Brödermann") ; Crossref search -> 10.1109/cvpr52734.2025.01025 ; https://arxiv.org/abs/2412.09680
CLAIMS: MISMATCH at Chapter2.tex:71 (see laine2020modular): PBR-NeRF is a 2025 NeRF inverse-rendering method for geometry, materials and illumination with physically-based-rendering priors. It is not an early physics-plus-rendering attempt.

KEY: xie2024physgaussian
STATUS: CORRECTED
CHANGES: doi added 10.1109/CVPR52733.2024.00420; title capitalisation "Physics-integrated {3D} gaussians for generative dynamics" -> "Physics-Integrated {3D} {Gaussians} for Generative Dynamics" (the unbraced "gaussians" would have printed in lower case); booktitle form
SOURCES: https://openaccess.thecvf.com/content/CVPR2024/html/Xie_PhysGaussian_..._CVPR_2024_paper.html (4389-4398) ; Crossref search -> 10.1109/cvpr52733.2024.00420 (4389-4398)
CLAIMS: OK. Chapter2.tex:66: PhysGaussian drives 3DGS with a forward MPM simulation and has no inverse path. The OmniPhysGS paper confirms that it "requires manually specifying material properties for each particle".

KEY: lin2025omniphysgs
STATUS: VERIFIED-UNCHANGED (formatting only: booktitle form, url removed)
CHANGES: none factual. ICLR's proceedings page and OpenReview write "Yadong MU", arXiv writes "Mu"; kept "Mu".
SOURCES: https://proceedings.iclr.cc/paper_files/paper/2025/hash/af31604708f3e44b4de9fdfa6dcaa9d1-Abstract-Conference.html ; https://arxiv.org/abs/2501.18982 (PDF read) ; OpenReview search (forum 9HZtP6I5lv, "ICLR 2025 Poster")
CLAIMS: MISMATCH: Chapter2.tex:66 "Conversely, OmniPhysGS~\citep{lin2025omniphysgs} replaces first-principles dynamics with \emph{neural surrogates}, gaining efficiency at the potential cost of physical-law preservation and generalization to unseen materials."
-- the paper says it keeps a differentiable MPM simulator ("we adopt MPM to simulate the dynamics of 3DGS"). It learns only the per-Gaussian constitutive model: a network chooses, via hardmax, one of 12 expert-designed constitutive models per neighbourhood and predicts its parameters, under score-distillation guidance from a video diffusion model. The hardmax "enforces strict adherence to the physical laws". The paper names "time-intensive" per-scene SDS optimisation as a limitation, so "gaining efficiency" is not supported. Its own ablation shows that a pure neural surrogate (an MLP predicting stress) fails.
-- proposed: "Conversely, OmniPhysGS~\citep{lin2025omniphysgs} keeps MPM but \emph{learns} the constitutive model of each Gaussian from a video-diffusion prior, at the potential cost of physical-law preservation and generalization to unseen materials."

KEY: zhang2024physdreamerphysicsbasedinteraction3d
STATUS: PUBLISHED-VERSION-SUBSTITUTED
CHANGES: @misc arXiv:2404.13026 -> @inproceedings ECCV 2024: booktitle "Proceedings of the European Conference on Computer Vision ({ECCV})", pages 388--406, year 2024 (first online 20 October 2024; LNCS vol. 15060, not printed per policy), doi 10.1007/978-3-031-72627-9_22. Authors unchanged (8, same order); names reformatted Last, First.
SOURCES: https://api.crossref.org/works/10.1007/978-3-031-72627-9_22 ; https://link.springer.com/chapter/10.1007/978-3-031-72627-9_22 (meta tags: pages 388-406, LNCS vol 15060, published 20 October 2024) ; https://arxiv.org/abs/2404.13026 (PDF read)
CLAIMS: OK. Chapter2.tex:71: the paper "models the physical material field with a neural field" (triplanes plus MLP) and uses "differentiable MPM" and differentiable rendering of 3D Gaussians. It optimises the material field and the initial velocity field to match a video produced by an image-to-video model.

KEY: wu20244dgaussiansplattingrealtime
STATUS: PUBLISHED-VERSION-SUBSTITUTED
CHANGES: @misc arXiv:2310.08528 -> @inproceedings CVPR 2024, pages 20310--20320, doi 10.1109/CVPR52733.2024.01920. Year 2024 unchanged; authors unchanged (9).
SOURCES: https://openaccess.thecvf.com/content/CVPR2024/html/Wu_4D_Gaussian_Splatting_for_Real-Time_Dynamic_Scene_Rendering_CVPR_2024_paper.html ; Crossref search -> 10.1109/cvpr52733.2024.01920 ; https://arxiv.org/abs/2310.08528 (comments "CVPR 2024")
CLAIMS: OK. Chapter2.tex:64 "learn complex deformation fields ... to capture non-rigid motion".

KEY: yang2023deformable3dgaussianshighfidelity
STATUS: PUBLISHED-VERSION-SUBSTITUTED
CHANGES: @misc arXiv:2309.13101 (2023) -> @inproceedings CVPR 2024, pages 20331--20341, doi 10.1109/CVPR52733.2024.01922. year 2023 -> 2024, so the in-text citation becomes "Yang et al., 2024" (the key keeps "2023").
SOURCES: https://openaccess.thecvf.com/content/CVPR2024/html/Yang_Deformable_3D_Gaussians_for_High-Fidelity_Monocular_Dynamic_Scene_Reconstruction_CVPR_2024_paper.html ; Crossref search -> 10.1109/cvpr52733.2024.01922 ; https://arxiv.org/abs/2309.13101
CLAIMS: OK

KEY: keetha2024splatam
STATUS: CORRECTED
CHANGES: title "{SplaTAM}: Splat track \& map {3D} gaussians ..." -> "{SplaTAM}: Splat, Track \& Map {3D} {Gaussians} for Dense {RGB-D} {SLAM}". The published PDF and IEEE have the comma after "Splat" (CVF's BibTeX drops it), and the unbraced "gaussians" would have printed in lower case. doi added 10.1109/CVPR52733.2024.02018. Booktitle form.
SOURCES: https://openaccess.thecvf.com/content/CVPR2024/html/Keetha_SplaTAM_Splat_Track__Map_3D_Gaussians_for_Dense_RGB-D_CVPR_2024_paper.html and its PDF (title line) ; Crossref search -> 10.1109/cvpr52733.2024.02018 (21357-21366)
CLAIMS: OK. "Targets SLAM for online, real-time reconstruction": the paper describes "a simple online tracking and mapping system" and "dense photometric loss for SLAM in real-time". Minor wording point, not a citation error: in "Yet these motions are fundamentally kinematic", SplaTAM reconstructs static scenes and models no motion, so "these motions" refers only to 4DGS and Deformable 3DGS.

KEY: sharpdeep
STATUS: VERIFIED-UNCHANGED (formatting only: names Last, First; title braces removed; pages with --; url and issn dropped)
CHANGES: none factual
SOURCES: https://api.crossref.org/works/10.1016/j.jcp.2018.05.031 (Gibou, Hyde, Fedkiw; JCP 380, 442-463, March 2019)
CLAIMS: OK. A review of sharp-interface and deep-learning methods for multiphase flows, which supports "the growing interplay between learning and simulation".
