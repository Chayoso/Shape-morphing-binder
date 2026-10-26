# Batch C report (shape morphing / interpolation / fluid control / optimal transport; 24 keys)

Access notes: dblp.org and its mirrors are behind an Anubis bot wall and dl.acm.org behind Cloudflare (403),
so DBLP records were read through the official DBLP SPARQL endpoint (https://sparql.dblp.org/sparql, query by
DOI or by record key), and ACM DL pages through Wayback Machine snapshots of dl.acm.org where needed.
Crossref API = https://api.crossref.org/works/<DOI>. Abstracts came from Crossref, OpenAlex (inverted index of
the same abstract), the publisher page (Springer), or the paper PDF.

---

KEY: 1988-SpacetimeConstraints
STATUS: CORRECTED
CHANGES: entry type @article -> @inproceedings (policy: pre-2002 SIGGRAPH paper cited as the SIGGRAPH proceedings);
         journal "ACM SIGGRAPH Computer Graphics" 22(4) -> booktitle "Proceedings of the Annual Conference on Computer
         Graphics and Interactive Techniques ({SIGGRAPH})"; doi 10.1145/378456.378507 (Computer Graphics reprint DOI)
         -> 10.1145/54852.378507 (proceedings DOI). Authors, title, pages 159--168, year 1988 unchanged.
SOURCES: https://api.crossref.org/works/10.1145/54852.378507 (proceedings-article, "Proceedings of the 15th annual
         conference on Computer graphics and interactive techniques", pp. 159-168, 1988, Witkin/Kass);
         https://api.crossref.org/works/10.1145/378456.378507; DBLP conf/siggraph/WitkinK88 (SPARQL, pp. 159-168).
CLAIMS: OK (Chapter2.tex:39 "closely related to spatiotemporal optimization" -- abstract: animation posed as a
        constrained optimization over the motion, satisfying Newton's laws).

KEY: 2000-ARAPShapeInterpolation
STATUS: VERIFIED-UNCHANGED
CHANGES: formatting only (booktitle normalised to the policy SIGGRAPH form; series/publisher/isbn/numpages dropped).
SOURCES: https://api.crossref.org/works/10.1145/344779.344859 (Proceedings of the 27th annual conference ... SIGGRAPH '00,
         pp. 157-164); DBLP conf/siggraph/AlexaCL00 (SPARQL); abstract via OpenAlex (doi:10.1145/344779.344859).
CLAIMS: OK. Chapter2.tex:32 (deforms triangle/tetrahedral meshes via energy minimization for interpolation: abstract --
        2D/3D shapes decomposed into isomorphic simplicial complexes, paths defined by minimising an error functional
        of the per-simplex affine maps). Chapter4.tex:468 and Appendix.tex:616 OK. (Nit, not a mismatch: the
        "ARAP energy" as a named deformation energy is usually credited to Sorkine and Alexa 2007; Alexa et al. 2000
        is the ARAP *interpolation* paper, which is what the sentence describes.)

KEY: 2004-ActualMorphing
STATUS: CORRECTED
CHANGES: booktitle "Proceedings of the Ninth ACM Symposium on Solid Modeling and Applications" -> "Proceedings of the
         {ACM} Symposium on Solid Modeling and Applications ({SM})"; the input carried only an ACM "10.5555/1217875.1217924"
         pseudo-DOI in url (doi.org returns 404 for it) -> added the real registered DOI 10.2312/sm.20041407 (DataCite,
         The Eurographics Association, resolves to diglib.eg.org/handle/10.2312/sm20041407). Authors, title, pages
         309--314, year 2004 unchanged. Note: the Eurographics DL / DataCite record titles the item "Actual Morphing:
         A Physical-Based Approach for Blending Two 2D/3D Shapes" (filed under "Posters Session"); the paper itself
         (author PDF with the Eurographics copyright line), DBLP and the ACM DL all print "Actual Morphing: A
         Physics-Based Approach to Blending", which is kept.
SOURCES: https://cg.cs.tsinghua.edu.cn/~shimin/pdf/acm_sm.pdf (paper: "ACM Symposium on Solid Modeling and Applications
         (2004), G. Elber, N. Patrikalakis, P. Brunet (Editors)", authors Shi-Min Hu, Chen-Feng Li, Hui Zhang, abstract);
         DBLP conf/sma/HuLZ04 (SPARQL: pp. 309-314, 2004, ee dl.acm.org/citation.cfm?id=1217924);
         https://api.datacite.org/dois/10.2312/sm.20041407 (pp. 309-314, 2004); https://doi.org/10.2312/sm.20041407 (302).
CLAIMS: MISMATCH (minor): Chapter2.tex:32 "Subsequent extensions have emphasized physical realism~\citep{2004-ActualMorphing,
        2020-HamiltonianShapeInterpolation} to improve accuracy" -- paper says: it chooses the morphing path by a
        "quasi-physical law" (minimum strain energy, FEM shape functions) for plausibility; it is not an extension of
        ARAP (it cites ARAP as prior path-interpolation work) and claims no accuracy gain. Hamiltonian (2020) likewise
        claims physical plausibility and robustness to noisy real scans, not accuracy. -- proposed: "Subsequent methods
        have pursued physical plausibility~\citep{...}, e.g., ..." (drop "to improve accuracy").

KEY: 2004-FluidControlAdjointMethod
STATUS: VERIFIED-UNCHANGED
CHANGES: formatting only.
SOURCES: https://api.crossref.org/works/10.1145/1015706.1015744 (TOG 23(3):449-456, 2004, McNamara/Treuille/Popović/Stam,
         abstract); DBLP journals/tog/McNamaraTPS04 (SPARQL).
CLAIMS: OK. Chapter2.tex:41 (keyframe objective + adjoint gradients: abstract = gradient-based control with adjoint
        derivatives; the keyframe formulation is confirmed by Chen et al. 2024 Sec. 2.2 "It was refined in [McNamara et
        al. 2004], which employed the adjoint method" for keyframe control). Chapter4.tex:455 OK (control forces).

KEY: 2005-PoissonShapeInterpolation
STATUS: VERIFIED-UNCHANGED
CHANGES: formatting only (booktitle "Proceedings of the 2005 ACM Symposium on Solid and Physical Modeling" ->
         "Proceedings of the {ACM} Symposium on Solid and Physical Modeling ({SPM})").
SOURCES: https://api.crossref.org/works/10.1145/1060244.1060274 (pp. 267-274, 2005-06-13); DBLP conf/sma/XuZWB05 (SPARQL);
         abstract via OpenAlex. (A journal version exists, Graphical Models 2006, DBLP journals/cvgip/XuZWB06; the SPM
         paper is itself peer reviewed, so no substitution.)
CLAIMS: OK. Chapter2.tex:32 (Poisson equation on the domain mesh, avoids shrinkage: abstract says exactly this).
        Appendix.tex:616 OK (gradient-domain interpolation).

KEY: 2012-DeformableObjectsAlive
STATUS: VERIFIED-UNCHANGED
CHANGES: formatting only (articleno 69 / numpages 9 -> pages = {69:1--69:9}).
SOURCES: https://api.crossref.org/works/10.1145/2185520.2185565 (TOG 31(4), 9 pp., byline incl. "Robert Sumner",
         "Markus Gross"); DBLP journals/tog/CorosMTSSG12 (SPARQL: 69:1-69:9); paper PDF
         https://studios.disneyresearch.com/wp-content/uploads/2019/03/Deformable-Objects-Alive.pdf.
CLAIMS: MISMATCH (minor): Chapter2.tex:42 "optimized nonlinear, time-dependent objectives with SQP, addressing issues such
        as rest-state adaptation during continuous control of mesh-based characters" -- paper says: rest-shape
        adaptation is the control mechanism itself (characters move only by internal forces from adapted rest shapes),
        not an issue the method addresses; SQP is solved every time step for time-dependent motion goals (this part is
        right). -- proposed: "... with SQP, using rest-shape adaptation as the control of deformable (curve, shell and
        solid) characters".

KEY: 2013-DataDrivenShapeMorphing
STATUS: VERIFIED-UNCHANGED
CHANGES: formatting only (author names to Last, First). Number kept as 2 (DBLP); Wiley labels the issue "2pt4" (Issue 2, Part 4).
SOURCES: https://api.crossref.org/works/10.1111/cgf.12065 (CGF 32(2pt4):449-457, 2013, abstract); DBLP journals/cgf/GaoLHH13 (SPARQL).
CLAIMS: MISMATCH (minor): Chapter2.tex:34 "constructed local shape spaces from model databases and posed deformation as
        global optimization" -- paper says: morphing is cast as a global optimization that finds a minimal-distance path
        through the local shape spaces, then an extended ARAP interpolation follows it. -- proposed: "posed morphing as a
        global shortest-path optimization".

KEY: 2020-HamiltonianShapeInterpolation
STATUS: VERIFIED-UNCHANGED
CHANGES: formatting only (booktitle -> "Proceedings of the European Conference on Computer Vision ({ECCV})").
SOURCES: https://api.crossref.org/works/10.1007/978-3-030-58548-8_11 (ECCV 2020, LNCS, pp. 179-196);
         https://link.springer.com/chapter/10.1007/978-3-030-58548-8_11 (meta tags + abstract);
         DBLP conf/eccv/EisenbergerC20 (SPARQL, ECCV (4)).
CLAIMS: MISMATCH (minor, same sentence as 2004-ActualMorphing): Chapter2.tex:32 "emphasized physical realism ... to improve
        accuracy" -- paper says: physically plausible interpolation (volume preservation, momentum conservation) that is
        robust to imperfect correspondences and noise in real scans. -- proposed: see 2004-ActualMorphing.

KEY: 2011-ShapeCorrespondenceSurvey
STATUS: CORRECTED
CHANGES: author "Kaick, Oliver van" -> "van Kaick, Oliver" (prints "van Kaick, O."); added volume 30, number 6,
         pages 1681--1707 (missing in the input). Title/year/DOI unchanged.
SOURCES: https://api.crossref.org/works/10.1111/j.1467-8659.2011.01884.x (CGF 30(6):1681-1707, 2011, abstract);
         DBLP journals/cgf/KaickZHC11 (SPARQL).
CLAIMS: OK (Chapter3.tex:116, techniques for point-to-point correspondences: the survey reviews exactly these).

KEY: 10.1145/3130800.3130820
STATUS: VERIFIED-UNCHANGED
CHANGES: formatting only (articleno 213 / numpages 15 -> pages = {213:1--213:15}).
SOURCES: https://api.crossref.org/works/10.1145/3130800.3130820 (TOG 36(6), 15 pp., abstract); DBLP journals/tog/ZhuPBK17
         (SPARQL: 213:1-213:15).
CLAIMS: OK (Chapter2.tex:56 "intricate mesh setups": the method builds a consistent multimesh with Comesh Optimization over
        2D drawings).

KEY: bansal2018lie
STATUS: VERIFIED-UNCHANGED
CHANGES: formatting only (booktitle -> "Proceedings of the {ACM SIGGRAPH} European Conference on Visual Media Production
         ({CVMP})"; articleno 5 / numpages 10 -> pages = {5:1--5:10}).
SOURCES: https://api.crossref.org/works/10.1145/3278471.3278477 (Proceedings of the 15th ACM SIGGRAPH European Conference on
         Visual Media Production, 10 pp., 2018); DBLP conf/cvmp/BansalT18 (SPARQL: 5:1-5:10); abstract opening via OpenAlex.
CLAIMS: OK (Chapter2.tex:28; the title and abstract opening support "interpolation via Lie body representations of
        triangle meshes"; the full abstract was not retrievable, ACM DL blocked).

KEY: bao2005physically
STATUS: VERIFIED-UNCHANGED
CHANGES: formatting only.
SOURCES: https://api.crossref.org/works/10.1002/cav.100 (CAVW 16(3-4):509-518, 2005, abstract); DBLP journals/jvca/BaoGQ05 (SPARQL).
CLAIMS: MISMATCH: Chapter2.tex:30 "used point samples on surface meshes to obtain results with greater physical
        plausibility than linear blend methods" -- paper says: it morphs point-sampled (point-set) surfaces by dynamic
        meshless simulation, with no surface mesh; the comparison is to linear interpolation. -- proposed: "morphed
        point-sampled surfaces with a meshless physical simulation, obtaining more physically plausible results than
        linear interpolation".

KEY: breen2001level
STATUS: PUBLISHED-VERSION-SUBSTITUTED
CHANGES: the input cited the one-page SIGGRAPH 99 "Conference Abstracts and Applications" sketch (p. 228, 1999,
         doi 10.1145/311625.312113), although the key says 2001. Substituted the full peer-reviewed article of the same
         title: @article, IEEE Transactions on Visualization and Computer Graphics 7(2):173--192, 2001,
         doi 10.1109/2945.928169. Authors and title unchanged. (Hu et al. 2004 also cite this work as [BW01].)
         If the author really meant the 1999 sketch, revert to: booktitle "{ACM SIGGRAPH} 99 Conference Abstracts and
         Applications", pages 228, year 1999, doi 10.1145/311625.312113 (verified via Crossref and DBLP conf/siggraph/BreenW99).
SOURCES: https://api.crossref.org/works/10.1109/2945.928169 (TVCG 7(2):173-192, 2001); DBLP journals/tvcg/BreenW01 (SPARQL,
         full names David E. Breen, Ross T. Whitaker); https://api.crossref.org/works/10.1145/311625.312113; abstract via OpenAlex.
CLAIMS: OK (Chapter2.tex:28 implicit-surface approach: level-set (iso-surface) deformable model).

KEY: kilian2007geometric
STATUS: CORRECTED
CHANGES: pages "64--es" (ACM's article-64 "electronic supplement" notation, not a page range) -> pages = {64}
         (article number). The input's numpages 8 could not be verified (Crossref, DBLP and the ACM DL record give only
         "64" / "64 - es"; the author PDF is a 5-page preprint), so no 64:1--64:N range is given.
SOURCES: https://api.crossref.org/works/10.1145/1276377.1276457 (TOG 26(3), page 64, abstract); DBLP journals/tog/KilianMP07
         (SPARQL: 64); ACM DL page via http://web.archive.org/web/20241217051816/https://dl.acm.org/doi/10.1145/1276377.1276457
         ("Volume 26, Issue 3, Pages 64 - es").
CLAIMS: OK (Chapter2.tex:28, shapes as points in a Riemannian shape space, interpolation by geodesics).

KEY: liu2011rigid
STATUS: CORRECTED
CHANGES: author "Martin, Ralph R" -> "Martin, Ralph R."; added doi 10.1007/s11390-011-1154-3 (missing); dropped publisher.
         Title, volume 26, number 3, pages 548--557, year 2011 unchanged. "Ya-Shu Liu" confirmed by Springer and Crossref.
SOURCES: https://link.springer.com/article/10.1007/s11390-011-1154-3 (meta tags: Liu, Ya-Shu; Yan, Han-Bing; Martin, Ralph R.;
         JCST 26(3):548-557, 2011/05; abstract); https://api.crossref.org/works?query.bibliographic=... (DOI); DBLP
         journals/jcst/LiuYM11 (SPARQL).
CLAIMS: OK (Chapter2.tex:32: abstract -- avoids a consistent tetrahedral mesh, needs only a consistent triangle surface mesh).

KEY: turk2005shape
STATUS: VERIFIED-UNCHANGED
CHANGES: formatting only (booktitle -> SIGGRAPH policy form; year 1999 as in the input; the key's "2005" is only the key).
SOURCES: https://api.crossref.org/works/10.1145/311535.311580 (SIGGRAPH '99, pp. 335-342); DBLP conf/siggraph/TurkO99 (SPARQL);
         abstract via OpenAlex.
CLAIMS: OK (Chapter2.tex:28 implicit-surface approach: variational implicit functions).

KEY: sumner2004deftransfer
STATUS: VERIFIED-UNCHANGED
CHANGES: none.
SOURCES: https://api.crossref.org/works/10.1145/1015706.1015736 (TOG 23(3):399-405, 2004, abstract); DBLP journals/tog/SumnerP04.
CLAIMS: OK (Appendix.tex:616 gradient-domain form: per-triangle transformations solved in a linear least-squares system).

KEY: kazhdan2012cmcf
STATUS: VERIFIED-UNCHANGED
CHANGES: formatting only (title in Title Case: "... Flow Be Modified to Be Non-Singular?"; sentence-cased output identical).
SOURCES: https://api.crossref.org/works/10.1111/j.1467-8659.2012.03179.x (CGF 31(5):1745-1754, 2012, abstract);
         DBLP journals/cgf/KazhdanSB12 (SPARQL).
CLAIMS: OK (Appendix.tex:617: the modified flow converges to a conformal parameterization of genus-zero surfaces onto the sphere).

KEY: 10.1145/1015706.1015743
STATUS: VERIFIED-UNCHANGED
CHANGES: formatting only.
SOURCES: https://api.crossref.org/works/10.1145/1015706.1015743 (TOG 23(3):441-448, 2004, abstract); DBLP journals/tog/FattalL04.
CLAIMS: OK (Chapter4.tex:455: a driving-force term added to the flow equations).

KEY: treuille2003keyframe
STATUS: VERIFIED-UNCHANGED
CHANGES: formatting only.
SOURCES: https://api.crossref.org/works/10.1145/882262.882337 (TOG 22(3):716-723, 2003, abstract); DBLP journals/tog/TreuilleMPS03.
CLAIMS: OK (Chapter4.tex:455: optimises "wind" forces applied to the velocity field).

KEY: chen2024eigenfluids
STATUS: CORRECTED
CHANGES: authors "Levin, David" -> "Levin, David I. W." and "Langlois, Timothy" -> "Langlois, Timothy R." (paper byline and
         DBLP; ACM/Crossref metadata omit the initials); pages "1--11" -> "44:1--44:11" (ACM DL: Article No. 44, Pages 1-11);
         booktitle "ACM SIGGRAPH 2024 Conference Papers" -> "{ACM SIGGRAPH} Conference Papers" (policy form).
SOURCES: https://api.crossref.org/works/10.1145/3641519.3657468 (11 pp., 2024-07-13); paper PDF
         https://dgp.toronto.edu/projects/fluidcontrol/siggraph_24_74.pdf (byline, ACM Reference Format "11 pages", DOI);
         DBLP conf/siggraph/ChenLL24 (SPARQL, no pagination); ACM DL page via
         http://web.archive.org/web/20240924092503/https://dl.acm.org/doi/10.1145/3641519.3657468 ("Article No.: 44, Pages 1 - 11").
CLAIMS: OK (Chapter4.tex:455: "users can optimize for control forces to satisfy specified keyframes", Sec. 1). Wording nit:
        "Classical physics-based shape control" also covers this 2024 and the 2025 paper; consider "Physics-based shape
        control" (optional).

KEY: li2025flowmapadjoint
STATUS: CORRECTED
CHANGES: author "Lin, Ming" -> "Lin, Ming C." (paper byline; Crossref has "Ming Lin"); pages "1--12" -> "128:1--128:12"
         (DBLP); booktitle "SIGGRAPH Asia 2025 Conference Papers" -> "{SIGGRAPH Asia} Conference Papers" (policy form).
         Note: the arXiv copy's ACM Reference Format says "15 pages" (it includes the supplement); ACM metadata (Crossref
         1-12, DBLP 128:1-128:12) is used.
SOURCES: https://api.crossref.org/works/10.1145/3757377.3763903 (Proceedings of the SIGGRAPH Asia 2025 Conference Papers,
         pp. 1-12, 9 authors); DBLP conf/siggrapha/0004HBZ00LT025 (SPARQL: 128:1-128:12); arXiv PDF
         https://arxiv.org/pdf/2511.01259 (byline, abstract, Sec. 6 smoke control "optimizes a time-varying control force").
CLAIMS: OK (Chapter4.tex:455: shape-morphing / smoke-control tasks optimise control forces). See the "Classical" nit above.

KEY: mccann1997convexity
STATUS: VERIFIED-UNCHANGED
CHANGES: none.
SOURCES: https://api.crossref.org/works/10.1006/aima.1997.1634 (Adv. Math. 128(1):153-179, 1997-06, McCann, Robert J.);
         author copy https://www.math.utoronto.ca/mccann/papers/advances.pdf ("advances in mathematics 128, 153-179 (1997)",
         Definition 1.1 "Displacement interpolation").
CLAIMS: OK (Chapter4.tex:469 and Appendix.tex:375: the paper defines displacement interpolation, Def. 1.1).

KEY: bonneel2011displacement
STATUS: CORRECTED
CHANGES: pages "1--12" -> "158:1--158:12" (article 158 per DBLP; 12 pages per Crossref and the ACM DL). Rest unchanged.
SOURCES: https://api.crossref.org/works/10.1145/2070781.2024192 (TOG 30(6), pp. 1-12, 2011-12, abstract); DBLP
         journals/tog/BonneelPPH11 (SPARQL: 158); ACM DL page via
         http://web.archive.org/web/20250527104450/https://dl.acm.org/doi/10.1145/2070781.2024192 ("Volume 30, Issue 6, Pages 1 - 12").
CLAIMS: OK (Chapter4.tex:469: optimal-transport displacement interpolation via RBF mass transport).
