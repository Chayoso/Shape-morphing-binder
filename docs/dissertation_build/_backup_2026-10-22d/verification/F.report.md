# Batch F report: robot learning (VLA models, robot datasets, embodied-AI simulators), 22 entries

Conventions used throughout this batch:
- PMLR (CoRL) entries: `year` is the PMLR record's year (v100 = 2020, v229 = 2023, v270 = 2025). The CoRL meeting years were 2019, 2023 and 2024. The PMLR proceedings carry no DOI.
- Author order follows the byline of the published PDF. Where the publisher's web metadata disagrees with the PDF, the PDF wins: PMLR reverses RT-2's author list, and PMLR's BridgeData V2 order differs from its PDF.
- More than 20 authors: the first 10 as published, then `and others`. A group author printed in the published byline (Open X-Embodiment Collaboration, Octo Model Team, NVIDIA) is kept first, in braces.
- DBLP was rate-limited (HTTP 429 / cookie wall) during this run, and the OpenReview v1 API asked for a challenge. Every field was therefore checked against the publisher pages (PMLR, roboticsproceedings.org, proceedings.neurips.cc, openaccess.thecvf.com, iclr.cc), Crossref, the OpenReview v2 search API, and the arXiv abs pages. The PDFs were read where a claim had to be checked.

---

KEY: openvla
STATUS: VERIFIED-UNCHANGED
CHANGES: none factual. Formatting only: booktitle "Proceedings of the 8th Conference on Robot Learning" -> policy form "Proceedings of the Conference on Robot Learning ({CoRL})"; dropped url and publisher. Authors, title, v270, pp. 2679--2713 and year 2025 agree with PMLR.
SOURCES: https://proceedings.mlr.press/v270/kim25c.html (BibTeX, abstract); PDF https://raw.githubusercontent.com/mlresearch/v270/main/assets/kim25c/kim25c.pdf (byline; Sec. 3 "we build on the Prismatic-7B VLM")
CLAIMS: OK. Ch2:87 says it builds on the Prismatic VLM backbone: the paper fine-tunes a pretrained Prismatic-7B VLM. Ch2:92 says it follows OpenVLA's action representation: the paper discretises actions into tokens. Ch5:33 and Ch5:68 are plain mentions. (For Ch2:92's comparative claim, see the cross-chapter item under rt2.)

KEY: rt2
STATUS: CORRECTED
CHANGES: added volume 229, pages 2165--2183 and series PMLR. Booktitle -> policy CoRL form. The author list was cut to the first 10 of 54, then `and others` (the input listed 11; Florence removed). Title brace {RT}-2 -> {RT-2}. Dropped the arXiv url.
  Note: PMLR's web/BibTeX metadata lists the 54 authors in REVERSE order (Zitkovich first; PMLR id zitkovich23a) and splits "Gonzalez Arenas" as "Arenas, Montserrat Gonzalez". The published PDF byline is alphabetical from Anthony Brohan ("Authors listed in alphabetical order"). The entry follows the PDF, so it is cited as Brohan et al. (2023).
SOURCES: https://proceedings.mlr.press/v229/zitkovich23a.html ; https://proceedings.mlr.press/v229/zitkovich23a/zitkovich23a.pdf (byline, abstract) ; https://proceedings.mlr.press/v229/ (index)
CLAIMS: OK for Ch2:87, which says co-fine-tuning on web VQA plus robot trajectories, with actions expressed as text tokens. Ch5:68 is a plain mention.
  MISMATCH (cross-chapter, concerns the citations rt2, openvla, llarva, tracevla): Ch2:92 "we supervise \emph{Set-of-Mark (SoM)} ... and \emph{Trace-of-Mark (ToM)} ... which strengthens spatio--temporal reasoning for cutting and improves initiation accuracy, contact stability across style transitions, and cross-object generalization relative to prior VLA formulations~\citep{rt2,openvla,llarva,tracevla}". Chapter 5 contains no SoM/ToM supervision (the terms appear only in Chapter 2). It evaluates RDT, Octo and OpenVLA "without changing their architectures", and it never compares against RT-2, LLARVA or TraceVLA. Ch5:33 also says "not a closed-loop force-feedback controller". The same holds for Ch2:87 ("SoM/ToM supervision and an MPM-based force feedback loop") and Ch2:92 ("force--time fidelity objective (e.g., sDTW/peak/impulse terms)"). None of these appear in Chapter 5. -- proposed: Ch2:87 "...that preserves OpenVLA-style action interfaces while training on demonstrations generated and selected under MLS-MPM material dynamics." In Ch2:92, delete the SoM/ToM sentence and the force--time sentence, or replace them with: "we keep the policies unchanged and instead generate and select their demonstrations under deformable-material dynamics."

KEY: o2024open
STATUS: CORRECTED
CHANGES: added the group author {Open X-Embodiment Collaboration} first: it opens the published byline, and Crossref's ICRA title even carries ": Open X-Embodiment Collaboration". The 10 named authors after it are unchanged, and they match the ICRA record. (The current arXiv version inserts Abhinav Gupta third; the ICRA version does not.) Added pages 6892--6903 and DOI 10.1109/ICRA57147.2024.10611477. Booktitle -> "Proceedings of the {IEEE} International Conference on Robotics and Automation ({ICRA})". Dropped the arXiv url. Title braces -> {Open X-Embodiment} ... {RT-X}.
  Note: apalike will cite it as "Open X-Embodiment Collaboration et al. (2024)", no longer "O'Neill et al.".
SOURCES: https://api.crossref.org/works/10.1109/ICRA57147.2024.10611477 (title, container, pages, 279 named authors) ; https://export.arxiv.org/api/query?id_list=2310.08864 ; https://arxiv.org/pdf/2310.08864 (Sec. III)
CLAIMS: OK. Ch2:97 says "over one million real-robot trajectories spanning 22 embodiments by pooling 60 prior datasets from 34 labs into a standardized RLDS/tfrecord format ... releases pre-trained RT-X checkpoints". Sec. III of the paper reads "1M+ real robot trajectories spanning 22 robot embodiments ... pooling 60 existing robot datasets from 34 robotic research labs ... RLDS data format, which saves data in serialized tfrecord files" and "Pre-Trained Checkpoints: a selection of RT-X model checkpoints". (The abstract gives 21 institutions; the 34-lab figure is from Sec. III.) Ch5:75: see the libero2023 block.

KEY: mees2024octo
STATUS: CORRECTED
CHANGES: added the group author {Octo Model Team} first. The RSS 2024 PDF byline reads "Octo Model Team Dibya Ghosh, Homer Walke, ...". The named authors are reordered to the PDF byline: "... Lawrence Yunliang Chen, Pannag Sanketi, Quan Vuong, Ted Xiao, Dorsa Sadigh ...". The input followed the RSS web page / Crossref order, which puts Vuong and Xiao before Sanketi. All 18 named authors are kept (18 is under 20). Booktitle -> "Proceedings of Robotics: Science and Systems ({RSS})". Title brace {Octo}. DOI kept. url dropped.
  Note: it will now be cited as "Octo Model Team et al. (2024)". If you prefer "Ghosh et al.", drop the group author: the RSS web metadata and Crossref list no group.
SOURCES: https://www.roboticsproceedings.org/rss20/p090.html (BibTeX, abstract) ; https://www.roboticsproceedings.org/rss20/p090.pdf (byline) ; https://api.crossref.org/works/10.15607/RSS.2024.XX.090
CLAIMS: OK (Ch5:33 and Ch5:68 are plain mentions).

KEY: liu2025rdt1bdiffusionfoundationmodel
STATUS: VERIFIED-UNCHANGED
CHANGES: none factual (ICLR 2025 poster; the 9 authors match). Formatting: authors written as "Last, First"; booktitle -> policy ICLR form; title -> "{RDT-1B}: A Diffusion Foundation Model for Bimanual Manipulation". OpenReview prints "a"; it is capitalised here for Title Case and is kept by apalike after the colon.
SOURCES: OpenReview v2 search API (forum yAzN4tz7oI, venue "ICLR 2025 Poster", authors, title) -> https://openreview.net/forum?id=yAzN4tz7oI
CLAIMS: OK (Ch5:33 and Ch5:68 are plain mentions).

KEY: black2024pi0visionlanguageactionflowmodel
STATUS: PUBLISHED-VERSION-SUBSTITUTED
CHANGES: @misc arXiv:2410.24164 (2024) -> @inproceedings, Proceedings of Robotics: Science and Systems ({RSS}), 2025, DOI 10.15607/RSS.2025.XXI.010. The RSS version has 25 authors: it adds Laura Smith and gives "Michael Robert Equi". The list is therefore cut to the first 10, then `and others`. Title -> "{$\pi_0$}: ...". The note field was dropped.
SOURCES: https://www.roboticsproceedings.org/rss21/p010.html (authors, BibTeX, DOI) ; https://www.roboticsproceedings.org/rss21/index.html ; https://api.crossref.org/works/10.15607/RSS.2025.XXI.010 (the PDF has no extractable text, so the byline was checked through the RSS page and Crossref)
CLAIMS: OK (Ch5:68 is a plain mention).

KEY: llarva
STATUS: CORRECTED
CHANGES: year 2024 -> 2025 (PMLR v270 record; the meeting was CoRL 2024). Added volume 270, pages 3333--3355 and series PMLR. Booktitle -> policy CoRL form. Dropped the arXiv url. The authors match the PDF byline.
SOURCES: https://proceedings.mlr.press/v270/niu25a.html ; https://raw.githubusercontent.com/mlresearch/v270/main/assets/niu25a/niu25a.pdf
CLAIMS: OK. Ch2:87 says it "predicts 2D visual traces as an auxiliary target to align vision and action spaces". The abstract says "predicting intermediate 2-D representations, which we refer to as visual traces, can help further align vision and action spaces".

KEY: lapa
STATUS: CORRECTED
CHANGES: author "Joo, Sejune" -> "Joo, Se June" (as on OpenReview). Booktitle -> policy ICLR form. Dropped the arXiv url. Venue (ICLR 2025 poster), title and the other 15 authors are verified.
SOURCES: OpenReview v2 search API (forum VYOe2eBQeh, venue "ICLR 2025 Poster", _bibtex, abstract) -> https://openreview.net/forum?id=VYOe2eBQeh
CLAIMS: OK. Ch2:87 says "predicting VQ-VAE--based latent actions ... before mapping to robot actuation". The abstract says "VQ-VAE-based objective to learn discrete latent actions ... finetune the VLA ... to map from latent to robot actions". Ch2:92 is a plain mention.

KEY: moto
STATUS: VERIFIED-UNCHANGED
CHANGES: none factual (ICCV 2025, pp. 19752--19763, DOI 10.1109/ICCV51701.2025.01837; CVF and IEEE pages agree). Formatting: booktitle -> "Proceedings of the {IEEE/CVF} International Conference on Computer Vision ({ICCV})"; title brace {Moto}; url dropped.
SOURCES: https://openaccess.thecvf.com/content/ICCV2025/html/Chen_Moto_Latent_Motion_Token_as_the_Bridging_Language_for_Learning_ICCV_2025_paper.html (BibTeX, abstract) ; https://api.crossref.org/works/10.1109/ICCV51701.2025.01837
CLAIMS: OK. Ch2:87 says "motion tokens before mapping to robot actuation": the paper uses a Latent Motion Tokenizer, motion-token autoregression, and co-fine-tuning to robot control.

KEY: tracevla
STATUS: VERIFIED-UNCHANGED
CHANGES: none factual (ICLR 2025 poster, the 8 authors including Daum{\'e} III). Formatting: booktitle -> policy ICLR form; url dropped.
SOURCES: OpenReview v2 search API (forum b1CVu9l5GO, venue "ICLR 2025 Poster", _bibtex, abstract) -> https://openreview.net/forum?id=b1CVu9l5GO
CLAIMS: OK. Ch2:87 says "visual trace prompting ... improves generalist policies across embodiments". The abstract says it outperforms OpenVLA, with "robust generalization across diverse embodiments and scenarios". (For Ch2:92, see the cross-chapter item under rt2.)

KEY: libero2023
STATUS: VERIFIED-UNCHANGED
CHANGES: none factual (NeurIPS 36, pp. 44776--44791, DOI 10.52202/075280-1939; the NeurIPS site and Crossref agree). Formatting: booktitle "Advances in Neural Information Processing Systems (Datasets and Benchmarks Track)" -> the policy form "Advances in Neural Information Processing Systems ({NeurIPS})". Restore the track wording if you want it shown. url dropped.
SOURCES: https://proceedings.neurips.cc/paper_files/paper/2023/hash/8c3c666820ea055a77726d66fc7d447f-Abstract-Datasets_and_Benchmarks.html ; PDF (same path, -Paper-Datasets_and_Benchmarks.pdf) ; https://api.crossref.org/works/10.52202/075280-1939
CLAIMS: Ch2:97 is OK. It says "provides language-conditioned manipulation suites, but does not include deformable cutting". The paper has 130 language-conditioned tasks in four suites, built on robosuite, with no deformable or cutting task.
  MISMATCH (minor): Ch5:75 "General robot datasets such as DROID, Open X-Embodiment, BridgeData V2, and LIBERO provide broad task and embodiment coverage". Only Open X-Embodiment spans many embodiments (22). DROID was collected entirely on one Franka Panda platform ("all data is collected on the same robot hardware stack"). BridgeData V2 uses one low-cost WidowX platform. LIBERO is a single-arm simulated benchmark. -- proposed: "...provide broad task and scene coverage (Open X-Embodiment also spans 22 embodiments)."

KEY: bridgedata2023
STATUS: CORRECTED
CHANGES: added volume 229, pages 1723--1736 and series PMLR. Booktitle -> policy CoRL form. Dropped the arXiv url. The authors are unchanged: they match the PDF byline order. PMLR's web metadata gives a different order (Walke, Black, Zhao, Vuong, Zheng, ...) and fuller names ("Homer Rich", "Tony Z.", "Andre Wang"); the PDF byline was followed. Title brace -> {BridgeData V2}.
SOURCES: https://proceedings.mlr.press/v229/walke23a.html ; https://proceedings.mlr.press/v229/walke23a/walke23a.pdf ; https://export.arxiv.org/api/query?id_list=2308.12952
CLAIMS: OK. Ch2:97 says "60,096 trajectories across 24 environments on a low-cost platform, enabling goal/language conditioning and cross-institution generalization", and the published PDF says exactly that. (Caution: the abstract on the PMLR web page says "53,896 trajectories". That is a stale metadata abstract; the PMLR PDF, the arXiv abstract and the conclusion all say 60,096.) For Ch5:75, see libero2023.

KEY: robonet2019
STATUS: CORRECTED
CHANGES: author "Bucher, Ben" -> "Bucher, Bernadette" (PMLR record and PDF byline). Year 2019 -> 2020 (PMLR v100 record; the meeting was CoRL 2019, Osaka). Added volume 100, pages 885--897 and series PMLR. Booktitle -> policy CoRL form. Dropped the arXiv url.
SOURCES: https://proceedings.mlr.press/v100/dasari20a.html ; http://proceedings.mlr.press/v100/dasari20a/dasari20a.pdf ; https://proceedings.mlr.press/v100/
CLAIMS: OK. Ch2:97 says "about 15M video frames across seven platforms". The abstract says "15 million video frames, from 7 different robot platforms".

KEY: khazatsky2025droidlargescaleinthewildrobot
STATUS: CORRECTED
CHANGES: authors: the input carried the full arXiv list of about 96 names. Some of them (Vitor Guizilini, Muhammad Zubair Irshad, Kevin Lin) are not in the RSS list. The list is cut to the first 10 of the RSS version (they are the same 10), then `and others`. Added DOI 10.15607/RSS.2024.XX.120. Booktitle -> policy RSS form. Dropped the arXiv url.
SOURCES: https://www.roboticsproceedings.org/rss20/p120.html (BibTeX, DOI) ; https://www.roboticsproceedings.org/rss20/p120.pdf (byline, text) ; https://api.crossref.org/works/10.15607/RSS.2024.XX.120
CLAIMS: MISMATCH (minor), Ch2:97 "76k demonstrations ($\sim$350 hours) over 564 scenes and 80+ tasks" -- paper says: "76k demonstration trajectories or 350 hours of interaction data, collected across 564 scenes and 86 tasks". (The RSS web-page abstract says "65k". That is stale; the RSS PDF says 76k.) -- proposed: "...over 564 scenes and 86 tasks".
  Ch5:90 table row OK: Real; multi-camera yes (two external Zed 2 and one wrist Zed Mini); language yes (crowd-sourced instructions); continuous actions yes. For Ch5:75, see libero2023.

KEY: gu2023ManiSkill2
STATUS: VERIFIED-UNCHANGED
CHANGES: none factual (ICLR 2023; all 15 authors match). Key written exactly as `gu2023ManiSkill2`: the input bib had `gu2023maniskill2`, while the dissertation cites gu2023ManiSkill2. Booktitle -> policy ICLR form.
SOURCES: https://iclr.cc/virtual/2023/poster/12152 (title, author order, OpenReview link https://openreview.net/forum?id=b_CQDy9vrD1) ; WebSearch confirming ICLR 2023
CLAIMS: OK as a citation for "ManiSkill" (Ch5:24). Advisory: the text says "ManiSkill" without a version. If IntSim actually runs on ManiSkill3 (the current mani_skill package), cite ManiSkill3 instead. I could not tell which version IntSim uses from the sources at hand.

KEY: deitke2022
STATUS: CORRECTED
CHANGES: added DOI 10.52202/068431-0433 (on the NeurIPS page; confirmed by Crossref). Booktitle -> policy NeurIPS form. The NeurIPS page title has a leading house emoji, which is omitted. Authors, volume 35 and pp. 5982--5994 are verified. Crossref spells "Vanderbilt"; the NeurIPS page has "VanderBilt", which is kept.
SOURCES: https://proceedings.neurips.cc/paper_files/paper/2022/hash/27c546ab1e4f1d7d638e6a8dfbad9a07-Abstract-Conference.html ; https://api.crossref.org/works/10.52202/068431-0433
CLAIMS: OK. Ch5:378 uses ProcTHOR as a source of indoor backgrounds.

KEY: kolve2017ai2
STATUS: CORRECTED
CHANGES: the input author list was truncated with "and others" after 10 names, but the work has only 13 authors. All 13 are now listed, adding Kembhavi, Aniruddha; Gupta, Abhinav; Farhadi, Ali. Title -> "{AI2-THOR}: An Interactive {3D} Environment for Visual {AI}". note -> howpublished. Added DOI 10.48550/arXiv.1712.05474. No peer-reviewed version exists: arXiv only, v1 2017-12-14, latest v4 2022.
SOURCES: https://arxiv.org/abs/1712.05474 (citation meta, versions, DOI)
CLAIMS: OK (Ch5:378).

KEY: szot2021habitat
STATUS: CORRECTED
CHANGES: added volume 34 and pages 251--266. Booktitle -> policy NeurIPS form. Title -> "{Habitat 2.0}: Training Home Assistants to Rearrange Their Habitat" (Title Case). The authors are correct: first 10 of 21, then others. No DOI is shown on the NeurIPS 2021 page, and none was found in Crossref.
SOURCES: https://proceedings.neurips.cc/paper_files/paper/2021/hash/021bbc7ee20b71134d53e20206bd6feb-Abstract.html
CLAIMS: OK. Ch5:378 cites it for ReplicaCAD, and the abstract introduces "ReplicaCAD: an artist-authored, annotated, reconfigurable 3D dataset of apartments".

KEY: gong2023arnold
STATUS: CORRECTED
CHANGES: the input was truncated with "and others" after 10 names, but the work has 12 authors. Added Jia, Baoxiong and Huang, Siyuan. Added pages 20426--20438 and DOI 10.1109/ICCV51070.2023.01873, both from IEEE Xplore via Crossref. Note: the CVF open-access BibTeX gives pp. 20483--20495; the IEEE pages are used because they go with the DOI, and the CVF PDF's watermark says the final version is on IEEE Xplore. Booktitle -> policy ICCV form.
SOURCES: https://openaccess.thecvf.com/content/ICCV2023/html/Gong_ARNOLD_A_Benchmark_for_Language-Grounded_Task_Learning_with_Continuous_States_ICCV_2023_paper.html (BibTeX) ; the CVF PDF (text) ; https://api.crossref.org/works/10.1109/iccv51070.2023.01873
CLAIMS: MISMATCH, Ch5:91 table row "Arnold~\cite{gong2023arnold} & -- & Isaac Sim & Synthetic & Yes & -- & Yes & --". The Language column is "--", but the paper says "ARNOLD is comprised of 8 language-conditioned tasks", "Each task is associated with ... detailed task descriptions in human language form", and its Table 1 has a "Language" column. Secondary point on "Physics sim. --": ARNOLD runs PhysX 5.0 and "simulates fluids at the particle level" with GPU position-based dynamics (Sec. 3). It has no deformable-solid or MPM solver, but it is not free of physics simulation. -- proposed: Language "Yes". Physics sim. "PBD (fluids)", or keep "--" and define the column as "deformable-solid solver". The multi-camera (five cameras), continuous action (7-DoF) and Isaac Sim entries are OK.

KEY: tobin2017domainrandomizationtransferringdeep
STATUS: CORRECTED
CHANGES: added pages 23--30 and DOI 10.1109/IROS.2017.8202133. Booktitle -> "Proceedings of the {IEEE/RSJ} International Conference on Intelligent Robots and Systems ({IROS})". Authors written as "Last, First". The title and the 6 authors are verified.
SOURCES: https://api.crossref.org/works/10.1109/IROS.2017.8202133
CLAIMS: OK. Ch5:111 says domain randomization mitigates the visual sim-to-real gap.

KEY: nvidia2025cosmosworldfoundationmodel
STATUS: CORRECTED
CHANGES: no peer-reviewed version was found: the work is an arXiv technical report (v1 2025-01-07, now v3), so it stays @misc. Authors: arXiv lists the group author "NVIDIA" first, then 77 named authors. The entry now reads {NVIDIA} plus the first 10 named authors, then others (the input listed 77 names with no group). Title brace {Cosmos}. note -> howpublished. Added DOI 10.48550/arXiv.2501.03575. It will be cited as "NVIDIA et al. (2025)".
SOURCES: https://arxiv.org/abs/2501.03575 (citation meta: 79 author entries = "NVIDIA", ":", 77 names; abstract; DOI) ; WebSearch (no conference version found)
CLAIMS: Ch5:111 ("world models can mitigate these differences") is acceptable.
  POSSIBLE MISMATCH: Ch5:728 "We also examine Cosmos-based appearance enhancement~\citep{nvidia2025cosmosworldfoundationmodel} while preserving the underlying cutting sequence". The cited platform paper covers video curation, pre-trained world foundation models, post-training examples and tokenizers. It does not cover appearance transfer of simulator renderings. Sim-to-real appearance transfer that keeps the scene structure (conditioned on segmentation, depth and edge) is the subject of Cosmos-Transfer1: NVIDIA et al., "Cosmos-Transfer1: Conditional World Generation with Adaptive Multimodal Control", arXiv:2503.14492, 2025 (checked at https://arxiv.org/abs/2503.14492). -- proposed: if Cosmos-Transfer was the model used, add that citation at Ch5:728, for example "Cosmos-based appearance enhancement~\citep{nvidia2025cosmosworldfoundationmodel,<cosmos-transfer1 key>}". Otherwise name the Cosmos model that was used.

KEY: yu2025forcevla
STATUS: VERIFIED-UNCHANGED (pages need a decision)
CHANGES: added DOI 10.52202/085713-3124. Booktitle -> policy NeurIPS form. Title in Title Case: "Force-Aware", "Contact-Rich"; the published form is "Force-aware ... Contact-rich" and prints the same. The authors (12), volume 38 and year 2025 are verified.
  PAGE DISCREPANCY: Crossref, for this DOI, gives pp. 103609--103639, which is the input's value, kept here. proceedings.neurips.cc gives citation_firstpage/lastpage 93409--93439. The offset is systematic for NeurIPS 2025: I checked 3 other papers and the site and Crossref pages differ for each. Pick one convention for every NeurIPS 2025 entry in the bibliography: the DOI/Crossref pages (used here) or the NeurIPS-site pages.
SOURCES: https://proceedings.neurips.cc/paper_files/paper/2025/hash/8633b46e12cc5f2ee1f05a6ca2c65b38-Abstract-Conference.html (meta, abstract) ; https://proceedings.neurips.cc/paper_files/paper/2025/vol38-main-conference (index) ; https://api.crossref.org/works/10.52202/085713-3124
CLAIMS: OK. Ch5:70 says it "explores force-conditioned policy inputs". The abstract says it "treats external force sensing as a first-class modality ... integrates ... real-time 6-axis force feedback during action decoding".

---

## Other cross-chapter claim issues found while checking these contexts

- Ch2:97 "\emph{These trends highlight a gap for large, multi-embodiment \textbf{real-robot} cutting datasets ...; Chapter \ref{CH5_CUTTINGSIM} addresses this by proposing a collection/evaluation protocol that includes ratio/direction conditioning, SoM/ToM annotations, F/T and pose labels, and a force-aware sim$\leftrightarrow$real loop built on MLS-MPM/CPIC.}" -- Chapter 5 describes CulinaryCut as synthetic (its Table: "Synthetic"). It is generated on one Franka Panda in ManiSkill. SoM/ToM and CPIC appear nowhere in Chapter 5, and force is not a policy input. -- proposed: "...a gap for large, physics-grounded cutting datasets; Chapter \ref{CH5_CUTTINGSIM} addresses this with simulated demonstrations generated and selected under MLS-MPM material dynamics, with ratio/direction conditioning and force records."
