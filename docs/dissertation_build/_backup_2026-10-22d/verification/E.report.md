# Batch E report: multimodal LLMs, VLMs, benchmarks (17 keys)

Access notes: the arXiv export API and DBLP refused requests (rate limit and a bot challenge), so arXiv
metadata came from the arXiv abs pages (citation_* meta tags, comments, DOI line), and DBLP records were
read through OpenReview's DBLP mirror (api2.openreview.net/notes/search). Every field below comes from a
page that was opened: CVF open access BibTeX, Crossref, proceedings.neurips.cc, proceedings.iclr.cc,
proceedings.mlr.press, ACL Anthology, Springer and ECVA.

---

KEY: achiam2023gpt
STATUS: CORRECTED
CHANGES: author: the arXiv record names the group "OpenAI" first (281 authors) -> `{OpenAI} and Achiam, Josh and ... Anadkat, Shyamal and others` (group plus the first 10 named authors, per POLICY). title "{GPT}-4 technical report" -> "{GPT-4} Technical Report". note -> howpublished; doi 10.48550/arXiv.2303.08774 added. arXiv only (v1 2023-03-15, v6 latest), no peer-reviewed version.
  Effect: the in-text citation now prints "OpenAI et al. (2023)", not "Achiam et al. (2023)".
SOURCES: https://arxiv.org/abs/2303.08774
CLAIMS: MISMATCH: Chapter2.tex:82 "motivating multimodal extensions that accept images and video~\citep{achiam2023gpt}" -- paper says: GPT-4 is "a large-scale, multimodal model which can accept image and text inputs and produce text outputs"; it says nothing about video. -- proposed: "motivating multimodal extensions that accept images~\citep{achiam2023gpt}"

KEY: ataallah2024minigpt4video
STATUS: CORRECTED
CHANGES: venue "Proceedings of the IEEE/CVF CVPR Workshops" (2024) -> @misc, howpublished = {arXiv preprint arXiv:2404.03413}, doi 10.48550/arXiv.2404.03413. I found no archival version. The paper is not among the papers of any of the 78 CVPR 2024 workshops on CVF open access (all 78 workshop pages scanned). The official GitHub README cites it as arXiv:2404.03413. The DBLP record (OpenReview mirror) is "CoRR 2024" only. Its companion paper Goldfish is the one in ECCV 2024. Some papers that cite it say "CVPRW 2024", but there is no proceedings entry, so I could not verify that. Authors and title match arXiv.
SOURCES: https://arxiv.org/abs/2404.03413 ; https://openaccess.thecvf.com/CVPR2024_workshops/menu (and the 78 workshop pages) ; https://raw.githubusercontent.com/Vision-CAIR/MiniGPT4-video/main/README.md ; https://api2.openreview.net/notes/search?term=MiniGPT4-Video ; Crossref title search (no hit)
CLAIMS: MISMATCH (weak): Chapter2.tex:82 "video LMMs extend temporal reasoning with interleaved visual--text tokens and long-context handling~\citep{ataallah2024minigpt4video,li2024llavainterleave}" -- paper says: interleaved visual-textual tokens over a sequence of frames, benchmarked on short-video QA (MSVD, MSRVTT, TGIF, TVQA). Long videos are the subject of the companion Goldfish paper, and LLaVA-Interleave does not claim long-context handling either. -- proposed: "...with interleaved visual--text tokens over multiple frames~\citep{...}"

KEY: cai2023vipllava
STATUS: CORRECTED
CHANGES: author order (arXiv order) "Cai, Liu, Park, Mustikovela, Meyer, Chai, Lee" -> published CVPR order "Cai, Mu and Liu, Haotian and Mustikovela, Siva Karthik and Meyer, Gregory P. and Chai, Yuning and Park, Dennis and Lee, Yong Jae". pages 12914--12923 added; doi 10.1109/CVPR52733.2024.01227 added; booktitle in the POLICY form.
SOURCES: https://openaccess.thecvf.com/content/CVPR2024/html/Cai_ViP-LLaVA_Making_Large_Multimodal_Models_Understand_Arbitrary_Visual_Prompts_CVPR_2024_paper.html ; Crossref 10.1109/CVPR52733.2024.01227 ; https://arxiv.org/abs/2312.00784
CLAIMS: OK (arbitrary visual prompts such as boxes, arrows and scribbles support "spatial prompts ... pointing, boxes")

KEY: chen2023shikra
STATUS: VERIFIED-UNCHANGED
CHANGES: none factual. The arXiv-only status is confirmed: the official README cites arXiv:2306.15195, and neither OpenReview nor Crossref has a published version. Formatting only: note -> howpublished, name braced as {Shikra}, doi 10.48550/arXiv.2306.15195 added.
SOURCES: https://arxiv.org/abs/2306.15195 ; https://raw.githubusercontent.com/shikras/shikra/main/README.md ; api2.openreview.net search "Shikra" ; Crossref bibliographic search
CLAIMS: OK (referential dialogue with coordinate inputs and outputs)

KEY: eichenberg2022magma
STATUS: CORRECTED
CHANGES: booktitle "Findings of the Association for Computational Linguistics: Empirical Methods in Natural Language Processing" -> "Findings of the Association for Computational Linguistics: {EMNLP}" (the ACL Anthology name is "Findings of the Association for Computational Linguistics: EMNLP 2022"; the year is dropped per POLICY). pages 2416--2428 added; doi 10.18653/v1/2022.findings-emnlp.179 added. Authors and title match.
SOURCES: https://aclanthology.org/2022.findings-emnlp.179.bib ; Crossref 10.18653/v1/2022.findings-emnlp.179
CLAIMS: OK ("adapter methods ... lightweight multimodal augmentation" matches adapter-based finetuning of a generative LM). See the name-dropping note.

KEY: fang2024mmbenchvideo
STATUS: CORRECTED
CHANGES: volume 37, pages 89098--89124 and doi 10.52202/079017-2827 added (NeurIPS 2024 Datasets and Benchmarks Track); booktitle gets "({NeurIPS})". Authors and title match.
SOURCES: https://proceedings.neurips.cc/paper_files/paper/2024/hash/a2326c9715a516c91174132e0170073a-Abstract-Datasets_and_Benchmarks_Track.html (and its .bib) ; Crossref 10.52202/079017-2827
CLAIMS: OK ("long-form temporal understanding" matches lengthy YouTube videos and temporal reasoning)

KEY: li2023seedbench
STATUS: VERIFIED-UNCHANGED
CHANGES: none factual (CVPR 2024, 13299--13308, doi 10.1109/CVPR52733.2024.01263 all confirmed). Formatting only: url dropped, booktitle in the POLICY form.
SOURCES: https://openaccess.thecvf.com/content/CVPR2024/html/Li_SEED-Bench_Benchmarking_Multimodal_Large_Language_Models_CVPR_2024_paper.html ; Crossref 10.1109/CVPR52733.2024.01263 ; https://arxiv.org/abs/2307.16125
CLAIMS: MISMATCH (shared with li2023seedbench2; see below). SEED-Bench is 19K multiple-choice questions in 12 dimensions covering image and video (spatial and temporal) comprehension. It is not "hierarchical" and not "text-rich".

KEY: li2023seedbench2
STATUS: VERIFIED-UNCHANGED
CHANGES: none factual. arXiv only: the DBLP mirror shows "CoRR 2023" only, and no venue version was found. Formatting only: note -> howpublished, title brace {SEED-Bench-2} (was {SEED}-Bench-2), doi 10.48550/arXiv.2311.17092 added.
SOURCES: https://arxiv.org/abs/2311.17092 ; api2.openreview.net search "SEED-Bench-2 Benchmarking"
CLAIMS: MISMATCH: Chapter2.tex:82 "hierarchical and text-rich settings (SEED-Bench / SEED-Bench-2)" -- paper says: SEED-Bench-2 sorts MLLM capabilities into hierarchical levels L0 to L4 (24K questions, 27 dimensions, including text and image generation). "Text-rich" is the subject of a different paper, SEED-Bench-2-Plus ("... with Text-Rich Visual Comprehension"), which is not cited. -- proposed: "spatio--temporal and hierarchical settings (SEED-Bench / SEED-Bench-2)"

KEY: li2024llavainterleave
STATUS: PUBLISHED-VERSION-SUBSTITUTED
CHANGES: @misc arXiv:2407.07895 (2024) -> @inproceedings ICLR 2025 (Spotlight). title "LLaVA-NeXT-Interleave: ..." -> published title "{LLaVA-Interleave}: Tackling Multi-image, Video, and {3D} in Large Multimodal Models". booktitle "Proceedings of the International Conference on Learning Representations ({ICLR})"; pages 81182--81199 (ICLR proceedings); year 2024 -> 2025. Authors unchanged (the proceedings print "Zejun MA"; kept as "Ma, Zejun"). No DOI. The in-text year changes to 2025; the key is kept.
SOURCES: https://proceedings.iclr.cc/paper_files/paper/2025/hash/c9f95e9ec39fa5ad3d0a562b993b92aa-Abstract-Conference.html ; https://proceedings.iclr.cc/paper_files/paper/2025/file/c9f95e9ec39fa5ad3d0a562b993b92aa-Bibtex-Conference.bib ; OpenReview forum oSQiao9GqB (via api2 search) ; https://arxiv.org/abs/2407.07895
CLAIMS: MISMATCH (weak): the same sentence as under ataallah2024minigpt4video. LLaVA-Interleave handles multi-image, multi-frame (video), multi-view (3D) and multi-patch inputs in one interleaved format; it does not claim "long-context handling".

KEY: liu2023llava
STATUS: CORRECTED
CHANGES: pages 34892--34916 added (NeurIPS 36). volume 36 and doi 10.52202/075280-1516 confirmed; url dropped; booktitle gets "({NeurIPS})".
SOURCES: https://proceedings.neurips.cc/paper_files/paper/2023/hash/6dcf277ea32ce3288914faf369fe6de0-Abstract-Conference.html (and its .bib) ; Crossref 10.52202/075280-1516
CLAIMS: OK for LLaVA (a CLIP encoder joined to Vicuna through a projection matrix; instruction-following visual chat). See radford2021clip for the sentence's other citation.

KEY: liu2024mmbench
STATUS: VERIFIED-UNCHANGED
CHANGES: none factual (ECCV 2024, LNCS, pp. 216--233, doi 10.1007/978-3-031-72658-3_13 confirmed; all 12 authors in order). Formatting only: the title follows Springer's capitalisation "All-Around" (prints the same in sentence case), url dropped, booktitle in the POLICY form.
SOURCES: Crossref 10.1007/978-3-031-72658-3_13 ; https://www.ecva.net/papers/eccv_2024/papers_ECCV/html/959_ECCV_2024_paper.php
CLAIMS: OK ("general multimodal comprehension")

KEY: radford2021clip
STATUS: VERIFIED-UNCHANGED
CHANGES: none factual (ICML 2021, PMLR 139:8748--8763, all 12 authors in order). Formatting only: series "PMLR" -> "Proceedings of Machine Learning Research"; booktitle gets "({ICML})".
SOURCES: https://proceedings.mlr.press/v139/radford21a.html
CLAIMS: MISMATCH (attribution): Chapter2.tex:82 "A common recipe aligns a frozen visual encoder to an open LLM via a lightweight projector or adapter, enabling instruction-following in vision--language dialogue~\citep{radford2021clip,liu2023llava,zhu2023minigpt4}" -- paper says: CLIP is contrastive image-text pretraining of the visual encoder. It does not align an encoder to an LLM and has no dialogue. -- proposed: "aligns a frozen visual encoder (e.g., CLIP~\citep{radford2021clip}) to an open LLM via a lightweight projector, enabling instruction-following in vision--language dialogue~\citep{liu2023llava,zhu2023minigpt4}"

KEY: rasheed2024glamm
STATUS: CORRECTED
CHANGES: author "Mullappilly, Sahal Shaji" (the arXiv form) -> "Shaji, Sahal" (the name as published in CVPR, on both CVF and IEEE/Crossref). "Khan, Fahad Shahbaz" -> "Khan, Fahad S." (as published). pages 13009--13018 and doi 10.1109/CVPR52733.2024.01236 added.
SOURCES: https://openaccess.thecvf.com/content/CVPR2024/html/Rasheed_GLaMM_Pixel_Grounding_Large_Multimodal_Model_CVPR_2024_paper.html ; Crossref 10.1109/CVPR52733.2024.01236 ; https://arxiv.org/abs/2311.03356
CLAIMS: OK (pixel-level grounding and segmentation masks in conversation)

KEY: yang2025magma
STATUS: CORRECTED
CHANGES: authors: the input list was a corrupted mix of the arXiv list (it kept Lars Liden and dropped Seonghyeon Ye). The published CVPR list has 12 authors: "... Gu, Yu and Cai, Mu and Ye, Seonghyeon and Jang, Joel and Deng, Yuquan and Gao, Jianfeng" (no Liden). pages 14203--14214 and doi 10.1109/CVPR52734.2025.01325 added.
SOURCES: https://openaccess.thecvf.com/content/CVPR2025/html/Yang_Magma_A_Foundation_Model_for_Multimodal_AI_Agents_CVPR_2025_paper.html ; Crossref 10.1109/CVPR52734.2025.01325 ; https://arxiv.org/abs/2502.13130 and https://arxiv.org/html/2502.13130
CLAIMS: MISMATCH: Chapter2.tex:82 "the more recent Magma agent model introduces \emph{Set-of-Mark} and \emph{Trace-of-Mark} as surrogate tasks" -- paper says: Set-of-Mark is earlier work that Magma cites as its ref. [125] (Yang et al. 2023, "Set-of-Mark Prompting Unleashes Extraordinary Visual Grounding in GPT-4V"). Magma "propose[s] the use of" SoM and ToM labels as "surrogate action supervisions" for its two surrogate tasks, action grounding and action planning. -- proposed: "...the more recent Magma agent model uses Set-of-Mark and Trace-of-Mark labels as surrogate action supervision to bridge ..."
  Also: Chapter2.tex:92 adopts "Set-of-Mark (SoM)" and "Trace-of-Mark (ToM)" supervision with no citation at all; cite yang2025magma there. See the cross-chapter note below: Chapter 5 does not use SoM/ToM.

KEY: zhang2024gpt4roi
STATUS: CORRECTED
CHANGES: venue "Proceedings of the European Conference on Computer Vision" -> "Proceedings of the European Conference on Computer Vision Workshops ({ECCVW})". It appeared in Springer LNCS "Computer Vision -- ECCV 2024 Workshops" (DBLP: "ECCV Workshops (8) 2024"), not the main conference. year 2024 -> 2025 (the volume was published in 2025; online 2025-05-12). pages 52--70 and doi 10.1007/978-3-031-91813-1_4 added. Authors (9, in order) and title match. The in-text year becomes 2025; the key is kept. The earlier ICLR 2024 submission was not accepted.
SOURCES: Crossref 10.1007/978-3-031-91813-1_4 ; https://link.springer.com/chapter/10.1007/978-3-031-91813-1_4 ; api2.openreview.net search "GPT4RoI"
CLAIMS: OK (region-of-interest features in instruction tuning)

KEY: zhu2023minigpt4
STATUS: CORRECTED
CHANGES: booktitle "International Conference on Learning Representations" -> "Proceedings of the International Conference on Learning Representations ({ICLR})"; pages 18378--18394 added (ICLR 2024 proceedings); title brace {MiniGPT-4} (was {MiniGPT}-4). Authors match (the proceedings print "chen, jun", kept as "Chen, Jun").
SOURCES: https://proceedings.iclr.cc/paper_files/paper/2024/file/50623630a2372839c078474efa6c0cb8-Bibtex-Conference.bib ; https://proceedings.iclr.cc/paper_files/paper/2024/hash/50623630a2372839c078474efa6c0cb8-Abstract-Conference.html ; https://arxiv.org/abs/2304.10592
CLAIMS: OK (a frozen visual encoder and frozen Vicuna joined by one projection layer)

KEY: prismaticvlm
STATUS: CORRECTED
CHANGES: volume 235, pages 23123--23144 and series "Proceedings of Machine Learning Research" added; url (arXiv) dropped. Authors and title match PMLR.
SOURCES: https://proceedings.mlr.press/v235/karamcheti24a.html
CLAIMS: MISMATCH: Chapter2.tex:87 "Several studies adapt large multimodal models to robot control~\citep{rt2,openvla,prismaticvlm,llarva,tracevla,lapa,moto}" -- paper says: Prismatic is a study of the design space of VLMs (image preprocessing, visual backbones, LLMs, training recipe), with no robot control. Magma's own version of this sentence ("Several studies have investigated the application of LMMs in robotics [6, 95, 146, 66, 54, 142, 130]") does not include Prismatic either. -- proposed: remove prismaticvlm from this list and keep it only in the OpenVLA sentence ("building on an open-source Prismatic VLM backbone~\citep{openvla,prismaticvlm}"), which is correct.

---

## Is each citation needed? (reported, nothing removed)

All 17 keys are cited only in Chapter2.tex:82 and :87. Chapter 5 trains or evaluates no LMM. It runs RDT, Octo and
OpenVLA without changing them, and it uses no region prompts, video LMMs, LMM benchmarks or SoM/ToM.
- Justified: prismaticvlm (the backbone of OpenVLA, which Chapter 5 evaluates); liu2023llava and radford2021clip (minimal LMM background).
- Justified only if the SoM/ToM framing stays: yang2025magma.
- Name-dropping (one mention in a list, no role anywhere else in the dissertation): cai2023vipllava, rasheed2024glamm, chen2023shikra,
  zhang2024gpt4roi, ataallah2024minigpt4video, li2024llavainterleave, liu2024mmbench, li2023seedbench,
  li2023seedbench2, fang2024mmbenchvideo, eichenberg2022magma. achiam2023gpt and zhu2023minigpt4 are marginal.

## Close paraphrase of Magma's related work (high risk)

Chapter2.tex:82 ("Large Multimodal Models") and :87 ("Vision--Language--Action for Robotics") follow the Related Work section of
Magma (arXiv 2502.13130, Sec. 2) paragraph by paragraph, including its citation clusters:
- Magma: "LLMs like ChatGPT, GPT-4, and Llama have demonstrated impressive reasoning and generalization capabilities for text ... LLaVA and MiniGPT-4, which combine LLMs' capabilities with a CLIP based image encoder ... region-level LMMs [7, 137, 12, 100, 134], and video LMMs [69, 135, 139, 112]. In parallel, more sophisticated benchmarks are proposed ..."
- Magma: "Several studies have investigated the application of LMMs in robotics [...]. Among these, RT-2 ... discretized robot action tokens. OpenVLA ... Prismatic VLM backbone. LLARVA generated 2D visual traces ... auxiliary task, while TraceVLA used visual trace prompting ... learning from videos by predicting the latent VQVAE tokens [11, 129] ... instead of asking model to predict latent tokens, we propose SoM and ToM ..."
- Chapter2.tex:92 ("Instead of only predicting latent tokens from video~\citep{lapa,moto}, we supervise SoM ... ToM ...") repeats Magma's own contribution statement.
Rewrite these two paragraphs in your own structure and cite only the works the dissertation uses.

## Cross-chapter claim not supported by Chapter 5 (outside the bibliography, but tied to yang2025magma)

Chapter2.tex:82 says Chapter 5 incorporates "force-aware signals (impulse/force traces) into the learning loop". :87 says Chapter 5
couples OpenVLA-style actions "with SoM/ToM supervision and an MPM-based force feedback loop". :92 says "we supervise SoM ... ToM
..., which ... improves initiation accuracy ..." and mentions a "force--time fidelity objective (e.g., sDTW/peak/impulse terms)".
Chapter5.tex and Appendix.tex contain no SoM, Set-of-Mark, Trace-of-Mark or sDTW. Chapter5.tex:180 says the impulse measurement
"is recorded alongside generated trajectories but is not a VLA observation at inference time". Chapter5.tex:281 separates the
safety filter "from any learned force-aware behavior". These Chapter 2 sentences describe a design that Chapter 5 does not contain.
