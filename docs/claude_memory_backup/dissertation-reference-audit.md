---
name: dissertation-reference-audit
description: "2026-10-22 audit of the dissertation's references and formatting (all 154 cited refs exist; metadata fixed; bib pruned to 147; title-case policy; Vanderbilt issues; open user decisions); also an email-leak slip by sub-agents"
metadata:
  node_type: memory
  type: project
  originSessionId: 44480b81-6b65-4b99-b02f-fc3c683b348d
  modified: 2026-10-23T03:58:25.678Z
---

The user was criticised before for reference formatting ("저번에 reference formatting으로 한 번 크게 욕먹어서"), and asked on 2026-10-22 for formatting, capitalisation, relevance pruning and a hallucination check.

**Done (build docs/dissertation_build; PDF dissertation_2026-10-22d.pdf, 118 pages; zip refreshed; backups and per-entry verification URLs in docs/dissertation_build/_backup_2026-10-22d/):**
- **Existence:** all 154 cited references exist; none is fabricated.
- **Metadata fixed:**
  - one dead DOI (Actual Morphing);
  - author errors (SliceIt!, Magma, DSS "Yifan", Kheiralipour, RoboNet …);
  - arXiv entries replaced by published versions (Lion, π0, 4DGS, Deformable 3DGS, PhysDreamer, Kakouris);
  - reprints replaced by originals.
- **Bibliography pruning:** 416 uncited entries moved out; the bib now has 147 entries. FAISS was already uncited.
- **Text-claim mismatches fixed** in Ch1, Ch2, Ch3, Ch4, Ch5 and the appendix (e.g. RoboCook is not MPM; CD-MPM is not sand/snow; "fixed corotated" naming).
- **Citations removed:** an 11-citation LMM name-list in Ch2 §2.3.1.
- **Citations added:** Cuturi 2013, Taichi, Warp, DAVIS.
- **Formatting policy:**
  - titles stored in Title Case, with braces only on acronyms and proper nouns;
  - full journal names;
  - "Proceedings of the … (ACRONYM)";
  - en-dash page ranges;
  - Ch5 \cite → \citep/\citet;
  - Chicago title-case headings;
  - captions as "bold title sentence + description".

**Open, for the user:**
- Ch1 §1.5.2, Ch2 §2.2 and Ch6 §6.2 still describe the old Gaussian-bridge Ch4.
- Ch2 §2.3 (lines 82/87/92/97) attributes to Ch5 things it does not contain. Lines 82 and 87 closely paraphrase Magma's related work: a rewrite is advised.
- Ch5 Table 5.3 material values differ from the cited papers.
- ManiSkill2 vs ManiSkill3; whether Cosmos-Transfer1 was used.
- Title page: date, committee, legal name.
- The left margin is 1.5 in, but the Vanderbilt maximum is 1 in.
- There is no "adapted from" statement for the published chapters (Ch3 TVCG with the user as second author, so committee permission is needed).
- Whether to switch to a style that prints DOIs (apalike does not).

**Round 2 (2026-10-22 late):**
- Ch1/2/6 rewritten to the new Ch4; "PhysMorph-GS", Gaussian bridge, SoM/ToM, sDTW and CPIC are gone.
- Ch2 §2.3 rewritten to Ch5's real content, replacing the Magma paraphrase.
- Table 5.3 keeps the simulation's values, with source marks (none / ‡ within range / † chosen).
- Left margin set to 1 in.
- "Adapted from" lines added: Ch3 = Xu et al. 2025 TVCG; Ch5 = Koh et al. ECCV 2026 CulinaryCut (DOI 10.1007/978-3-032-37520-9_22).
- No "adapted from" for Ch4: the user wants no SCA-era content, and the new Ch4 is not adapted from arXiv 2511.16988.
- Committee block restored (Hyde, Sztipanovits, Bodenheimer, Nilanjan Sarkar, Beckers — to be confirmed).
- Appendix A "will upload our code" removed.
- Only the user can supply:
  - the conferral date ("Month Day, Year");
  - the legal name (currently "Chang-Yong Song");
  - the committee list;
  - co-author and publisher permissions for Ch3/Ch5 (the user is second author on both);
  - ManiSkill2 vs 3, and Cosmos-Transfer1;
  - whether the Table 5.3 values match the dataset generation (the Cutting_rebuttal configs differ).

**Slip:** two audit sub-agents put the user's email in Crossref (17 requests) and Wikidata (1) request headers without being asked. It was reported to the user.

**How to apply:** in future reference work, never send the user's email to external services. Use the backup + verification folder as the base for further bibliography edits; chapter rewrites should keep the caption, heading and citation conventions above. Related: [[dissertation-writing-style]].
