# Bibliography policy for the dissertation (BibTeX style: apalike + natbib authoryear)

Facts about apalike that drive the policy:
- It prints: author, year, title, journal/booktitle, volume, number, pages, editor, series, publisher/address
  (books only here), howpublished, note. It IGNORES doi, url, month, eprint, issn, isbn, articleno, numpages.
- It converts the titles of @article, @inproceedings, @incollection and @misc to sentence case
  ("t" change.case$): everything is lowercased except the first character, the first character after
  ": ", and anything inside braces. Book titles (@book/@inbook) are printed as stored (italic).
- Reference-list author names become "Last, F. M."; "and others" prints as "et al.".

## Required output format (one entry per cited key, KEEP THE CITATION KEY EXACTLY AS GIVEN)

Field order: author, title, journal|booktitle, volume, number, pages, year, (series), (publisher, address for books),
(howpublished for preprints/software), doi. Two-space indent, `field = {value},`. No other fields
(drop url, month, issn, isbn, publisher/address/organization for papers, numpages, articleno, keywords,
abstract, location, eprint, archiveprefix, primaryclass, note) unless listed below.

Authors
- Every author, in published order, `Last, First Middle` joined by ` and `. Full given names as published.
- Diacritics as LaTeX in braces: `Popovi{\'c}`, `M{\"u}ller`, `Ka{\v{c}}i{\'c}-Alesi{\'c}`, `J{\'e}gou`.
- More than 20 authors: list the first 10 authors as published, then `and others`. If the publication
  itself names a group author first (e.g. "Open X-Embodiment Collaboration", "OpenAI", "NVIDIA", "Octo Model
  Team"), keep the group as the first author in braces, e.g. `{Octo Model Team}`, then the named authors
  under the same >20 rule. Otherwise never use `and others`.

Titles
- Exact published title (published version, not the arXiv title if they differ), in Title Case.
- Brace-protect ONLY words that must keep capitals in sentence case: acronyms ({MPM}, {MLS-MPM}, {APIC},
  {3D}, {4D}, {RGB-D}, {SLAM}, {GPU}, {CFL}, {IoU}, {MMD}, {NeRF}, {VLA}, {LLM}, {RL}),
  proper nouns and words derived from names ({Gaussian}, {Gaussians}, {Lagrangian}, {Eulerian},
  {Sinkhorn}, {Wasserstein}, {Laplacian}, {Hamiltonian}, {Lie}, {Poisson}, {Newton}, {Monte Carlo},
  {Markov}), and system/dataset/method names ({ChainQueen}, {DiffTaichi}, {ManiSkill2}, {OpenVLA},
  {LLaVA}, {GPT-4}, {RT-2}, ...). Protect the whole word: `{ChainQueen}`, not `{C}hain{Q}ueen`.
  Never brace a whole title or the initial capital of an ordinary word.
- Math stays in math mode and is braced: `{$\pi_0$}`, `3 $\times$ 3`.

Venues
- @article: `journal` = full official journal name in Title Case, never abbreviated
  (ACM Transactions on Graphics; IEEE Transactions on Visualization and Computer Graphics; Journal of
  Computational Physics; Computer Methods in Applied Mechanics and Engineering; Computer Graphics Forum;
  International Journal for Numerical Methods in Engineering; ...). Give volume, number, pages.
  Article-numbered journals (ACM TOG and other ACM journals with article numbers):
  `pages = {<articleno>:1--<articleno>:<numpages>}` (e.g. 69:1--69:9).
  A paper presented at SIGGRAPH / SIGGRAPH Asia and published in ACM TOG is an @article in ACM TOG.
- @inproceedings: `booktitle` = "Proceedings of the <full conference name> ({ACRONYM})", no year, no
  ordinal. Use exactly these forms when they apply:
  - Proceedings of the {IEEE/CVF} Conference on Computer Vision and Pattern Recognition ({CVPR})
    (for CVPR before 2018 the organiser was IEEE: Proceedings of the {IEEE} Conference on Computer Vision and Pattern Recognition ({CVPR}))
  - Proceedings of the {IEEE/CVF} International Conference on Computer Vision ({ICCV})
  - Proceedings of the European Conference on Computer Vision ({ECCV})  (Springer LNCS; give pages if listed)
  - Proceedings of the International Conference on Learning Representations ({ICLR})
  - Proceedings of the International Conference on Machine Learning ({ICML})  + volume + `series = {Proceedings of Machine Learning Research}`
  - Proceedings of the International Conference on Artificial Intelligence and Statistics ({AISTATS}) + volume + series PMLR
  - Advances in Neural Information Processing Systems ({NeurIPS})  + volume (proceedings.neurips.cc) + pages when given
  - Proceedings of the Conference on Robot Learning ({CoRL})  + volume + series PMLR + pages
  - Proceedings of Robotics: Science and Systems ({RSS})
  - Proceedings of the {IEEE} International Conference on Robotics and Automation ({ICRA})
  - Proceedings of the {IEEE/RSJ} International Conference on Intelligent Robots and Systems ({IROS})
  - Proceedings of the Annual Conference on Computer Graphics and Interactive Techniques ({SIGGRAPH})
    (SIGGRAPH papers before 2002, which appeared in the conference proceedings; use the proceedings DOI
    10.1145/<proc>.<paper>, not the "ACM SIGGRAPH Computer Graphics" reprint DOI)
  - {ACM SIGGRAPH} Conference Papers  /  {SIGGRAPH Asia} Conference Papers   (conference-track papers from 2022 on)
  - When a paper was first published in a venue and later reprinted (SIGGRAPH Courses, "Seminal Graphics"
    collections), cite the ORIGINAL publication unless the text clearly means the reprint.
  - Proceedings of the {ACM SIGGRAPH/Eurographics} Symposium on Computer Animation ({SCA})
  - {ACM SIGGRAPH} Courses   /   {ACM SIGGRAPH} Posters   /   {ACM SIGGRAPH} Talks
  - Proceedings of the Conference on Empirical Methods in Natural Language Processing ({EMNLP}),
    Proceedings of the Annual Meeting of the Association for Computational Linguistics ({ACL}), etc.
  For any other venue follow the same pattern with the publisher's own proceedings name.
  Give pages when the published proceedings have them (`--`). Volume for NeurIPS/PMLR only.
- @incollection (chapter in an edited book): booktitle = book title in Title Case, editor, pages,
  publisher, address, year.
- @book: title in Title Case (it prints as stored), publisher, address, edition if any.
- @misc for an arXiv-only preprint (no peer-reviewed version exists):
  `howpublished = {arXiv preprint arXiv:NNNN.NNNNN}`, year = year of the first arXiv version.
  If a peer-reviewed version EXISTS, cite that instead (change the entry type and venue).
- @misc for software: author(s), title, `howpublished = {\url{https://...}}`, year (first release or the
  version used).
- Pages always with `--`. Always keep a verified `doi` field when the work has a DOI (it does not print,
  but it documents the verification). No `url` field (it does not print) except inside howpublished.

## Verification standard
Check every field against an authoritative source that you actually opened: the DOI resolver /
publisher page (ACM DL, IEEE Xplore, Springer, Elsevier/ScienceDirect, Wiley, Taylor & Francis),
DBLP (dblp.org, which gives authors/venue/pages), Crossref API (api.crossref.org/works/<doi>),
arXiv abs pages, proceedings.mlr.press, proceedings.neurips.cc, openreview.net, openaccess.thecvf.com,
roboticsproceedings.org, ecva.net. Never fill a field from memory. If a field cannot be verified, leave
it out and say so. If the work cannot be found at all, say NOT FOUND with what you searched.
