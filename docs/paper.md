% ---------------------------------------------------------------------
% PhysMorph-GS — SCA 2026 Submission
% EG author guidelines format (v2.04)
% ---------------------------------------------------------------------

\usepackage{amsmath,amssymb}
\title{PhysMorph-GS: Surface-Aware Render-Guided Volumetric Morphing with Differentiable Physics}

\author[1105]
{\parbox{\textwidth}{\centering Paper 1105
        }
        \\
{\parbox{\textwidth}{\centering }
}
}

%-------------------------------------------------------------------------
\begin{document}

\teaser{
 \includegraphics[width=0.95\linewidth]{figures/teaser}
 \centering
  \caption{Physically plausible volumetric morphing from an isosphere to four target shapes (Spot, Bunny, Bob, Teapot). Each MPM particle simultaneously serves as a 3D Gaussian primitive. Rendering supervision acts exclusively on the deformation gradient $\mathbf{F}$ (shaping Gaussian covariance) while positions are governed by physics and Chamfer-guided plasticity.}
 \label{fig:teaser}
}

\maketitle
%-------------------------------------------------------------------------
\begin{abstract}
Differentiable particle-based simulation produces physically plausible motion, but shape morphing from sparse visual observations remains severely underconstrained: physics-only optimization over the full volume may recover coarse global structure, yet often fails to reconstruct fine geometric details such as thin protrusions, while naive image-space coupling can interfere with early physical exploration and drive the system toward incorrect basins before the desired structural bifurcation emerges. We present a surface-aware render-guided volumetric morphing framework that preserves full volumetric physics while restricting observation to a 3D Gaussian Splatting-based surface shell, and converts multi-view alpha/depth supervision into smoothed control-space guidance within the physics loop. We further introduce a coarse-to-fine curriculum that weakens rendering supervision in early stages and gradually strengthens the coupling once physically meaningful shape seeds have formed. Experiments on challenging morphing cases show improved target-shape alignment over a physics-only baseline and better preservation of thin structures such as bunny ears, suggesting that the key challenge lies not in simply adding stronger image losses, but in how and when visual supervision is injected into differentiable physics.

\begin{CCSXML}
<ccs2012>
<concept>
<concept_id>10010147.10010371.10010382</concept_id>
<concept_desc>Computing methodologies~Physical simulation</concept_desc>
<concept_significance>500</concept_significance>
</concept>
<concept>
<concept_id>10010147.10010371.10010352</concept_id>
<concept_desc>Computing methodologies~Rendering</concept_desc>
<concept_significance>300</concept_significance>
</concept>
</ccs2012>
\end{CCSXML}

\ccsdesc[500]{Computing methodologies~Physical simulation}
\ccsdesc[300]{Computing methodologies~Rendering}

\printccsdesc
\end{abstract}

%=========================================================================
\section{Introduction}
\label{sec:intro}
%=========================================================================

Shape morphing, the process of smoothly deforming one 3D shape into another, is a fundamental operation in computer graphics with applications in visual effects, product design exploration through shape variations, and simulation-based content generation. The ideal morphing system would produce transitions that are both \emph{physically plausible} (respecting material properties and conservation laws) and \emph{visually faithful} to a desired target shape.

Prior work on this problem has evolved along several directions. Geometric approaches have laid the foundation for shape morphing by interpolating mesh vertex positions~\cite{alexa2023rigid} or blending implicit representations~\cite{turk2005shape}, achieving smooth visual transitions. However, since these methods do not model material mechanics, intermediate shapes may self-intersect, violate volume conservation, or exhibit physically implausible folding. On the other hand, physics-based deformable models~\cite{terzopoulos1988deformable,bouaziz2023projective,muller2004point} have made significant contributions by simulating material response to restore physical plausibility, but they provide no mechanism to actively guide the deformation toward a specific target shape, as the final configuration is determined by the material model and boundary conditions alone.

Recent advances in differentiable physics bridge this gap by enabling gradient-based optimization of simulation parameters to match desired outcomes~\cite{hu2019difftaichi,8794333,du2021diffpd}. In the context of shape morphing, Xu et al.~\cite{11088224} demonstrated that a differentiable Material Point Method (MPM) can morph one volumetric shape into another by optimizing a control deformation field that steers the simulation toward a target mass distribution. The result is a physically plausible animation that approximately matches the target shape.

However, the physics-only formulation has two fundamental limitations. First, the volumetric mass-matching loss on the Eulerian grid provides only coarse shape guidance, struggling with fine geometric details. Second, since the simulation output is a point cloud, visual assessment requires surface reconstruction (e.g., marching cubes~\cite{lorensen1998marching}), which introduces severe artifacts under large deformation due to non-uniform particle density. 3D Gaussian Splatting (3DGS)~\cite{kerbl20233d} addresses both limitations simultaneously: each particle maps directly to a Gaussian primitive, enabling differentiable rendering without explicit surface reconstruction, while multi-view silhouette and depth comparisons provide fine-grained shape feedback beyond what grid-based losses can capture. PhysGaussian~\cite{xie2024physgaussian} has demonstrated this coupling in practice, using the deformation gradient to warp Gaussian covariance.

While combining rendering supervision with physics simulation is straightforward in principle, the results depend critically on \emph{how and when} the rendering signal is injected into the physics loop. Our investigation reveals two failure modes of naive coupling. First, injecting rendering gradients into particle \emph{positions} creates an adversarial dynamic: the elastic restoring forces counteract the visual objective at every timestep, producing persistent oscillation (Section~\ref{sec:ablation}). Second, strong rendering supervision in \emph{early} episodes, before the physics has established meaningful structural seeds, can drive the system toward incorrect basins, preventing the desired topological bifurcation from emerging (e.g., legs separating from a body).

We address both challenges through a \emph{surface-aware render-guided} framework with three key design decisions:

\begin{enumerate}
\item \textbf{Surface-shell observation.} Only surface particles participate in rendering, focusing gradients on visually relevant geometry.

\item \textbf{Control-space guidance via $\mathbf{F}$.} Render gradients are routed through a differentiable $\mathbf{F} \to \boldsymbol{\Sigma}$ mapping~\cite{irving2004invertible}, modifying the control deformation field rather than particle positions. This indirectly influences trajectories through the physics solver while respecting material properties.

\item \textbf{Coarse-to-fine curriculum.} Early episodes run physics-only to establish structural seeds; rendering supervision and Chamfer-guided plasticity~\cite{fan2017point,lee1969elastic} are introduced gradually once meaningful shape features have formed.
\end{enumerate}

We highlight the Chamfer-guided plasticity mechanism. The multiplicative decomposition $\mathbf{F} = \mathbf{F}_e \mathbf{F}_p$~\cite{lee1969elastic} separates elastic from plastic deformation. By updating $\mathbf{F}_p$ based on 3D Chamfer displacement to the target surface, we migrate the elastic rest configuration toward the target. This converts the elastic restoring force from an adversary (pulling particles back to the source shape) into an ally that actively drives particles toward the target.

Our contributions are:
\begin{itemize}
\item \textbf{Surface-aware MPM--Gaussian duality.} We establish a one-to-one correspondence between MPM particles and 3D Gaussian primitives, with a differentiable $\mathbf{F} \to \boldsymbol{\Sigma}$ mapping that enables render gradients to flow to $\mathbf{F}$ (covariance) without directly affecting positions. Surface-aware masking restricts rendering to a shell of visually relevant particles.

\item \textbf{Control-space render injection.} Multi-view rendering gradients are injected exclusively into the deformation gradient ($\partial L / \partial \mathbf{F}$) while zeroing the position component ($\partial L / \partial \mathbf{x} = \mathbf{0}$). The render signal indirectly influences particle trajectories through the physics solver's control field, eliminating the adversarial dynamic of position-coupled approaches and achieving 27\% lower silhouette error with $3\times$ less oscillation.

\item \textbf{Coarse-to-fine Chamfer-guided plasticity.} A phased curriculum weakens rendering supervision in early episodes, then gradually introduces Chamfer-driven plastic deformation that migrates the elastic rest configuration. Physics elastic forces become allies that drive particles toward the target, reducing convergence rebound from 24\% to 7\%.
\end{itemize}

%=========================================================================
\section{Related Work}
\label{sec:related}
%=========================================================================

\subsection{Shape Morphing}
Shape morphing research has evolved along three main directions: geometric, metric-based, and physics-based approaches.

\noindent\textbf{Geometric approaches.}
Early methods established the foundations of shape morphing through mesh vertex interpolation~\cite{alexa2023rigid} or implicit function blending~\cite{turk2005shape}, producing visually smooth transitions. As-rigid-as-possible interpolation~\cite{alexa2023rigid} preserves local rigidity while allowing global deformation. Learning-based methods~\cite{groueix20183d,li2024deformnet} further advanced the field by encoding deformations in latent spaces for data-driven interpolation, though without physical constraints.

\noindent\textbf{Metric-based approaches.}
An alternative direction directly optimizes shape similarity metrics between source and target. Chamfer distance~\cite{fan2017point} measures point-set similarity via nearest-neighbor correspondences, though it can exhibit structural biases~\cite{song2026structuralfailurechamferdistance}. Optimal transport~\cite{peyre2019computational} provides a geometrically richer alternative through the Wasserstein distance, which captures global correspondences between mass distributions. However, metric-based optimization alone does not guarantee physical plausibility of intermediate shapes.

\noindent\textbf{Physics-based approaches.}
Physics-based deformable models~\cite{terzopoulos1988deformable,bouaziz2023projective,muller2004point} have contributed significantly by grounding morphing in material mechanics. Target-driven animation has been explored for fluids~\cite{manteaux2016space,10.1145/882262.882337} using keyframe control, but volumetric solid morphing with visual supervision remains underexplored. Xu et al.~\cite{11088224} recently demonstrated differentiable MPM-based morphing by optimizing a deformation control field, establishing the physics-only baseline we build upon.

\subsection{Material Point Method}
MPM~\cite{sulsky1994particle} discretizes continuous materials as Lagrangian particles that transfer information through an Eulerian grid, and has become one of the most versatile simulation tools in computer graphics. It has been successfully applied to snow~\cite{stomakhin2013material}, sand~\cite{klar2016drucker}, cloth~\cite{jiang2017anisotropic}, fracture~\cite{wolper2019cd}, and surface-tension-driven phenomena~\cite{conservative_surface_tension}. The APIC~\cite{10.1145/2766996} and MLS-MPM~\cite{hu2018moving} transfer schemes further improved accuracy and angular momentum conservation. A comprehensive tutorial is provided by Jiang et al.~\cite{jiang2016material}. The multiplicative decomposition $\mathbf{F} = \mathbf{F}_e \mathbf{F}_p$~\cite{lee1969elastic} separates elastic and plastic deformation, with stress computed from $\mathbf{F}_e$ alone, enabling independent control of plastic deformation; invertible finite element formulations~\cite{2012-FixedCoratedElasticty} ensure numerical robustness under large deformation.

Of particular relevance is the development of \emph{differentiable MPM}. ChainQueen~\cite{8794333} introduced a real-time differentiable physical simulator for soft robotics control, and DiffTaichi~\cite{hu2019difftaichi} provided a differentiable programming framework that facilitates differentiable implementations of diverse physical simulations. These frameworks enable backpropagation through simulation parameters, opening the door to gradient-based optimization for inverse design, control, and shape morphing as addressed in this work. Differentiable cloth simulation~\cite{10.1145/3527660} and differentiable projective dynamics~\cite{du2021diffpd} further extend this paradigm.

\subsection{3D Gaussian Splatting and Differentiable Rendering}
3DGS~\cite{kerbl20233d} has introduced an efficient scene representation using collections of anisotropic Gaussians with learnable position, covariance, opacity, and color, enabling real-time high-quality rendering. The rasterization-based pipeline efficiently provides gradients with respect to all Gaussian parameters, enabling optimization through backpropagation.

Following the success of 3DGS, numerous extensions have been actively developed. For dynamic scenes, deformation field-based approaches~\cite{wu20244d,yang2023deformable3dgaussianshighfidelity,liang2025gaufre} and explicit motion models~\cite{lee2024fullyexplicitdynamic,gao2024superpointgaussiansplatting} represent temporal deformation. Earlier work on differentiable point-based splatting~\cite{yifan2019differentiable} and modular rendering primitives~\cite{laine2020modular} laid the groundwork for geometry optimization through rendering. In parallel, neural radiance fields~\cite{mildenhall2020nerf,mueller2022instant} and neural implicit surfaces~\cite{wang2021neus} provide alternative differentiable representations, with dynamic variants~\cite{pumarola2020dnerf,park2021hypernerf} handling temporally varying scenes.

\subsection{Physics-Integrated Rendering}
Differentiable rendering has been extensively surveyed~\cite{kato2020differentiable}. Pioneering work on approximate differentiable renderers~\cite{loper2014opendr,liu2019softras} and subsequent high-performance modular primitives~\cite{laine2020modular} have established the foundations for rendering-based optimization. Building on these advances, PhysGaussian~\cite{xie2024physgaussian} demonstrated an elegant integration of 3DGS into MPM for generative dynamics, using the deformation gradient to warp Gaussians. OmniPhysGS~\cite{lin2025omniphysgs} extended this to general constitutive models. PAC-NeRF~\cite{li2023pacnerf} coupled NeRF with MPM for system identification, and PhysDreamer~\cite{zhang2024physdreamerphysicsbasedinteraction3d} showed that physics can be learned from video for interactive 3D objects. Recent work also explores Gaussian splatting for soft-body simulation in robotics~\cite{zhang2025realtosimrobotpolicyevaluation}. These works have focused primarily on \emph{forward} simulation with learned material properties; our work addresses the complementary \emph{inverse} problem of steering physics toward a visual target. Target-driven control of fluid simulations~\cite{mcnamara2004fluid,10.1145/1015706.1015743,10.1145/3016963} shares our goal of matching visual targets, but operates in Eulerian settings without the Lagrangian particle--Gaussian duality. Differentiable physics engines for robotics~\cite{degrave2019differentiable} address related inverse problems but without rendering-based objectives.

%=========================================================================
\section{Method}
\label{sec:method}
%=========================================================================

\subsection{Overview}

Given a source shape (isosphere) and a target shape (e.g., the Spot cow model), our goal is to produce a physically plausible morphing animation where the source continuously deforms into the target over a sequence of simulation episodes. The source volume is discretized as a set of $N$ MPM particles, each carrying position $\mathbf{x}_i$, velocity $\mathbf{v}_i$, and deformation gradient $\mathbf{F}_i$. Each particle simultaneously serves as a 3D Gaussian primitive for differentiable rendering.

The optimization proceeds in \emph{episodes}, each consisting of a short MPM simulation ($T$ timesteps) followed by multi-view rendering and gradient extraction. The particle state is promoted across episodes: positions and velocities at the end of episode $k$ become the initial conditions for episode $k\!+\!1$, creating a progressive morphing trajectory (Figure~\ref{fig:pipeline}).

Each episode performs three stages:
\begin{enumerate}
\item \textbf{Physics rollout.} An MPM forward simulation runs $T$ timesteps. A learnable control deformation field $\delta\mathbf{F}_c$ is optimized via the Adam optimizer~\cite{kingma2017adammethodstochasticoptimization} to minimize an end-layer mass loss that measures deviation from the target mass distribution on the Eulerian grid. If available, the render gradient $\partial L_\text{render} / \partial \mathbf{F}$ from the previous episode is injected as a penalty on $\mathbf{F}$ (with the position component $\partial L / \partial \mathbf{x}$ set to zero).

\item \textbf{Multi-view rendering and gradient extraction.} The final particle state $(\mathbf{x}_T, \mathbf{F}_T)$ is rendered from $V\!=\!8$ ring-placed cameras using the differentiable $\mathbf{F} \to \boldsymbol{\Sigma}$ mapping (Section~\ref{sec:duality}). Rendering losses (distance transform, soft IoU, and depth) are computed against target silhouettes obtained by rasterizing the target mesh. Backpropagation through the differentiable rendering chain yields $\partial L / \partial \mathbf{F}$, which is stored for injection in the next episode.

\item \textbf{Chamfer-guided plasticity} (activated after episode $k_0$). The Chamfer distance from each particle to the target surface defines a displacement field. After spatial diffusion and Jacobian estimation, this field updates the plastic deformation gradient $\mathbf{F}_p$, migrating the elastic rest configuration toward the target (Section~\ref{sec:chamfer}).
\end{enumerate}

The three stages work together: physics determines particle positions; rendering shapes Gaussian covariance via $\mathbf{F}$; and plasticity ensures that accumulated deformation is retained across episodes rather than being undone by elastic restoring forces.

% FIGURE: Pipeline
\begin{figure*}[t]
  \centering
  \includegraphics[width=0.95\linewidth]{figures/pipeline}
  \caption{Pipeline overview. Each episode: (1)~physics rollout with F-only render penalty from the previous episode, (2)~multi-view rendering through the differentiable $\mathbf{F}\!\to\!\boldsymbol{\Sigma}$ chain, extracting $\partial L/\partial\mathbf{F}$ for the next episode, and (3)~Chamfer-guided plasticity updating $\mathbf{F}_p$ to migrate the rest configuration. Positions are governed solely by physics; rendering acts only on $\mathbf{F}$ (covariance).}
  \label{fig:pipeline}
\end{figure*}

\subsection{MPM--Gaussian Duality}
\label{sec:duality}

The central observation enabling our approach is a natural correspondence between the state variables of MPM simulation and the parameters of 3D Gaussian Splatting. Each MPM particle $i$ carries position $\mathbf{x}_i \in \mathbb{R}^3$ and a deformation gradient $\mathbf{F}_i \in \mathbb{R}^{3 \times 3}$ that tracks how the local material neighborhood has been deformed from the rest configuration. In 3DGS, each Gaussian primitive is parameterized by a mean (position) $\boldsymbol{\mu}_i$ and a covariance matrix $\boldsymbol{\Sigma}_i \in \mathbb{R}^{3 \times 3}$ that determines its spatial extent and orientation.

We establish the duality $\boldsymbol{\mu}_i = \mathbf{x}_i$ (shared position) and derive $\boldsymbol{\Sigma}_i$ from $\mathbf{F}_i$ via polar decomposition~\cite{irving2004invertible}. The deformation gradient can be uniquely decomposed as:
\begin{equation}
\mathbf{F}_i = \mathbf{R}_i \mathbf{S}_i
\label{eq:polar}
\end{equation}
where $\mathbf{R}_i \in \text{SO}(3)$ is a proper rotation and $\mathbf{S}_i \in \mathbb{R}^{3 \times 3}$ is symmetric positive-definite (the stretch tensor). The Gaussian covariance is then:
\begin{equation}
\boldsymbol{\Sigma}_i = \mathbf{S}_i \, \boldsymbol{\Sigma}_0 \, \mathbf{S}_i^\top
\label{eq:cov}
\end{equation}
where $\boldsymbol{\Sigma}_0 = \sigma_0^2 \mathbf{I}$ is the isotropic rest covariance. The rest size $\sigma_0$ is determined adaptively from local particle spacing via KNN queries: $\sigma_0 = c \cdot \bar{d}_\text{KNN}$, where $\bar{d}_\text{KNN}$ is the mean distance to the nearest neighbor and $c$ is a scaling factor (we use $c = 0.7$).

Physically, this mapping has a natural interpretation: an initially spherical Gaussian ($\boldsymbol{\Sigma}_0 = \sigma_0^2 \mathbf{I}$) is stretched by $\mathbf{S}_i$ according to the local material deformation. A particle undergoing uniaxial stretching ($\mathbf{S} = \text{diag}(2, 1, 1)$) produces an elongated Gaussian; isotropic compression ($\mathbf{S} = 0.5 \, \mathbf{I}$) produces a smaller Gaussian. This mirrors how actual material deformation changes local geometry.

Crucially, the mapping $\mathbf{F} \to \boldsymbol{\Sigma}$ is fully differentiable through both the polar decomposition and the quadratic form in Eq.~\ref{eq:cov}. Given a rendering loss $L_\text{render}$ computed from rasterized images, the chain rule yields:
\begin{equation}
\frac{\partial L}{\partial \mathbf{F}_i} = \frac{\partial L}{\partial \boldsymbol{\Sigma}_i} \frac{\partial \boldsymbol{\Sigma}_i}{\partial \mathbf{S}_i} \frac{\partial \mathbf{S}_i}{\partial \mathbf{F}_i}
\label{eq:chain}
\end{equation}
This gradient tells us how to modify the \emph{deformation} of particle $i$ to improve rendering quality without producing any direct gradient on position $\mathbf{x}_i$.

\subsection{Control-Space Render Injection}
\label{sec:f_inject}

The differentiable MPM framework supports injecting external gradients into the physics optimization loop as an additive penalty consisting of $(\mathbf{g}^\mathbf{F},\, \mathbf{g}^\mathbf{x})$, where $\mathbf{g}^\mathbf{F}_i \in \mathbb{R}^{3 \times 3}$ acts on the deformation gradient and $\mathbf{g}^\mathbf{x}_i \in \mathbb{R}^3$ acts on particle positions. We set:
\begin{equation}
(\mathbf{g}^\mathbf{F}, \; \mathbf{g}^\mathbf{x}) = \left( \gamma \, \frac{\partial L_\text{render}}{\partial \mathbf{F}}, \; \mathbf{0} \right)
\label{eq:penalty}
\end{equation}
where $\gamma$ is the render-F gain (we use $\gamma\!=\!0.1$).

\subsubsection{Why position injection fails.} The most direct alternative, setting $\mathbf{g}^\mathbf{x} = \partial L_\text{render} / \partial \mathbf{x}$, creates a fundamental conflict. In hyperelasticity, the elastic restoring force $-\partial \Psi / \partial \mathbf{x}$ opposes any displacement from equilibrium, and its magnitude \emph{grows with displacement}. The render penalty $\lambda \nabla_\mathbf{x} L_\text{render}$ pushes particles toward visual targets. These two forces operate on the same variable ($\mathbf{x}$) with generally opposing directions, producing persistent oscillation that worsens with coupling strength (Figure~\ref{fig:oscillation}). Moreover, the position gradient is noisy in practice due to depth ambiguity (multiple particles per pixel), tangential ambiguity (underdetermined perpendicular component), and interior blindness (64\% of particles receive zero gradient).

\subsubsection{Control-space injection mechanism.} The injected $\partial L / \partial \mathbf{F}$ is added to the physics gradient during the backward pass, influencing the Adam update of the control deformation field $\delta\mathbf{F}_c$. In the subsequent forward pass, the modified $\delta\mathbf{F}_c$ changes the stress distribution, which indirectly alters particle trajectories. This influence is \emph{mediated by the elastic energy landscape}: the physics solver respects material properties, conservation laws, and stability constraints. Rather than applying an external force that directly opposes elastic restoring forces, we adjust the \emph{control signal} so that the physics itself produces trajectories more consistent with the visual target. An analogy is adjusting a puppet's internal skeleton rather than pushing its limbs with external forces.

The render gradient is computed at the end of episode $k$ and injected into the physics rollout of episode $k\!+\!1$. This one-episode delay is acceptable because the particle state changes smoothly between consecutive episodes.

\subsection{Chamfer-Guided Plasticity}
\label{sec:chamfer}

Even with control-space render injection, physics-based morphing faces a fundamental stability challenge: the elastic energy $\Psi(\mathbf{F}_e)$ has a global minimum at $\mathbf{F}_e = \mathbf{I}$ (the undeformed rest configuration). As the optimizer pushes particles toward the target, accumulated elastic energy creates increasingly strong restoring forces, causing \emph{rebound}, i.e., degradation of achieved shape quality as the system relaxes toward the source shape.

We address this through the multiplicative decomposition $\mathbf{F} = \mathbf{F}_e \mathbf{F}_p$~\cite{lee1969elastic}, where stress depends only on $\mathbf{F}_e = \mathbf{F} \mathbf{F}_p^{-1}$. Modifying $\mathbf{F}_p$ redefines the material's rest state without creating stress. If $\mathbf{F}_p$ is updated so that the current configuration becomes the new rest state, then $\mathbf{F}_e \approx \mathbf{I}$ and restoring forces point \emph{toward the target} rather than back toward the source.

We drive the plasticity update using Chamfer nearest-neighbor correspondences~\cite{fan2017point}. While Chamfer distance can exhibit structural biases toward convex regions~\cite{song2026structuralfailurechamferdistance}, spatial diffusion of the displacement field mitigates this effect in our setting. At each episode after a warmup of $k_0$ episodes, we: (1)~compute the nearest-neighbor displacement $\mathbf{d}_i = \mathbf{x}_i^\text{nn} - \mathbf{x}_i$ from each particle to the target surface via a KD-tree; (2)~smooth the displacement field using iterative KNN averaging ($k\!=\!64$, 3 iterations) to suppress noise; (3)~estimate the symmetric displacement Jacobian
\begin{equation}
\delta\mathbf{F}_p^{(i)} = \frac{1}{2}\left(\mathbf{J}_i + \mathbf{J}_i^\top\right), \quad \mathbf{J}_i = \frac{1}{|K_i|}\sum_{j \in K_i} \frac{(\mathbf{d}_j - \mathbf{d}_i)(\mathbf{x}_j - \mathbf{x}_i)^\top}{\|\mathbf{x}_j - \mathbf{x}_i\|^2}
\label{eq:dfp}
\end{equation}
and (4)~update multiplicatively: $\mathbf{F}_p \leftarrow (\mathbf{I} + \eta_i \, \delta\mathbf{F}_p) \, \mathbf{F}_p$, where $\eta_i$ is an adaptive rate scaled by local displacement magnitude. A damping factor and isochoric projection prevent unbounded accumulation.

The combined effect is progressive migration of the elastic equilibrium toward the target, converting physics from adversary to ally.

\subsection{Rendering Losses}
\label{sec:losses}

We render particles from 8 cameras and compute losses against target silhouettes. Our primary loss uses the signed distance transform~\cite{borgefors1986distance} of the target silhouette:
\begin{equation}
L_\text{DT} = \frac{1}{|\Omega|} \sum_{p \in \Omega} \alpha_p \cdot \text{DT}(p)
\label{eq:dt}
\end{equation}
where $\alpha_p$ is the predicted alpha and $\text{DT}(p)$ is positive outside and negative inside the target boundary. Unlike BCE, the DT gradient is proportional to boundary distance, providing a consistent directional ``pull'' that can be viewed as a \emph{projected Chamfer distance} on the 2D image plane.

We supplement DT with soft IoU ($L_\text{IoU} = 1 - \text{intersection}/\text{union}$) for global overlap and a depth loss for z-direction alignment. Adaptive gradient matching~\cite{yu2020gradient} normalizes auxiliary loss gradients to a target fraction of the DT gradient norm.

\subsection{Surface-Aware Rendering}
\label{sec:surface}

A volumetric particle representation contains many interior particles that do not contribute to rendered images. We identify surface particles via density-based reconstruction on a $64^3$ voxel grid with Gaussian-filtered density, followed by marching cubes~\cite{lorensen1998marching} surface extraction and distance thresholding. Only particles within a threshold distance of the reconstructed surface (typically 36\% of total) participate in the 3DGS rendering pass and receive render gradients. This focuses the gradient signal on visually relevant particles and reduces computation.

To further improve coverage, we augment the initial density-based mask with a Chamfer-based criterion: particles whose nearest-neighbor distance to the target surface falls below a threshold are progressively added to the render mask (union, never shrink). This ensures that particles approaching the target surface receive render gradients even if they were initially classified as interior.

%=========================================================================
\section{Experiments}
\label{sec:experiments}
%=========================================================================

\subsection{Setup}

We use a differentiable MPM implementation with fixed corotated elasticity~\cite{2012-FixedCoratedElasticty}. The simulation domain is $[-16, 16]^3$ with grid resolution $64^3$ ($\Delta x = 0.5$). Source shapes are sampled with shell-biased sampling (surface PPC\,5, interior PPC\,1), yielding approximately 490K particles. Time step $\Delta t = 1/240$\,s with $T\!=\!20$ timesteps per episode, 60 episodes total.

Eight cameras are placed in a ring configuration at $21^\circ$ elevation, rendering at $960 \times 540$. Gaussian size $\sigma_0$ is computed adaptively from KNN spacing. We evaluate on four target meshes: Spot (cow), Stanford Bunny, Bob (humanoid), and Utah Teapot. All morphings start from the same isosphere source. We report alpha MSE (silhouette error), Chamfer distance (3D shape accuracy), and physics loss reduction.

\subsection{Main Results}

% TABLE 1: Main comparison
\begin{table}[t]
\centering
\caption{Best alpha MSE ($\downarrow$) across methods and targets. \textbf{Bold}: best per target.}
\label{tab:main}
\begin{tabular}{lcccc}
\hline
Method & Spot & Bunny & Bob & Teapot \\
\hline
Physics-only & -- & -- & -- & -- \\
+ Control-space inj. & -- & -- & -- & -- \\
+ Chamfer plast. & -- & -- & -- & -- \\
Full (ours) & -- & -- & -- & -- \\
\hline
\end{tabular}
\end{table}

% FIGURE: Multi-target results
\begin{figure*}[t]
  \centering
  \includegraphics[width=0.95\linewidth]{figures/multi_target}
  \caption{Morphing results on four targets. Each row: source (isosphere), intermediate frames, and final result. Top to bottom: Spot, Bunny, Bob, Teapot. For visualization, Laplacian surface smoothing (4~iterations) is applied to the final particle positions to reduce compression artifacts; quantitative metrics are reported on unsmoothed results.}
  \label{fig:multi_target}
\end{figure*}

\subsection{Ablation: Position vs.\ Control-Space Injection}
\label{sec:ablation}

We compare three injection strategies on the Spot target over 60 episodes (Table~\ref{tab:ablation_inject}). Position injection ($\mathbf{g}^\mathbf{x} \neq \mathbf{0}$) performs \emph{worse} than no injection (0.1051 vs.\ 0.0956), confirming that the adversarial dynamic is actively harmful. Control-space injection ($\mathbf{g}^\mathbf{x} = \mathbf{0}$) achieves 27\% lower best alpha MSE with $3\times$ less rebound.

% TABLE 2: Injection ablation
\begin{table}[t]
\centering
\caption{Position vs.\ control-space injection on Spot (60 episodes).}
\label{tab:ablation_inject}
\begin{tabular}{lccc}
\hline
Strategy & Best $\alpha$ MSE & Final $\alpha$ MSE & Rebound \\
\hline
No injection & 0.0956 & 0.1118 & +17\% \\
Position inj. & 0.1051 & 0.1301 & +24\% \\
Control-space (ours) & \textbf{0.0697} & \textbf{0.0752} & \textbf{+8\%} \\
\hline
\end{tabular}
\end{table}

% FIGURE: Oscillation comparison
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/oscillation}
  \caption{Alpha MSE over 60 episodes on the Spot target. Position injection (orange) oscillates as elastic forces fight render gradients. Control-space injection (blue) converges stably. Physics-only baseline (gray) for reference.}
  \label{fig:oscillation}
\end{figure}

We track the injected render gradient norm $\|\partial L / \partial \mathbf{F}\|$ over episodes (Figure~\ref{fig:grad_tracking}). It decreases from 16.2 to $\sim$0.85, indicating progressive satisfaction of the rendering objective. The physics loss converges 97.9\% regardless of injection, confirming that the two objectives do not interfere.

% FIGURE: Gradient tracking
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/grad_tracking}
  \caption{Render gradient norm ($\|\partial L/\partial \mathbf{F}\|$) and physics control gradient (dFc) over episodes. The render gradient decreases as the visual objective is satisfied; dFc remains stable.}
  \label{fig:grad_tracking}
\end{figure}

\subsection{Ablation: Chamfer Plasticity}

Without plasticity, rebound reaches 33\% as elastic energy undoes the morphing. With Chamfer plasticity ($\eta\!=\!0.05$, $k_0\!=\!20$), the Chamfer distance decreases progressively (1.16 $\to$ 0.10) and rebound is reduced to 8\%. The physics loss trajectory is unaffected, as the $\mathbf{F}_p$ update does not introduce stress.

% FIGURE: Chamfer plasticity effect
\begin{figure}[t]
  \centering
  \includegraphics[width=\linewidth]{figures/chamfer_effect}
  \caption{Effect of Chamfer plasticity. \textbf{Left}: Chamfer distance decreases as $\mathbf{F}_p$ absorbs deformation. \textbf{Right}: alpha MSE with (solid) and without (dashed) plasticity.}
  \label{fig:chamfer_effect}
\end{figure}

\subsection{Ablation: DT vs.\ BCE Loss}

The DT loss achieves 4\% lower best alpha MSE than BCE (0.1645 vs.\ 0.1712) with smaller but more directionally coherent gradients (initial $\|\partial L / \partial \mathbf{F}\| = 16.2$ vs.\ 36.9). The improvement compounds with control-space injection and Chamfer plasticity.

\subsection{Timing}

Each episode takes approximately 4 minutes on a single GPU (NVIDIA RTX series), comprising physics rollout ($\sim$3 min), multi-view rendering and backward pass ($\sim$40 s), and Chamfer plasticity ($\sim$20 s). A full 60-episode run completes in approximately 4 hours.

%=========================================================================
\section{Discussion and Limitations}
\label{sec:discussion}
%=========================================================================

Several limitations remain. First, particles cannot be created or destroyed during simulation, limiting resolution at thin protrusions (e.g., ears, tails) where the source shape has insufficient particle density. Adaptive particle sampling~\cite{adams2007adaptively} could address this. Second, the surface mask is computed from the initial configuration and progressively augmented via Chamfer thresholding, but particles that migrate to the surface late in the morphing may still miss render gradients. Third, particle compression during large deformation produces surface wrinkling, a position-space artifact that covariance-based rendering cannot fully resolve. We apply Laplacian surface smoothing~\cite{taubin1995signal} for visualization; quantitative metrics use unsmoothed results.

%=========================================================================
\section{Conclusion}
\label{sec:conclusion}
%=========================================================================

We presented PhysMorph-GS, a framework for physics-based volumetric morphing guided by differentiable 3D Gaussian Splatting. By routing rendering supervision through the deformation gradient $\mathbf{F}$ rather than particle positions, we avoid the adversarial dynamic between elastic restoring forces and visual objectives. Chamfer-guided plasticity further stabilizes convergence by migrating the elastic rest configuration toward the target. Experiments on 20 source-target morphing pairs demonstrate consistent improvements over the physics-only baseline: 27\% lower silhouette error, $3\times$ less oscillation, and 97.9\% physics loss reduction. Future work includes adaptive particle redistribution for thin structures and dynamic surface masks with stability guarantees.

%-------------------------------------------------------------------------
\bibliographystyle{eg-alpha-doi}
\bibliography{egbibsample}

\end{document}
