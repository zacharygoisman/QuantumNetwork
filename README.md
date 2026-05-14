# QuantumNetwork
Research project for optimizing quantum network routing and spectrum allocation, with the eventual of being used for a physical system.

We have two goals at the moment:
1. Optimize channel allocation within a source to all lightpaths using said source. (Spectrum Allocation)
2. Find the best source and route for each pair of users. (Routing)

This may be expanded on later when considering a dynamic network or a physical system.

**Definitions**
For simplicity, we restrict our topologies to have two types of nodes: sources and users. A **source** is a node that generates and sends entangled photon pairs to **user** nodes. These photon pairs are sent in the form of frequency bin **channels** which all have identical **flux** from the same source, and due to energy conservation, frequency channel 1 has a complementary channel -1 sent to the second user in the pair. Rather than have additional switch nodes, each node is also capable of forwarding these channels to other nodes. Furthermore, I will be describing an entangled link between two user pairs as a **link**.

**How does this system work?**  
After generating the topology, the pipeline has three phases to determine optimal routing and allocation. These are routing (using Double Yen), spectrum allocation (using APOPT), and frequency scheduling (using CP-SAT).


**Routing**


**Spectrum Allocation**


**Frequency Scheduling**
Lastly, we use the CP-SAT solver to check if our network has interfering frequency channels. This algorithm is very efficient at scheduling due to its ability to elliminate combinations and quickly backtrack. If we are unable to find a solution, the pipeline backtracks to the next best combination we found in the routing phase.





-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
**Detailed mathematical formulation:**

We summarize our complete global RSA problem formally as the following:
\begin{enumerate}
\item Begin with a graph consisting of $S+U$ vertices split into two types: $S$ sources $\{s_1,...,s_S\}=\cV_S$ and $U$ users $\{u_1,...,u_U\}=\cV_U$. These vertices are connected by $E$ undirected edges $\cE = \{e_1,...,e_E\} \subseteq \{(v,w):v,w \in \cV_S \cup \cV_U,v \neq w \}$, each of which is characterized by an efficiency parameter $h_{e}\in(0,1]$.

\item The users wish to establish a maximum of $L\leq \lfloor U/2\rfloor$ pairwise, monogamous links described by the set $\{(u_{1_A},u_{1_B}),...,(u_{L_A},u_{L_B})\}=\cL$, where each link $\ell\in[L]$ possesses an ``Alice'' $u_{\ell_A}$ and ``Bob'' $u_{\ell_B}$ and requests entanglement of fidelity exceeding some threshold $\cF_\ell\geq f_\ell$.

\item Each source $s_m$ ($m\in[S]$) possesses $K$ pairs of entangled frequency bins $\cK=\{k_{\pm 1},...,k_{\pm K}\}$, each produced at the same flux $\mu_m$. Under this definition, $|\cK|=2K$. The set of links serviced by source $s_m$ is denoted by $\cL_m\subset \cL$.

\item The quantum RSA problem seeks to determine the following:
\begin{enumerate}
\item Channel flux value $\mu_m$ for each source $m\in[S]$.

\item Source assignment $\ell_S\in[S]$ and lightpaths to each user: let $\cP_{\ell_A}\subseteq\mathcal E$ be the set of edges comprising a path from $s_{\ell_S}$ to $u_{\ell_A}$, and let $\cP_{\ell_B}\subseteq\mathcal E$ be the set of edges comprising a path from $s_{\ell_S}$ to $u_{\ell_B}$, for each link $\ell\in[L]$.
These lightpaths have efficiencies
$\eta_{\ell_A}=\prod_{e\in \cP_{\ell_A}} h_e$ and $\eta_{\ell_B}=\prod_{e\in \cP_{\ell_B}} h_e$.

\item Spectral assignment for each link $\ell\in[L]$, where $\cK_{\ell_A}\subset\cK$ bins are assigned to Alice and $\cK_{\ell_B}\subset\cK$ to Bob, which are energy-matched: i.e., for every $k_n\in\cK_{\ell_A}$, it must be the case that $k_{-n}\in\cK_{\ell_B}$; hence $|\cK_{\ell_A}|=|\cK_{\ell_B}|\equiv K_\ell$.
\end{enumerate}

The goal is to maximize the chosen definition of network utility $\cU$ subject to the following constraints:
\begin{enumerate}[label=(C\arabic*)]
\item \textbf{Fidelity:} $\cF_\ell\geq f_\ell \; \forall \; \ell\in[L]$.
\item \textbf{Frequency-bin availability:} $\sum_{(u_{\ell_A},u_{\ell_B})\in\cL_m} K_\ell \leq K \; \forall \; m\in[S]$.
\item \textbf{Contention-free distribution:} $\cK_{\ell_\alpha} \cap \cK_{\ell'_\beta}=\emptyset$ for all $\ell\neq\ell'$ and $\alpha,\beta\in\{A,B\}$ whose paths $\cP_{\ell_\alpha}$ and $\cP_{\ell'_\beta}$ share an edge.
\end{enumerate}
\end{enumerate}
