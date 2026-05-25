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

<<<<<<< Updated upstream
=======
These end up simplifying the state fidelity and total coincidence rate equations to be as follows:

$$
\mathcal{F}_\ell = \frac{1}{4}\left[ 1 + \frac{3\mu_{\ell_S} \tau K_\ell}{\mu_{\ell_S} \tau K_\ell + \left(\mu_{\ell_S} \tau K_\ell + \frac{2\tau d_{\ell_A}}{\eta_{\ell_A}}\right)\left(\mu_{\ell_S} \tau K_\ell + \frac{2\tau d_{\ell_B}}{\eta_{\ell_B}}\right)} \right]
$$

$$
\mathcal{R}_\ell = \mu_{\ell_S}\tau K_\ell + \left(\mu_{\ell_S} \tau K_\ell + \frac{2\tau d_{\ell_A}}{\eta_{\ell_A}}\right)\left(\mu_{\ell_S} \tau K_\ell + \frac{2\tau d_{\ell_B}}{\eta_{\ell_B}}\right)
$$

We also follow [G. Vardoyan and S. Wehner, 2023 IEEE Int. Conf. Quantum Comp. Eng. (QCE), vol. 01 (2023), pp. 1238–1248.] by setting utility to be the sum of the log of the rates of each link, in order to ensure weaker links are not left behind by the optimizer. This utility equation is for each source and is added up for all sources to get the total network utility.

$$
\mathcal{U}_m = \sum_{(u_{\ell_A},u_{\ell_B})\in\mathcal{L}_m} \log_{10}\mathcal{R}_\ell
$$

Since we want to maximize the utility such that fidelity is above a fidelity threshold and that there are enough channels from the source, we can write this problem as a MINLP problem.

$$
(\hat{\mu}_m,\{\hat{K}_\ell\}) = \argmax_{\mu_m, \{K_\ell\}} \mathcal{U}_m
$$

subject to the constraints

$$
\mathcal{F}_\ell \ge f_\ell \qquad\text{and}\qquad \sum_{\ell} K_\ell \le K \qquad \forall\;(u_{\ell_A},u_{\ell_B})\in\mathcal{L}_m
$$

where $\mu_m\in(0,\infty)$ and $K_\ell\in\mathbb{N}_0$. 


In the pipeline, this MINLP problem is solved with the APOPT solver which splits the larger problem into simpler NLP subproblems. APOPT determines the frequency channel allocation very quickly, enabling the pipeline to be able to run many times.
>>>>>>> Stashed changes

**Frequency Scheduling**
Lastly, we use the CP-SAT solver to check if our network has interfering frequency channels. This algorithm is very efficient at scheduling due to its ability to elliminate combinations and quickly backtrack. If we are unable to find a solution, the pipeline backtracks to the next best combination we found in the routing phase.





---

## Detailed Mathematical Formulation

We summarize our complete global RSA problem formally as the following:

1. Begin with a graph consisting of $S+U$ vertices split into two types: $S$ sources $\{s_1,...,s_S\}=\mathcal{V}_S$ and $U$ users $\{u_1,...,u_U\}=\mathcal{V}_U$. These vertices are connected by $E$ undirected edges $\mathcal{E} = \{e_1,...,e_E\} \subseteq \{(v,w):v,w \in \mathcal{V}_S \cup \mathcal{V}_U,v \neq w \}$, each of which is characterized by an efficiency parameter $h_{e}\in(0,1]$.

2. The users wish to establish a maximum of $L\leq \lfloor U/2\rfloor$ pairwise, monogamous links described by the set $\{(u_{1_A},u_{1_B}),...,(u_{L_A},u_{L_B})\}=\mathcal{L}$, where each link $\ell\in[L]$ possesses an "Alice" $u_{\ell_A}$ and "Bob" $u_{\ell_B}$ and requests entanglement of fidelity exceeding some threshold $\mathcal{F}_\ell\geq f_\ell$.

3. Each source $s_m$ ($m\in[S]$) possesses $K$ pairs of entangled frequency bins $\mathcal{K}=\{k_{\pm 1},...,k_{\pm K}\}$, each produced at the same flux $\mu_m$. Under this definition, $|\mathcal{K}|=2K$. The set of links serviced by source $s_m$ is denoted by $\mathcal{L}_m\subset \mathcal{L}$.

4. The quantum RSA problem seeks to determine the following:
   
   a. Channel flux value $\mu_m$ for each source $m\in[S]$.
   
   b. Source assignment $\ell_S\in[S]$ and lightpaths to each user: let $\mathcal{P}_{\ell_A}\subseteq\mathcal{E}$ be the set of edges comprising a path from $s_{\ell_S}$ to $u_{\ell_A}$, and let $\mathcal{P}_{\ell_B}\subseteq\mathcal{E}$ be the set of edges comprising a path from $s_{\ell_S}$ to $u_{\ell_B}$, for each link $\ell\in[L]$. These lightpaths have efficiencies $\eta_{\ell_A}=\prod_{e\in \mathcal{P}_{\ell_A}} h_e$ and $\eta_{\ell_B}=\prod_{e\in \mathcal{P}_{\ell_B}} h_e$.
   
   c. Spectral assignment for each link $\ell\in[L]$, where $\mathcal{K}_{\ell_A}\subset\mathcal{K}$ bins are assigned to Alice and $\mathcal{K}_{\ell_B}\subset\mathcal{K}$ to Bob, which are energy-matched: i.e., for every $k_n\in\mathcal{K}_{\ell_A}$, it must be the case that $k_{-n}\in\mathcal{K}_{\ell_B}$; hence $|\mathcal{K}_{\ell_A}|=|\mathcal{K}_{\ell_B}|\equiv K_\ell$.

The goal is to maximize the chosen definition of network utility $\mathcal{U}$ subject to the following constraints:

<<<<<<< Updated upstream
The goal is to maximize the chosen definition of network utility $\cU$ subject to the following constraints:
\begin{enumerate}[label=(C\arabic*)]
\item \textbf{Fidelity:} $\cF_\ell\geq f_\ell \; \forall \; \ell\in[L]$.
\item \textbf{Frequency-bin availability:} $\sum_{(u_{\ell_A},u_{\ell_B})\in\cL_m} K_\ell \leq K \; \forall \; m\in[S]$.
\item \textbf{Contention-free distribution:} $\cK_{\ell_\alpha} \cap \cK_{\ell'_\beta}=\emptyset$ for all $\ell\neq\ell'$ and $\alpha,\beta\in\{A,B\}$ whose paths $\cP_{\ell_\alpha}$ and $\cP_{\ell'_\beta}$ share an edge.
\end{enumerate}
\end{enumerate}
=======
**(C1) Fidelity:** $\mathcal{F}_\ell\geq f_\ell \; \forall \; \ell\in[L]$

**(C2) Frequency-bin availability:** $\sum_{(u_{\ell_A},u_{\ell_B})\in\mathcal{L}_m} K_\ell \leq K \; \forall \; m\in[S]$

**(C3) Contention-free distribution:** $\mathcal{K}_{\ell_\alpha} \cap \mathcal{K}_{\ell'_\beta}=\emptyset$ for all $\ell\neq\ell'$ and $\alpha,\beta\in\{A,B\}$ whose paths $\mathcal{P}_{\ell_\alpha}$ and $\mathcal{P}_{\ell'_\beta}$ share an edge

## Setup Instructions

This project uses Python dependencies listed in `requirements.txt`. To install them, first create and activate a virtual environment.

### 1. Create a virtual environment

Run the following command from the project directory:

**Windows:**
```bash
python -m venv env
```

**Mac/Linux:**
```bash
python3 -m venv env
```

### 2. Activate the virtual environment

**Windows Command Prompt:**
```bash
env\Scripts\activate
```

**Windows PowerShell:**
```bash
.\env\Scripts\Activate.ps1
```

If PowerShell gives an error saying that running scripts is disabled on this system, run:
```bash
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

**Mac/Linux:**
```bash
source env/bin/activate
```

### 3. Install the required packages

After the virtual environment is activated, install the required libraries using:

**Windows:**
```bash
pip install -r requirements.txt
```

**Mac/Linux:**
```bash
python3 -m pip install -r requirements.txt
```
>>>>>>> Stashed changes
